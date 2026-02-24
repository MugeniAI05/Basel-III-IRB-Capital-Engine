"""
Basel III IRB Capital Engine — CLI Entry Point

Run with:
    python main.py
"""

import sys
import numpy as np
from src.portfolio.generator import generate_portfolio
from src.portfolio.analytics import portfolio_summary, industry_breakdown
from src.stress.scenarios import run_all_scenarios


def fmt_pct(v, d=3):
    return f"{v*100:.{d}f}%"

def fmt_mn(v):
    return f"${v/1e6:.1f}M"

def fmt_bn(v):
    return f"${v/1e9:.2f}B"

def separator(char="═", width=70):
    print(char * width)

def header(title):
    separator()
    print(f"  {title}")
    separator()

def section(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def main():
    print("\n")
    header("🏦  BASEL III IRB CAPITAL ENGINE")
    print("  PD/LGD Modeling → Portfolio Capital Under Stress")
    print("  Vasicek Single-Factor Model · 10,000 Synthetic Corporate Loans")
    print()

    # ── Step 1: Generate Portfolio ────────────────────────────────────────────
    section("STEP 1 — PORTFOLIO GENERATION")
    print("  Generating 10,000 synthetic corporate loans...")
    df = generate_portfolio(n_loans=10_000, seed=42)
    print(f"  ✓ Portfolio generated: {len(df):,} loans across {df['industry'].nunique()} industries")

    # ── Step 2: Expected Loss ─────────────────────────────────────────────────
    section("STEP 2 — EXPECTED LOSS (EL = PD × LGD × EAD)")
    stats = portfolio_summary(df)

    print(f"  {'Total EAD':<30} {fmt_bn(stats['total_ead'])}")
    print(f"  {'Total Expected Loss':<30} {fmt_mn(stats['total_el'])}")
    print(f"  {'EL Rate':<30} {fmt_pct(stats['el_rate'])}")
    print(f"  {'Wtd Avg PD':<30} {fmt_pct(stats['wtd_avg_pd'])}")
    print(f"  {'Wtd Avg LGD':<30} {fmt_pct(stats['wtd_avg_lgd'], d=1)}")
    print()
    print("  → EL is provisions territory, NOT capital.")
    print(f"  → Capital is {stats['capital_to_el_ratio']:.1f}× larger than EL (tail risk buffer).")

    # ── Step 3: Vasicek ───────────────────────────────────────────────────────
    section("STEP 3 — VASICEK SINGLE-FACTOR MODEL")
    from src.models.vasicek import conditional_pd, SCENARIOS
    example_pd, example_r = 0.02, 0.15
    print(f"  Example loan: PD={fmt_pct(example_pd)}, R={fmt_pct(example_r, d=0)}")
    print()
    for key, scenario in SCENARIOS.items():
        cond = conditional_pd(example_pd, example_r, scenario.systemic_factor)
        print(f"  {scenario.name:<25} Y={scenario.systemic_factor:>5.1f}  →  Conditional PD = {fmt_pct(cond)}")
    print()
    print("  → As Y declines, conditional PD spikes exponentially.")

    # ── Step 4: IRB Capital ───────────────────────────────────────────────────
    section("STEP 4 — IRB CAPITAL FORMULA")
    print(f"  {'Total IRB Capital':<30} {fmt_mn(stats['total_capital'])}")
    print(f"  {'Total RWA':<30} {fmt_bn(stats['total_rwa'])}")
    print(f"  {'Required Capital (8%×RWA)':<30} {fmt_mn(stats['total_req_capital'])}")
    print(f"  {'Portfolio Capital Rate':<30} {fmt_pct(stats['capital_rate'])}")
    print(f"  {'RWA Density':<30} {fmt_pct(stats['rwa_density'], d=1)}")

    # ── Step 5: Industry Breakdown ────────────────────────────────────────────
    section("STEP 5 — INDUSTRY BREAKDOWN")
    ind = industry_breakdown(df)
    print(f"  {'Industry':<16} {'EAD($B)':>8} {'Avg PD':>8} {'Avg LGD':>8} {'Cap Rate':>10} {'EL Rate':>8}")
    print("  " + "─" * 60)
    for name, row in ind.iterrows():
        print(
            f"  {name:<16} "
            f"{row['total_ead']/1e9:>7.2f}B "
            f"{row['avg_pd']*100:>7.3f}% "
            f"{row['avg_lgd']*100:>7.1f}% "
            f"{row['capital_rate']*100:>9.3f}% "
            f"{row['el_rate']*100:>7.3f}%"
        )

    # ── Step 6: Stress Scenarios ──────────────────────────────────────────────
    section("STEP 6 — STRESS SCENARIOS")
    scenario_results = run_all_scenarios(df)

    print(f"  {'Scenario':<25} {'Y':>5} {'Cond EL Rate':>14} {'Coverage':>10} {'Design':>22}")
    print("  " + "─" * 80)
    for _, row in scenario_results.iterrows():
        within = row["within_irb_design_envelope"]
        design = "✓ Within 99.9% envelope" if within else "⚠ Beyond design envelope"
        print(
            f"  {row['scenario']:<25} "
            f"{row['systemic_factor']:>5.1f} "
            f"{row['cond_el_rate']*100:>13.4f}% "
            f"{row['cond_el_coverage']:>9.3f}× "
            f"{design}"
        )

    # ── Step 7: Model Risk Summary ─────────────────────────────────────────────
    section("STEP 7 — KEY MODEL RISKS")
    risks = [
        ("HIGH",   "Gaussian copula underestimates tail dependence"),
        ("HIGH",   "Basel correlations (12-24%) vs crisis realised (50-80%)"),
        ("MEDIUM", "TTC vs PIT PD tension — procyclicality risk"),
        ("MEDIUM", "Single-factor model misses sector contagion"),
        ("LOW",    "LGD estimation uncertainty under novel stress"),
    ]
    for severity, risk in risks:
        icon = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
        print(f"  {icon} [{severity:<6}] {risk}")

    separator()
    print("  ✅ Analysis complete. Run `streamlit run app.py` for the interactive dashboard.")
    separator()
    print()


if __name__ == "__main__":
    main()
