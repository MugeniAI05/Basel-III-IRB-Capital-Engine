"""
Stress Testing & CCAR-style Scenario Analysis.

─────────────────────────────────────────────────────────────────────────────
CONCEPTUAL FRAMEWORK — Capital Adequacy vs Conditional Expected Loss

There are two distinct but related questions in stress testing:

  (A) CONDITIONAL EXPECTED LOSS under scenario Y:
      E[Loss | Y] = Σ PD(Y) × LGD × EAD
      This is the *expected* loss if the economy realises state Y.
      It is NOT directly comparable to IRB capital without qualification.

  (B) CAPITAL ADEQUACY at a given confidence level:
      IRB capital = UL at 99.9% = E[Loss | Y=-3.09] − EL
      The model is "adequately capitalised" if IRB capital ≥ UL(99.9%).
      By IRB construction, this holds exactly at Y = Φ⁻¹(0.001) ≈ -3.09.

WHY THE DISTINCTION MATTERS:
  Comparing IRB capital to E[Loss | Y=-2] as a "coverage ratio" is a valid
  stress test metric — it tells you how many times the capital buffer
  exceeds the expected loss under a severe-but-not-tail scenario.
  But it is NOT saying "the bank is undercapitalised at Y=-2". Capital
  was never designed to cover E[Loss | Y=-2]; it was designed to cover
  the 99.9th percentile unexpected loss. The shortfall at Y=-3.09 is the
  by-construction binding constraint.

  In CCAR/DFAST, regulators run stressed loss projections (similar to
  E[Loss | Y_CCAR]) against available capital resources (not IRB capital),
  making the comparison operationally well-defined.

  This module explicitly labels metrics to reflect both perspectives,
  so the output is defensible to a bank reviewer.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from src.models.vasicek import (
    conditional_pd,
    portfolio_loss_rate,
    SCENARIOS,
    VasicekScenario,
    simulate_loss_distribution,
)


def _y_to_quantile(y: float) -> float:
    """Map systemic factor Y to its CDF percentile: Φ(Y)."""
    return float(norm.cdf(y))


def run_scenario(
    df: pd.DataFrame,
    scenario: VasicekScenario,
) -> dict:
    """
    Compute portfolio stress metrics under a single Vasicek scenario.

    Produces two clearly-labelled capital adequacy perspectives:

    (1) Conditional-EL coverage:
        cond_el_coverage = IRB_capital / E[Loss | Y]
        Measures how many times the standing capital buffer exceeds the
        *expected* loss under this scenario. A coverage > 1 means the
        capital buffer absorbs the scenario's expected loss in full.
        This is a useful stress severity gauge but NOT the IRB adequacy test.

    (2) IRB adequacy test (tail risk perspective):
        The model is formally adequate at Y = Φ⁻¹(0.001) ≈ -3.09.
        For other Y values, we report whether the scenario falls within
        the design envelope of the IRB model (i.e. whether Y > Φ⁻¹(0.001)).

    Args:
        df: Loan DataFrame from generator.generate_portfolio().
        scenario: VasicekScenario with systemic_factor Y.

    Returns:
        dict with fully-labelled stress metrics.
    """
    pds = df["pd"].values
    lgds = df["lgd"].values
    eads = df["ead"].values
    corrs = df["correlation"].values

    total_ead = eads.sum()
    total_irb_capital = df["capital"].sum()
    total_el = df["el"].sum()

    # Conditional PDs under this scenario
    stressed_pds = conditional_pd(pds, corrs, scenario.systemic_factor)

    # Conditional expected loss: E[Loss | Y]
    cond_expected_loss = (stressed_pds * lgds * eads).sum()
    cond_el_rate = cond_expected_loss / total_ead

    # Unexpected loss component: E[Loss|Y] − EL
    # This is the portion that capital is designed to absorb
    cond_unexpected_loss = max(cond_expected_loss - total_el, 0.0)

    # ── Perspective 1: Conditional-EL coverage ratio ──────────────────────────
    # Capital vs E[Loss|Y]: useful stress gauge, NOT the formal IRB adequacy test
    cond_el_coverage = total_irb_capital / cond_expected_loss if cond_expected_loss > 0 else np.inf
    cond_el_surplus = total_irb_capital - cond_expected_loss

    # ── Perspective 2: IRB tail adequacy ─────────────────────────────────────
    # The IRB model is designed to be exactly adequate at the 99.9th percentile.
    # Map Y to its quantile: scenarios inside the 99.9% envelope are "within design".
    scenario_quantile = _y_to_quantile(scenario.systemic_factor)
    irb_design_quantile = 0.999  # Basel confidence level
    within_irb_design = scenario_quantile >= (1.0 - irb_design_quantile)
    # i.e. scenario is within design if Y > Φ⁻¹(0.001) ≈ -3.09

    return {
        # Scenario identity
        "scenario":                     scenario.name,
        "systemic_factor":              scenario.systemic_factor,
        "scenario_quantile":            scenario_quantile,
        "description":                  scenario.description,

        # Conditional expected loss
        "cond_expected_loss":           cond_expected_loss,
        "cond_el_rate":                 cond_el_rate,
        "avg_stressed_pd":              (stressed_pds * eads).sum() / total_ead,

        # Capital stack
        "irb_capital":                  total_irb_capital,
        "standing_el":                  total_el,
        "cond_unexpected_loss":         cond_unexpected_loss,

        # Perspective 1: Conditional-EL coverage
        # Interpretation: "Capital covers scenario EL by X×"
        # NOT a formal adequacy test — just a stress severity metric.
        "cond_el_coverage":             cond_el_coverage,
        "cond_el_surplus":              cond_el_surplus,
        "cond_el_covered":              cond_el_surplus >= 0,

        # Perspective 2: IRB tail adequacy
        # Formally adequate by construction at the 99.9th percentile.
        "within_irb_design_envelope":   within_irb_design,
        "irb_design_note": (
            "Within IRB 99.9% design envelope"
            if within_irb_design else
            "Beyond IRB 99.9% design envelope — capital shortfall is expected by construction"
        ),
    }


def run_all_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all standard Vasicek scenarios against the portfolio.

    Returns:
        DataFrame with one row per scenario, including both capital
        adequacy perspectives.
    """
    results = [run_scenario(df, s) for s in SCENARIOS.values()]
    return pd.DataFrame(results)


def stress_sweep(
    df: pd.DataFrame,
    y_min: float = -3.5,
    y_max: float = 1.0,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Sweep the systemic factor Y and compute metrics at each point.

    Returns:
        DataFrame with: Y, quantile, cond_el_rate, capital_rate,
        cond_el_surplus_rate, within_design.
    """
    pds = df["pd"].values
    lgds = df["lgd"].values
    eads = df["ead"].values
    corrs = df["correlation"].values
    total_ead = eads.sum()
    irb_capital_rate = df["capital"].sum() / total_ead

    Y_vals = np.linspace(y_min, y_max, n_points)
    cond_el_rates = [portfolio_loss_rate(pds, lgds, eads, corrs, y) for y in Y_vals]
    quantiles = [_y_to_quantile(y) for y in Y_vals]

    return pd.DataFrame({
        "Y":                    Y_vals,
        "quantile":             quantiles,
        "cond_el_rate":         cond_el_rates,
        "capital_rate":         irb_capital_rate,
        "cond_el_surplus_rate": irb_capital_rate - np.array(cond_el_rates),
        "within_design":        [q >= 0.001 for q in quantiles],
    })


def sensitivity_capital_vs_pd(
    lgd: float = 0.45,
    correlation: float | None = None,
    confidence_level: float = 0.999,
    maturity: float = 2.5,
    n_points: int = 50,
) -> pd.DataFrame:
    """Sensitivity: capital ratio K (with and without MA) vs PD."""
    from src.models.irb_capital import irb_capital_ratio, basel_correlation

    pds = np.linspace(0.001, 0.10, n_points)
    corrs = basel_correlation(pds) if correlation is None else np.full(n_points, correlation)

    K_with_ma = irb_capital_ratio(pds, lgd, corrs, confidence_level, maturity, apply_maturity_adjustment=True)
    K_no_ma   = irb_capital_ratio(pds, lgd, corrs, confidence_level, maturity, apply_maturity_adjustment=False)
    el = pds * lgd

    return pd.DataFrame({
        "pd": pds, "K": K_with_ma, "K_no_ma": K_no_ma,
        "el_rate": el, "correlation": corrs,
    })


def sensitivity_capital_vs_correlation(
    pd_values: list[float] | None = None,
    lgd: float = 0.45,
    confidence_level: float = 0.999,
    maturity: float = 2.5,
) -> pd.DataFrame:
    """Sensitivity: capital ratio K vs asset correlation R."""
    from src.models.irb_capital import irb_capital_ratio

    if pd_values is None:
        pd_values = [0.01, 0.03, 0.07]

    correlations = np.linspace(0.02, 0.30, 50)
    result = pd.DataFrame({"R": correlations})
    for pd_val in pd_values:
        col = f"K_pd{int(pd_val*100)}pct"
        result[col] = irb_capital_ratio(pd_val, lgd, correlations, confidence_level, maturity)
    return result


def sensitivity_capital_vs_confidence(
    pd_values: list[float] | None = None,
    lgd: float = 0.45,
    correlation: float = 0.15,
    maturity: float = 2.5,
) -> pd.DataFrame:
    """Sensitivity: capital ratio K vs confidence level."""
    from src.models.irb_capital import irb_capital_ratio

    if pd_values is None:
        pd_values = [0.01, 0.03]

    conf_levels = np.linspace(0.990, 0.9999, 50)
    result = pd.DataFrame({"confidence_level": conf_levels})
    for pd_val in pd_values:
        col = f"K_pd{int(pd_val*100)}pct"
        result[col] = irb_capital_ratio(pd_val, lgd, correlation, conf_levels, maturity)
    return result


def sensitivity_capital_vs_maturity(
    pd_values: list[float] | None = None,
    lgd: float = 0.45,
    correlation: float | None = None,
) -> pd.DataFrame:
    """
    Sensitivity: capital ratio K vs effective maturity M.

    Shows the incremental capital cost of longer-dated exposures,
    driven entirely by the maturity adjustment MA(PD, M).
    """
    from src.models.irb_capital import irb_capital_ratio, basel_correlation

    if pd_values is None:
        pd_values = [0.005, 0.02, 0.07]

    maturities = np.linspace(1.0, 5.0, 50)
    result = pd.DataFrame({"maturity": maturities})
    for pd_val in pd_values:
        corr = float(basel_correlation(pd_val)) if correlation is None else correlation
        col = f"K_pd{int(pd_val*1000):04d}bps"
        result[col] = irb_capital_ratio(pd_val, lgd, corr, 0.999, maturities)
    return result
