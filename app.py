"""
Basel III IRB Capital Engine — Streamlit Dashboard

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd

from src.portfolio.generator import generate_portfolio
from src.portfolio.analytics import portfolio_summary, industry_breakdown, pd_distribution_bins
from src.models.irb_capital import irb_capital_requirement, irb_capital_ratio, basel_correlation
from src.models.vasicek import conditional_pd, SCENARIOS
from src.stress.scenarios import (
    run_all_scenarios,
    stress_sweep,
    sensitivity_capital_vs_pd,
    sensitivity_capital_vs_correlation,
    sensitivity_capital_vs_confidence,
    run_scenario,
)
from src.visualization.charts import (
    chart_pd_distribution,
    chart_industry_capital,
    chart_el_vs_capital,
    chart_conditional_pd_vs_y,
    chart_capital_vs_pd,
    chart_capital_vs_correlation,
    chart_capital_vs_confidence,
    chart_stress_sweep,
    chart_scenario_comparison,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Basel III IRB Capital Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0c10; }
    .metric-label { font-size: 11px !important; color: #a0aec0 !important; }
    .metric-value { font-size: 20px !important; font-family: 'IBM Plex Mono', monospace !important; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: #e2e8f0 !important; }
    h1 { color: #e2e8f0 !important; opacity: 1 !important; }
    h2, h3 { color: #cbd5e0 !important; opacity: 1 !important; }
    p, .stMarkdown { color: #a0aec0 !important; }
    div[data-testid="stSidebarContent"] { background-color: #0d1117; }
    div[data-testid="stSidebarContent"] p, div[data-testid="stSidebarContent"] .stMarkdown { color: #a0aec0 !important; }
    div[data-testid="stSidebarContent"] h2 { color: #e2e8f0 !important; opacity: 1 !important; }
    .stNumberInput label, .stSlider label, .stSelectbox label { color: #a0aec0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data (cached) ─────────────────────────────────────────────────────────────
@st.cache_data
def load_portfolio():
    df = generate_portfolio(n_loans=10_000, seed=42)
    summary = portfolio_summary(df)
    industry = industry_breakdown(df)
    hist = pd_distribution_bins(df)
    return df, summary, industry, hist


df, summary, industry_df, hist_df = load_portfolio()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/Basel%20III-IRB%20Capital%20Engine-00c8ff?style=flat-square", width=250)
st.sidebar.markdown("## Navigation")
section = st.sidebar.radio(
    "",
    [
        "01 · Portfolio Overview",
        "02 · Expected Loss",
        "03 · Vasicek Model",
        "04 · IRB Capital Formula",
        "05 · Sensitivity Analysis",
        "06 · Stress Scenarios",
        "07 · Model Risk",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Portfolio Stats**
- Loans: `{summary['n_loans']:,}`
- EAD: `${summary['total_ead']/1e9:.2f}B`
- IRB Capital: `${summary['total_capital']/1e6:.0f}M`
- EL Rate: `{summary['el_rate']*100:.3f}%`
""")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Basel III IRB Capital Engine")
st.markdown(
    "**PD/LGD Modeling → Portfolio Capital Under Stress** · "
    "10,000 synthetic corporate loans · Vasicek single-factor model · "
    "Full AIRB capital stack"
)
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if section == "01 · Portfolio Overview":
    st.header("Portfolio Overview")
    st.markdown("10,000 synthetic corporate loans generated across 8 industries with realistic PD/LGD/EAD heterogeneity.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total EAD", f"${summary['total_ead']/1e9:.2f}B")
    col2.metric("Loan Count", f"{summary['n_loans']:,}")
    col3.metric("Wtd Avg PD", f"{summary['wtd_avg_pd']*100:.3f}%")
    col4.metric("Wtd Avg LGD", f"{summary['wtd_avg_lgd']*100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_pd_distribution(hist_df), use_container_width=True)
    with c2:
        st.plotly_chart(chart_industry_capital(industry_df), use_container_width=True)

    st.subheader("Industry Breakdown")
    display = industry_df[["n_loans", "total_ead", "avg_pd", "avg_lgd", "capital_rate", "el_rate"]].copy()
    display["total_ead"] = (display["total_ead"] / 1e9).round(2)
    display["avg_pd"] = (display["avg_pd"] * 100).round(3)
    display["avg_lgd"] = (display["avg_lgd"] * 100).round(1)
    display["capital_rate"] = (display["capital_rate"] * 100).round(3)
    display["el_rate"] = (display["el_rate"] * 100).round(3)
    display.columns = ["Loans", "EAD ($B)", "Avg PD (%)", "Avg LGD (%)", "Capital Rate (%)", "EL Rate (%)"]
    st.dataframe(display, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPECTED LOSS
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "02 · Expected Loss":
    st.header("Expected Loss: EL = PD × LGD × EAD")
    st.markdown("""
    **EL is a statistical average** — the loss that will occur on average each year.
    It should be covered by loan spreads and provisions, **not capital**.
    Capital protects against *unexpected* losses: the gap between EL and the 99.9th-percentile loss.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total EL", f"${summary['total_el']/1e6:.1f}M", help="Annual expected loss")
    col2.metric("EL Rate", f"{summary['el_rate']*100:.3f}%")
    col3.metric("IRB Capital", f"${summary['total_capital']/1e6:.0f}M")
    col4.metric("Capital / EL", f"{summary['capital_to_el_ratio']:.1f}×", help="How many times larger capital is vs EL")

    st.plotly_chart(chart_el_vs_capital(industry_df), use_container_width=True)

    st.info(
        f"EL (${summary['total_el']/1e6:.1f}M) is **{summary['capital_to_el_ratio']:.1f}×** smaller than "
        f"IRB capital (${summary['total_capital']/1e6:.0f}M). "
        "This ratio illustrates the buffer required for unexpected tail losses at the 99.9% confidence level."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VASICEK
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "03 · Vasicek Model":
    st.header("Vasicek Single-Factor Model")
    st.markdown(r"""
    Asset value: $A_i = \sqrt{R} \cdot Y + \sqrt{1-R} \cdot \varepsilon_i$

    Conditional PD: $\text{PD}(Y) = \Phi\!\left[\frac{\Phi^{-1}(\text{PD}) - \sqrt{R} \cdot Y}{\sqrt{1-R}}\right]$

    When **Y → −∞** (systemic crisis), conditional PDs spike across all obligors simultaneously.
    """)

    c1, c2 = st.columns([1, 2])
    with c1:
        pd_val = st.slider("Unconditional PD", 0.001, 0.10, 0.02, 0.001, format="%.3f")
        R_val = st.slider("Asset Correlation (R)", 0.02, 0.30, 0.15, 0.01, format="%.2f")

        st.markdown("**Conditional PDs:**")
        for y, label in [(0, "Y = 0 (Baseline)"), (-1, "Y = −1 (Mild)"), (-2, "Y = −2 (Severe)"), (-3, "Y = −3 (GFC)")]:
            cond = conditional_pd(pd_val, R_val, y)
            st.metric(label, f"{cond*100:.3f}%", delta=f"{(cond-pd_val)*100:+.3f}pp")

    with c2:
        st.plotly_chart(chart_conditional_pd_vs_y(pd_val, R_val), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — IRB FORMULA
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "04 · IRB Capital Formula":
    st.header("IRB Capital Formula")
    st.markdown(r"""
    $$K = \text{LGD} \cdot \Phi\!\left[\frac{\Phi^{-1}(\text{PD}) + \sqrt{R} \cdot \Phi^{-1}(0.999)}{\sqrt{1-R}}\right] - \text{PD} \cdot \text{LGD}$$

    Then: $\text{Capital} = \text{EAD} \times K$ · $\text{RWA} = 12.5 \times \text{Capital}$ · $\text{Req. Capital} = 8\% \times \text{RWA}$
    """)

    c1, c2 = st.columns([1, 2])
    with c1:
        pd_i = st.slider("PD", 0.001, 0.10, 0.02, 0.001, format="%.3f")
        lgd_i = st.slider("LGD", 0.10, 0.80, 0.45, 0.01, format="%.2f")
        R_i = st.slider("Correlation (R)", 0.02, 0.30, 0.15, 0.01, format="%.2f")
        conf_i = st.slider("Confidence Level", 0.990, 0.9999, 0.999, 0.0001, format="%.4f")
        ead_i = st.number_input("EAD ($M)", value=10.0, step=1.0) * 1e6

        result = irb_capital_requirement(pd_i, lgd_i, ead_i, R_i, conf_i)
        st.metric("K (Capital Ratio)", f"{result['K']*100:.3f}%")
        st.metric("Capital", f"${result['capital']/1e6:.3f}M")
        st.metric("RWA", f"${result['rwa']/1e6:.2f}M")
        st.metric("Required Capital (8%×RWA)", f"${result['req_capital']/1e6:.3f}M")
        st.metric("Expected Loss", f"${result['el']/1e6:.3f}M")

    with c2:
        sens_df = sensitivity_capital_vs_pd(lgd=lgd_i, confidence_level=conf_i)
        st.plotly_chart(chart_capital_vs_pd(sens_df), use_container_width=True)

    st.info(
        "Notice K is **non-linear in PD**. For very low-PD (investment grade) borrowers, "
        "high asset correlation R means a large fraction of their risk is systematic — driving K up. "
        "This is why IG corporate loans carry non-trivial RWA despite low standalone PDs."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "05 · Sensitivity Analysis":
    st.header("Sensitivity Analysis")
    st.markdown("How capital requirements respond to changes in PD, LGD, correlation, and confidence level.")

    tab1, tab2, tab3 = st.tabs(["Capital vs Correlation", "TTC vs Downturn LGD", "Capital vs Confidence"])

    with tab1:
        corr_df = sensitivity_capital_vs_correlation()
        st.plotly_chart(chart_capital_vs_correlation(corr_df), use_container_width=True)
        st.markdown(
            "**Key insight**: Higher correlation amplifies systematic risk. "
            "Low-PD borrowers (IG) are most sensitive to correlation because they have the most to lose "
            "when the systematic factor deteriorates."
        )

    with tab2:
        lgd_base = st.slider("Base LGD", 0.20, 0.60, 0.45, 0.05, format="%.2f")
        lgd_shock = st.slider("Downturn LGD Shock (+pp)", 0.05, 0.30, 0.20, 0.05, format="%.2f")

        pds = np.linspace(0.001, 0.10, 50)
        R = basel_correlation(pds)
        K_base = irb_capital_ratio(pds, lgd_base, R)
        K_down = irb_capital_ratio(pds, np.minimum(lgd_base + lgd_shock, 0.99), R)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pds*100, y=K_base*100, name=f"TTC LGD = {lgd_base:.0%}",
                                  line=dict(color="#7fff6b", width=2)))
        fig.add_trace(go.Scatter(x=pds*100, y=K_down*100, name=f"Downturn LGD = {lgd_base+lgd_shock:.0%}",
                                  line=dict(color="#ff6b35", width=2)))
        fig.update_layout(title="TTC vs Downturn LGD Impact on Capital",
                           xaxis_title="PD (%)", yaxis_title="Capital K (%)",
                           paper_bgcolor="#0a0c10", plot_bgcolor="#111318",
                           font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("LGD enters K **linearly** — a 20pp downturn shock adds ~20% to capital requirements across all PD levels.")

    with tab3:
        conf_df = sensitivity_capital_vs_confidence()
        st.plotly_chart(chart_capital_vs_confidence(conf_df), use_container_width=True)
        st.markdown(
            "Moving from Basel's 99.9% to CCAR-level 99.5%–99.9% severities materially increases capital, "
            "especially for investment-grade (low PD) borrowers where the tail is fatter relative to EL."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STRESS SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "06 · Stress Scenarios":
    st.header("Stress Scenarios — CCAR Territory")
    st.markdown(
        "Shocks the systemic factor Y to simulate portfolio-wide default clustering. "
        "At Y = −2 (1-in-44 year event) the stressed loss rate approaches the IRB capital buffer."
    )

    scenario_df = run_all_scenarios(df)

    col1, col2, col3, col4 = st.columns(4)
    for col, (_, row) in zip([col1, col2, col3, col4], scenario_df.iterrows()):
        coverage = row["cond_el_coverage"]
        col.metric(
            row["scenario"],
            f"{row['cond_el_rate']*100:.3f}%",
            delta=f"Cond-EL Coverage: {coverage:.2f}×",
            delta_color="normal" if coverage >= 1 else "inverse",
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_scenario_comparison(scenario_df), use_container_width=True)
    with c2:
        sweep_df = stress_sweep(df)
        st.plotly_chart(chart_stress_sweep(sweep_df), use_container_width=True)

    st.subheader("Interactive Scenario")
    y_custom = st.slider("Custom Systemic Factor Y", -3.5, 1.0, -2.0, 0.1)
    from src.models.vasicek import VasicekScenario
    custom = VasicekScenario("Custom", y_custom, f"User-defined Y = {y_custom}")
    r = run_scenario(df, custom)

    col1, col2, col3 = st.columns(3)
    col1.metric("Cond. EL Rate", f"{r['cond_el_rate']*100:.3f}%",
                help="E[Loss|Y] / EAD — conditional expected loss rate under this scenario")
    col2.metric("Cond-EL Coverage", f"{r['cond_el_coverage']:.2f}×",
                help="IRB Capital / E[Loss|Y] — how many times capital covers the scenario's expected loss")
    col3.metric(
        "IRB Design Envelope",
        "✓ Within 99.9%" if r["within_irb_design_envelope"] else "⚠ Beyond 99.9%",
        help="Whether this scenario falls within the IRB model's 99.9% design confidence level",
    )

    st.info(
        "**Note on capital adequacy framing:** The 'Cond-EL Coverage' ratio measures how many times "
        "standing IRB capital covers E[Loss|Y] for this scenario — a useful stress severity gauge. "
        "It is not the same as the formal IRB adequacy test, which is satisfied by construction "
        "at Y = Φ⁻¹(0.001) ≈ −3.09 (the 99.9th percentile).",
        icon="ℹ️",
    )

    st.dataframe(
        scenario_df[[
            "scenario", "systemic_factor", "scenario_quantile",
            "cond_el_rate", "cond_el_coverage", "within_irb_design_envelope"
        ]]
        .assign(
            cond_el_rate=lambda d: (d.cond_el_rate * 100).round(4),
            cond_el_coverage=lambda d: d.cond_el_coverage.round(3),
            scenario_quantile=lambda d: (d.scenario_quantile * 100).round(4),
        )
        .rename(columns={
            "scenario":                   "Scenario",
            "systemic_factor":            "Y",
            "scenario_quantile":          "Percentile (%)",
            "cond_el_rate":               "Cond. EL Rate (%)",
            "cond_el_coverage":           "Cond-EL Coverage (×)",
            "within_irb_design_envelope": "Within IRB 99.9% Envelope",
        }),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL RISK
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "07 · Model Risk":
    st.header("Model Risk Commentary")
    st.markdown("Where the math diverges from reality. Critical reading for any practitioner.")

    risks = [
        ("🔴 HIGH", "Gaussian Copula Assumption",
         "The Vasicek model assumes jointly normal asset returns — implying symmetric tail dependence. "
         "In reality, defaults cluster far more severely in crises than Gaussian copulas predict. "
         "The 2008 crisis was a definitive empirical refutation. "
         "Student-t and Clayton copulas capture left-tail dependence more accurately."),

        ("🔴 HIGH", "Correlation Underestimation",
         "Basel prescribed correlations (12–24% for corporate) are calibrated on historical data. "
         "Realized correlations in crises spike to 50–80%. "
         "This creates fundamental procyclicality: the model looks safe precisely when systemic risk is building. "
         "Credit portfolio managers should stress-test with 2–3× Basel correlations."),

        ("🟡 MEDIUM", "PD Estimation: TTC vs PIT",
         "Through-the-cycle (TTC) PDs smooth economic cycles and produce stable capital, "
         "but underestimate current default risk in recessions. "
         "Point-in-time (PIT) PDs are more accurate but amplify procyclicality — "
         "capital rises when banks can least afford it. "
         "Most regulatory IRB models use TTC PDs with countercyclical buffer overlays."),

        ("🟡 MEDIUM", "Model Procyclicality",
         "Even TTC PD models embed procyclicality via the correlation structure. "
         "RWA density compresses during booms (low PDs), encouraging expansion, "
         "then surges in recessions, forcing deleveraging — the opposite of what Basel's "
         "countercyclical capital buffer (CCyB) tries to correct."),

        ("🟡 MEDIUM", "Single-Factor Model Limitations",
         "Vasicek's single systematic factor cannot capture sector contagion, "
         "geographic clustering, or second-order effects (e.g., tech collapse dragging VC-backed firms). "
         "Multi-factor models (sector factor, geographic factor) better capture portfolio-specific concentration risk "
         "but require more complex calibration and are rarely used in regulatory frameworks."),

        ("🟢 LOW", "LGD Estimation Uncertainty",
         "LGD estimates carry substantial uncertainty — especially for workout LGDs in novel stress scenarios. "
         "A 45% LGD assumption in a severely distressed illiquid market may realise at 70%. "
         "Capital is roughly proportional to LGD, so a 10pp LGD error translates ~linearly to a capital error. "
         "Basel AIRB requires separate downturn LGD estimates (CRE36.83)."),
    ]

    for severity, title, text in risks:
        with st.expander(f"{severity} — {title}", expanded=True):
            st.markdown(text)

    st.divider()
    st.subheader("The Core Tension: Simplicity vs Reality")
    st.markdown("""
    Basel IRB is not a true model of credit reality — it is a **regulatory framework designed for
    comparability and minimum standards**.

    Internal models are expected to go further:
    - Sector and geographic factor models
    - Correlated LGD/PD dynamics (LGD worsens when PD rises)
    - Stressed correlation matrices
    - Liquidity-adjusted and workout LGD
    - Granularity adjustments for concentration risk

    The delta between **regulatory capital** and **economic capital under stress** is where
    quantitative risk management earns its value — and where frameworks either hold or fail.
    """)
