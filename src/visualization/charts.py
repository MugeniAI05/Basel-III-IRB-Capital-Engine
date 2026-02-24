"""
Reusable Plotly chart builders for the IRB Capital Engine dashboard.

All functions return plotly.graph_objects.Figure objects,
ready for st.plotly_chart() in Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Design Constants ──────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#0a0c10",
    "surface":  "#111318",
    "border":   "#1e2433",
    "accent":   "#00c8ff",
    "accent2":  "#ff6b35",
    "accent3":  "#7fff6b",
    "accent4":  "#ffcc00",
    "muted":    "#4a5568",
    "text":     "#e2e8f0",
    "textDim":  "#718096",
}

PALETTE = [COLORS["accent"], COLORS["accent4"], COLORS["accent2"], "#ff3366", COLORS["accent3"]]


def _base_layout(**kwargs) -> dict:
    """Base Plotly layout for dark theme."""
    return dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="IBM Plex Mono, monospace", size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"]),
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        **kwargs,
    )


# ── Portfolio Charts ──────────────────────────────────────────────────────────

def chart_pd_distribution(hist_df: pd.DataFrame) -> go.Figure:
    """Histogram of loan count by PD bucket."""
    fig = go.Figure(go.Bar(
        x=hist_df["pd_range"].astype(str),
        y=hist_df["n_loans"],
        marker_color=COLORS["accent"],
        marker_opacity=0.85,
        name="Loan Count",
    ))
    fig.update_layout(
        title="PD Distribution — Loan Count",
        xaxis_title="PD Range",
        yaxis_title="Number of Loans",
        **_base_layout(),
    )
    return fig


def chart_industry_capital(industry_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: EL rate and capital rate by industry."""
    fig = go.Figure()
    industries = industry_df.index.tolist()

    fig.add_trace(go.Bar(
        name="EL Rate", x=industries,
        y=(industry_df["el_rate"] * 100).round(3),
        marker_color=COLORS["accent4"], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Capital Rate", x=industries,
        y=(industry_df["capital_rate"] * 100).round(3),
        marker_color=COLORS["accent2"], opacity=0.85,
    ))
    fig.update_layout(
        title="EL Rate vs Capital Rate by Industry (%)",
        yaxis_title="Rate (%)", barmode="group",
        **_base_layout(),
    )
    return fig


# ── EL vs Capital ─────────────────────────────────────────────────────────────

def chart_el_vs_capital(industry_df: pd.DataFrame) -> go.Figure:
    """Bar chart: EL ($M) and Capital ($M) by industry."""
    fig = go.Figure()
    industries = industry_df.index.tolist()

    fig.add_trace(go.Bar(
        name="Expected Loss ($M)", x=industries,
        y=(industry_df["total_el"] / 1e6).round(1),
        marker_color=COLORS["accent4"], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="IRB Capital ($M)", x=industries,
        y=(industry_df["total_capital"] / 1e6).round(1),
        marker_color=COLORS["accent2"], opacity=0.85,
    ))
    fig.update_layout(
        title="Expected Loss vs IRB Capital by Industry",
        yaxis_title="$M", barmode="group",
        **_base_layout(),
    )
    return fig


# ── Vasicek ───────────────────────────────────────────────────────────────────

def chart_conditional_pd_vs_y(
    pd_val: float,
    correlation: float,
) -> go.Figure:
    """Conditional PD curve vs systemic factor Y."""
    from src.models.vasicek import conditional_pd

    Y = np.linspace(-3.5, 1.5, 200)
    cond = conditional_pd(pd_val, correlation, Y) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Y, y=cond, mode="lines",
        line=dict(color=COLORS["accent"], width=2),
        name="Conditional PD",
    ))
    fig.add_hline(
        y=pd_val * 100, line_dash="dash",
        line_color=COLORS["textDim"],
        annotation_text=f"Baseline PD = {pd_val*100:.2f}%",
    )
    fig.add_vline(x=-2, line_dash="dash", line_color=COLORS["accent2"],
                  annotation_text="Severe Recession")
    fig.update_layout(
        title="Conditional PD vs Systemic Factor Y",
        xaxis_title="Systemic Factor Y",
        yaxis_title="Conditional PD (%)",
        **_base_layout(),
    )
    return fig


# ── IRB Capital ───────────────────────────────────────────────────────────────

def chart_capital_vs_pd(sensitivity_df: pd.DataFrame) -> go.Figure:
    """K% and EL% vs PD — shows non-linearity of IRB formula."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sensitivity_df["pd"] * 100,
        y=sensitivity_df["K"] * 100,
        mode="lines", line=dict(color=COLORS["accent"], width=2),
        fill="tozeroy", fillcolor="rgba(0,200,255,0.08)",
        name="Capital K (%)",
    ))
    fig.add_trace(go.Scatter(
        x=sensitivity_df["pd"] * 100,
        y=sensitivity_df["el_rate"] * 100,
        mode="lines", line=dict(color=COLORS["accent4"], width=1.5, dash="dash"),
        name="EL Rate (%)",
    ))
    fig.update_layout(
        title="IRB Capital Ratio K vs PD",
        xaxis_title="PD (%)", yaxis_title="Rate (%)",
        **_base_layout(),
    )
    return fig


def chart_capital_vs_correlation(corr_df: pd.DataFrame) -> go.Figure:
    """Capital K% vs asset correlation for multiple PD tiers."""
    fig = go.Figure()
    k_cols = [c for c in corr_df.columns if c.startswith("K_")]
    labels = {"K_pd1pct": "PD = 1%", "K_pd3pct": "PD = 3%", "K_pd7pct": "PD = 7%"}

    for i, col in enumerate(k_cols):
        fig.add_trace(go.Scatter(
            x=corr_df["R"] * 100,
            y=corr_df[col] * 100,
            mode="lines", line=dict(color=PALETTE[i], width=2),
            name=labels.get(col, col),
        ))
    fig.update_layout(
        title="Capital K% vs Asset Correlation R",
        xaxis_title="Correlation R (%)", yaxis_title="Capital K (%)",
        **_base_layout(),
    )
    return fig


def chart_capital_vs_confidence(conf_df: pd.DataFrame) -> go.Figure:
    """Capital K% vs confidence level."""
    fig = go.Figure()
    k_cols = [c for c in conf_df.columns if c.startswith("K_")]
    labels = {"K_pd1pct": "PD = 1%", "K_pd3pct": "PD = 3%"}

    for i, col in enumerate(k_cols):
        fig.add_trace(go.Scatter(
            x=conf_df["confidence_level"] * 100,
            y=conf_df[col] * 100,
            mode="lines", line=dict(color=PALETTE[i], width=2),
            name=labels.get(col, col),
        ))
    fig.add_vline(x=99.9, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text="Basel 99.9%")
    fig.update_layout(
        title="Capital K% vs Confidence Level",
        xaxis_title="Confidence Level (%)", yaxis_title="Capital K (%)",
        **_base_layout(),
    )
    return fig


# ── Stress ────────────────────────────────────────────────────────────────────

def chart_stress_sweep(sweep_df: pd.DataFrame) -> go.Figure:
    """Loss rate vs systemic factor Y with capital ratio overlay."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_df["Y"],
        y=sweep_df["cond_el_rate"] * 100,
        mode="lines", fill="tozeroy",
        line=dict(color=COLORS["accent2"], width=2),
        fillcolor="rgba(255,107,53,0.12)",
        name="Stressed Loss Rate (%)",
    ))
    fig.add_trace(go.Scatter(
        x=sweep_df["Y"],
        y=sweep_df["capital_rate"] * 100,
        mode="lines", line=dict(color=COLORS["accent"], width=2, dash="dash"),
        name="IRB Capital Rate (%)",
    ))
    fig.add_vline(x=-2, line_dash="dot", line_color=COLORS["accent2"],
                  annotation_text="Severe Recession (Y=−2)")
    fig.update_layout(
        title="Portfolio Loss Rate vs Systemic Factor Y",
        xaxis_title="Systemic Factor Y",
        yaxis_title="Rate (%)",
        **_base_layout(),
    )
    return fig


def chart_scenario_comparison(scenario_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing stressed loss vs capital across scenarios."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Stressed Loss ($M)", x=scenario_df["scenario"],
        y=(scenario_df["cond_expected_loss"] / 1e6).round(1),
        marker_color=COLORS["accent2"], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="IRB Capital ($M)", x=scenario_df["scenario"],
        y=(scenario_df["irb_capital"] / 1e6).round(1),
        marker_color=COLORS["accent"], opacity=0.85,
    ))
    fig.update_layout(
        title="Stressed Loss vs IRB Capital by Scenario",
        yaxis_title="$M", barmode="group",
        **_base_layout(),
    )
    return fig
