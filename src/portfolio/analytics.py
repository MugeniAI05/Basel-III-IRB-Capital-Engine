"""
Portfolio-level analytics: EL, capital, RWA, and industry breakdowns.
"""

import pandas as pd
import numpy as np


def portfolio_summary(df: pd.DataFrame) -> dict:
    """
    Compute top-level portfolio metrics.

    Args:
        df: Loan DataFrame from generator.generate_portfolio().

    Returns:
        dict of portfolio-level statistics.
    """
    total_ead = df["ead"].sum()
    total_el = df["el"].sum()
    total_capital = df["capital"].sum()
    total_rwa = df["rwa"].sum()
    total_req_capital = df["req_capital"].sum()

    return {
        "n_loans":            len(df),
        "total_ead":          total_ead,
        "total_el":           total_el,
        "total_capital":      total_capital,
        "total_rwa":          total_rwa,
        "total_req_capital":  total_req_capital,
        # Rates
        "el_rate":            total_el / total_ead,
        "capital_rate":       total_capital / total_ead,
        "rwa_density":        total_rwa / total_ead,
        # Weighted averages (EAD-weighted)
        "wtd_avg_pd":         (df["pd"] * df["ead"]).sum() / total_ead,
        "wtd_avg_lgd":        (df["lgd"] * df["ead"]).sum() / total_ead,
        "wtd_avg_K":          (df["K"] * df["ead"]).sum() / total_ead,
        "wtd_avg_correlation":(df["correlation"] * df["ead"]).sum() / total_ead,
        # Capital efficiency
        "capital_to_el_ratio": total_capital / total_el,
    }


def industry_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate portfolio metrics by industry.

    Returns:
        DataFrame indexed by industry with EAD, EL, capital, and rates.
    """
    grp = df.groupby("industry")

    agg = grp.agg(
        n_loans=("loan_id", "count"),
        total_ead=("ead", "sum"),
        total_el=("el", "sum"),
        total_capital=("capital", "sum"),
        total_rwa=("rwa", "sum"),
    )

    # EAD-weighted averages — use explicit formula to avoid pandas groupby.apply deprecation
    for col in ["pd", "lgd", "K", "correlation"]:
        weighted = df[col] * df["ead"]
        agg[f"avg_{col}"] = (
            df.groupby("industry")[col]
            .transform(lambda x: x)  # identity — just to get index alignment
            .groupby(df["industry"])
            .apply(lambda _: None)   # placeholder, replaced below
        )
    # Recompute cleanly using vectorised group sums
    for col in ["pd", "lgd", "K", "correlation"]:
        num = df.assign(_w=df[col] * df["ead"]).groupby("industry")["_w"].sum()
        den = agg["total_ead"]
        agg[f"avg_{col}"] = num / den

    agg["el_rate"]      = agg["total_el"] / agg["total_ead"]
    agg["capital_rate"] = agg["total_capital"] / agg["total_ead"]
    agg["ead_share"]    = agg["total_ead"] / agg["total_ead"].sum()

    return agg.sort_values("total_ead", ascending=False)


def pd_distribution_bins(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """
    Build a PD histogram for charting.

    Returns:
        DataFrame with columns: bin_label, n_loans, total_ead.
    """
    bins = np.linspace(0, 0.10, n_bins + 1)
    labels = [f"{bins[i]*100:.1f}–{bins[i+1]*100:.1f}%" for i in range(n_bins)]

    df = df.copy()
    df["pd_bin"] = pd.cut(df["pd"], bins=bins, labels=labels, include_lowest=True)

    hist = (
        df.groupby("pd_bin", observed=True)
        .agg(n_loans=("loan_id", "count"), total_ead=("ead", "sum"))
        .reset_index()
        .rename(columns={"pd_bin": "pd_range"})
    )
    return hist
