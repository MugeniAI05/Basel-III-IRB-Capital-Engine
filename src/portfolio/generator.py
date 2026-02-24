"""
Synthetic Corporate Loan Portfolio Generator.

Generates a realistic 10,000-loan portfolio with:
  - 8 industry sectors with distinct PD profiles
  - Log-normal EAD distribution ($1M–$50M range)
  - LGD distributed around industry norms
  - Heterogeneous PD via log-normal dispersion around sector means

This is designed to demonstrate portfolio heterogeneity required
for a meaningful IRB capital calculation.
"""

import numpy as np
import pandas as _pd
from dataclasses import dataclass
from src.models.irb_capital import irb_capital_requirement, basel_correlation


@dataclass
class IndustryConfig:
    """Configuration for a loan industry segment."""
    name: str
    pd_mean: float       # Sector mean PD (through-the-cycle)
    pd_vol: float        # Log-normal volatility around mean PD
    lgd_mean: float      # Sector mean LGD
    lgd_std: float       # LGD standard deviation
    ead_mean_mn: float   # Mean EAD in $M
    weight: float        # Portfolio weight (approximate)


INDUSTRY_CONFIGS = [
    IndustryConfig("Technology",    pd_mean=0.008, pd_vol=0.7, lgd_mean=0.40, lgd_std=0.08, ead_mean_mn=6.0,  weight=0.15),
    IndustryConfig("Energy",        pd_mean=0.025, pd_vol=0.8, lgd_mean=0.45, lgd_std=0.10, ead_mean_mn=8.0,  weight=0.12),
    IndustryConfig("Healthcare",    pd_mean=0.012, pd_vol=0.6, lgd_mean=0.38, lgd_std=0.08, ead_mean_mn=5.0,  weight=0.12),
    IndustryConfig("Real Estate",   pd_mean=0.035, pd_vol=0.9, lgd_mean=0.30, lgd_std=0.08, ead_mean_mn=10.0, weight=0.15),
    IndustryConfig("Manufacturing", pd_mean=0.018, pd_vol=0.7, lgd_mean=0.42, lgd_std=0.09, ead_mean_mn=5.0,  weight=0.12),
    IndustryConfig("Retail",        pd_mean=0.042, pd_vol=0.9, lgd_mean=0.50, lgd_std=0.10, ead_mean_mn=3.0,  weight=0.13),
    IndustryConfig("Finance",       pd_mean=0.015, pd_vol=0.7, lgd_mean=0.35, lgd_std=0.08, ead_mean_mn=7.0,  weight=0.11),
    IndustryConfig("Utilities",     pd_mean=0.010, pd_vol=0.6, lgd_mean=0.38, lgd_std=0.07, ead_mean_mn=9.0,  weight=0.10),
]


def generate_portfolio(
    n_loans: int = 10_000,
    seed: int = 42,
    pd_floor: float = 0.001,
    pd_cap: float = 0.10,
    lgd_floor: float = 0.20,
    lgd_cap: float = 0.60,
) -> _pd.DataFrame:
    """
    Generate a synthetic corporate loan portfolio.

    Args:
        n_loans: Number of loans to generate.
        seed: Random seed for reproducibility.
        pd_floor: Minimum PD (Basel III floor).
        pd_cap: Maximum PD.
        lgd_floor: Minimum LGD.
        lgd_cap: Maximum LGD.

    Returns:
        DataFrame with columns:
            loan_id, industry, pd, lgd, ead, correlation,
            el, K, capital, rwa, req_capital
    """
    rng = np.random.default_rng(seed)

    # Assign loans to industries by weight
    weights = np.array([c.weight for c in INDUSTRY_CONFIGS])
    weights /= weights.sum()
    industry_indices = rng.choice(len(INDUSTRY_CONFIGS), size=n_loans, p=weights)

    records = []
    for idx in industry_indices:
        cfg = INDUSTRY_CONFIGS[idx]

        # PD: log-normal dispersion around sector mean
        log_pd = np.log(cfg.pd_mean) + rng.normal(0, cfg.pd_vol)
        pd = np.clip(np.exp(log_pd), pd_floor, pd_cap)

        # LGD: normal dispersion around sector mean
        lgd = np.clip(rng.normal(cfg.lgd_mean, cfg.lgd_std), lgd_floor, lgd_cap)

        # EAD: log-normal, mean ≈ cfg.ead_mean_mn $M
        log_ead = np.log(cfg.ead_mean_mn * 1e6) + rng.normal(0, 0.8)
        ead = np.exp(log_ead)

        # Maturity: log-normal around 3 years, capped to [1, 5] per Basel
        log_mat = np.log(3.0) + rng.normal(0, 0.4)
        maturity = np.clip(np.exp(log_mat), 1.0, 5.0)

        records.append({"industry": cfg.name, "pd": pd, "lgd": lgd, "ead": ead, "maturity": maturity})

    df = _pd.DataFrame(records)
    df.index.name = "loan_id"
    df = df.reset_index()

    # Compute Basel correlation
    df["correlation"] = basel_correlation(df["pd"].values)

    # Compute IRB capital stack (Basel-complete with maturity adjustment)
    capital_results = irb_capital_requirement(
        pd=df["pd"].values,
        lgd=df["lgd"].values,
        ead=df["ead"].values,
        correlation=df["correlation"].values,
        maturity=df["maturity"].values,
    )
    df["el"]          = capital_results["el"]
    df["K"]           = capital_results["K"]
    df["K_no_ma"]     = capital_results["K_no_ma"]
    df["MA"]          = capital_results["MA"]
    df["capital"]     = capital_results["capital"]
    df["rwa"]         = capital_results["rwa"]
    df["req_capital"] = capital_results["req_capital"]

    return df
