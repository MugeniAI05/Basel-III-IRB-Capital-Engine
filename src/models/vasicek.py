"""
Vasicek Single-Factor Credit Model.

Models the asset value of borrower i as:

    A_i = √R · Y + √(1−R) · ε_i

where:
    Y   ~ N(0,1)  is the common systematic factor
    ε_i ~ N(0,1)  is the idiosyncratic factor
    R             is the asset correlation

Default occurs when A_i < Φ⁻¹(PD).

The conditional probability of default given Y is:

    PD(Y) = Φ[(Φ⁻¹(PD) − √R · Y) / √(1−R)]

References:
    Vasicek, O. (2002). Loan portfolio value. Risk Magazine.
    Basel II: International Convergence of Capital Measurement (2006).
"""

import numpy as np
from dataclasses import dataclass
from src.utils.math_utils import normal_cdf, normal_inv_cdf


# ── Conditional Default Probability ─────────────────────────────────────────

def conditional_pd(
    pd: float | np.ndarray,
    correlation: float | np.ndarray,
    systemic_factor: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute conditional PD under a given realisation of the systemic factor.

    Args:
        pd: Unconditional probability of default.
        correlation: Asset correlation R ∈ (0, 1).
        systemic_factor: Realised systemic factor Y. Negative = recession.
            Y = 0  → average conditions (conditional PD ≈ unconditional PD)
            Y = -1 → mild stress
            Y = -2 → severe recession
            Y = -3 → GFC-level shock

    Returns:
        Conditional PD Φ[(Φ⁻¹(PD) − √R·Y) / √(1−R)].
    """
    pd = np.clip(np.asarray(pd, dtype=float), 1e-10, 1 - 1e-10)
    R = np.clip(np.asarray(correlation, dtype=float), 1e-6, 1 - 1e-6)
    Y = np.asarray(systemic_factor, dtype=float)

    numerator = normal_inv_cdf(pd) - np.sqrt(R) * Y
    return normal_cdf(numerator / np.sqrt(1.0 - R))


def portfolio_loss_rate(
    pds: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    correlations: np.ndarray,
    systemic_factor: float,
) -> float:
    """
    Compute portfolio loss rate under a given systemic factor realisation.

    In the large homogeneous portfolio (LHP) limit, idiosyncratic risk
    diversifies away and the loss rate is deterministic conditional on Y.

    Args:
        pds: Array of unconditional PDs.
        lgds: Array of LGDs.
        eads: Array of EADs.
        correlations: Array of asset correlations.
        systemic_factor: Scalar Y.

    Returns:
        Portfolio loss rate (loss / total EAD).
    """
    cond_pds = conditional_pd(pds, correlations, systemic_factor)
    losses = cond_pds * lgds * eads
    return losses.sum() / eads.sum()


@dataclass
class VasicekScenario:
    """Container for a named stress scenario."""
    name: str
    systemic_factor: float
    description: str

    @property
    def percentile(self) -> float:
        """Approximate percentile corresponding to Y."""
        return normal_cdf(self.systemic_factor)


# ── Standard Scenarios ────────────────────────────────────────────────────────

SCENARIOS = {
    "baseline":        VasicekScenario("Baseline",          0.0,  "Average conditions, Y = 0"),
    "mild_stress":     VasicekScenario("Mild Stress",       -1.0, "1-in-6 year shock, Y = −1"),
    "severe_recession":VasicekScenario("Severe Recession",  -2.0, "1-in-44 year shock, Y = −2"),
    "gfc":             VasicekScenario("GFC-like",          -2.5, "1-in-161 year shock, Y = −2.5"),
    "tail":            VasicekScenario("Tail (99.9%)",      -3.09,"1-in-1000 year shock, Y ≈ −3.09"),
}


def simulate_loss_distribution(
    pds: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    correlations: np.ndarray,
    n_simulations: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate portfolio loss distribution via Monte Carlo Vasicek.

    Each simulation draws Y ~ N(0,1) and computes the deterministic
    conditional loss (LHP approximation — idiosyncratic risk cancels).

    Args:
        pds, lgds, eads, correlations: Loan-level arrays.
        n_simulations: Number of systemic factor draws.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_simulations,) with portfolio loss rates.
    """
    rng = np.random.default_rng(seed)
    Y_draws = rng.standard_normal(n_simulations)

    loss_rates = np.array([
        portfolio_loss_rate(pds, lgds, eads, correlations, y)
        for y in Y_draws
    ])
    return loss_rates
