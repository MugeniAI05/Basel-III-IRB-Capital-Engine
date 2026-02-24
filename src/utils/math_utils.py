"""
Statistical utility functions for Basel IRB calculations.

Uses scipy.stats for numerical accuracy. All functions are vectorized
to accept both scalar and numpy array inputs.
"""

import numpy as np
from scipy.stats import norm


def normal_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal CDF: Φ(x)."""
    return norm.cdf(x)


def normal_inv_cdf(p: float | np.ndarray) -> float | np.ndarray:
    """
    Inverse standard normal CDF: Φ⁻¹(p).

    Args:
        p: Probability in (0, 1).

    Returns:
        Quantile z such that Φ(z) = p.
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return norm.ppf(p)


def validate_pd(pd: float | np.ndarray, floor: float = 0.0003) -> float | np.ndarray:
    """
    Apply Basel III PD floor and ceiling.

    Basel III imposes a minimum PD of 0.03% for non-defaulted exposures.

    Args:
        pd: Raw probability of default.
        floor: Minimum PD (Basel floor = 0.0003).

    Returns:
        Floored PD.
    """
    return np.clip(pd, floor, 0.9999)


def validate_lgd(lgd: float | np.ndarray) -> float | np.ndarray:
    """Clip LGD to [0, 1]."""
    return np.clip(lgd, 0.0, 1.0)
