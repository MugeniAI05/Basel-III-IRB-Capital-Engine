"""
Basel III AIRB Capital Formula — Corporate Asset Class (Basel-Complete).

Implements the full Advanced IRB capital requirement for corporate exposures
as specified in Basel III (CRE31–CRE36), including the maturity adjustment
which is required for a spec-compliant corporate AIRB calculator.

─────────────────────────────────────────────────────────────────────────────
FORMULA COMPONENTS (CRE31.8 / CRE32.4):

  Step 1 — Asset Correlation (CRE31.44):
    R = 0.12 × (1 − e^(−50·PD)) / (1 − e^(−50))
      + 0.24 × [1 − (1 − e^(−50·PD)) / (1 − e^(−50))]

  Step 2 — Maturity Adjustment (CRE32.4):
    b(PD) = [0.11852 − 0.05478 × ln(PD)]²
    MA(PD, M) = (1 + (M − 2.5) × b(PD)) / (1 − 1.5 × b(PD))

    where M is effective maturity in years, capped at [1, 5].
    Basel default M = 2.5 years for AIRB.

  Step 3 — Conditional loss quantile (Vasicek kernel):
    Q(PD, R) = Φ[(Φ⁻¹(PD) + √R × Φ⁻¹(0.999)) / √(1−R)]

  Step 4 — Capital ratio K:
    K = (LGD × Q(PD, R) − PD × LGD) × MA(PD, M)

  Step 5 — RWA and required capital:
    RWA     = 12.5 × EAD × K
    Req Cap = 8% × RWA

─────────────────────────────────────────────────────────────────────────────
DESIGN NOTE — why the maturity adjustment matters:
  The Vasicek kernel alone (K without MA) captures cross-sectional risk at a
  single horizon. The maturity adjustment scales capital upward for longer-
  dated exposures because a 5-year loan has more opportunity to migrate to
  worse rating grades before maturity. At M=2.5 (Basel default), MA > 1 for
  virtually all PD levels, lifting K by 20–60% relative to the bare kernel.
  Omitting MA produces capital ratios that are systematically too low and do
  not match published Basel QIS or bank Pillar 3 disclosures.

References:
    BIS: Basel III — CRE31 (correlations), CRE32 (maturity), CRE36 (LGD)
    BCBS WP #14: "An Explanatory Note on the Basel II IRB Risk Weight Functions"
    Fed SR 07-5: "Guidance on Implementing the Advanced IRB Approach"
"""

import numpy as np
from src.utils.math_utils import normal_cdf, normal_inv_cdf, validate_pd, validate_lgd


# ── Basel Asset Correlation ────────────────────────────────────────────────────

def basel_correlation(pd: float | np.ndarray) -> float | np.ndarray:
    """
    Basel III prescribed asset correlation for corporate exposures (CRE31.44).

    R is a decreasing function of PD:
      - High-quality (low-PD) obligors: R → 0.24 (more systematic risk)
      - Distressed (high-PD) obligors:  R → 0.12 (more idiosyncratic risk)

    This reflects the empirical observation that investment-grade firms'
    asset values move more in lockstep with the economic cycle than
    speculative-grade firms, which are more driven by firm-specific factors.

    Args:
        pd: Probability of default (scalar or array).

    Returns:
        Asset correlation R ∈ [0.12, 0.24].
    """
    pd = np.asarray(pd, dtype=float)
    k = 50.0
    weight = (1.0 - np.exp(-k * pd)) / (1.0 - np.exp(-k))
    return 0.12 * weight + 0.24 * (1.0 - weight)


# ── Maturity Adjustment ────────────────────────────────────────────────────────

def maturity_b(pd: float | np.ndarray) -> float | np.ndarray:
    """
    Basel III maturity adjustment slope b(PD) (CRE32.4).

    Captures how much capital needs to increase per additional year of
    maturity, as a function of credit quality (PD).

    b(PD) = [0.11852 − 0.05478 × ln(PD)]²

    Higher-quality (lower-PD) obligors have larger b, meaning their capital
    is more sensitive to maturity. Intuitively, an IG borrower has more
    rating migration risk over a 5-year horizon than a CCC borrower
    (which is already near default).

    Args:
        pd: Probability of default (floored at Basel minimum before log).

    Returns:
        b: Maturity slope coefficient (non-negative scalar or array).
    """
    pd = validate_pd(np.asarray(pd, dtype=float))
    return (0.11852 - 0.05478 * np.log(pd)) ** 2


def maturity_adjustment(
    pd: float | np.ndarray,
    maturity: float | np.ndarray = 2.5,
) -> float | np.ndarray:
    """
    Basel III maturity adjustment factor MA(PD, M) (CRE32.4).

    MA scales the Vasicek capital ratio upward for maturities above 1 year,
    reflecting the increased risk of rating migration over longer horizons.

    MA(PD, M) = (1 + (M − 2.5) × b(PD)) / (1 − 1.5 × b(PD))

    At M = 2.5 (Basel AIRB default), the numerator simplifies to 1, so:
        MA(PD, 2.5) = 1 / (1 − 1.5 × b(PD))  > 1 always.

    At M = 1 year (minimum):
        MA(PD, 1) = (1 − 1.5 × b(PD)) / (1 − 1.5 × b(PD)) = 1.0
        — No maturity premium for 1-year exposures.

    Args:
        pd: Probability of default.
        maturity: Effective maturity M in years. Basel cap: [1.0, 5.0].
                  Default 2.5 years (Basel AIRB standard assumption).

    Returns:
        MA: Maturity adjustment multiplier ≥ 1 for M ≥ 1.
    """
    pd = validate_pd(np.asarray(pd, dtype=float))
    M = np.clip(np.asarray(maturity, dtype=float), 1.0, 5.0)
    b = maturity_b(pd)
    return (1.0 + (M - 2.5) * b) / (1.0 - 1.5 * b)


# ── IRB Capital Ratio ─────────────────────────────────────────────────────────

def irb_capital_ratio(
    pd: float | np.ndarray,
    lgd: float | np.ndarray,
    correlation: float | np.ndarray | None = None,
    confidence_level: float = 0.999,
    maturity: float | np.ndarray = 2.5,
    apply_maturity_adjustment: bool = True,
) -> float | np.ndarray:
    """
    Basel III AIRB capital ratio K for corporate exposures (Basel-complete).

    Implements the full CRE31/CRE32 formula including maturity adjustment.
    This matches the spec used in bank Pillar 3 disclosures and Fed QIS.

    Full formula:
        Q    = Φ[(Φ⁻¹(PD) + √R × Φ⁻¹(conf)) / √(1−R)]   (Vasicek kernel)
        K_raw = LGD × Q − PD × LGD                         (unexpected loss)
        K    = K_raw × MA(PD, M)                            (maturity-adjusted)

    Args:
        pd: Probability of default (annual, unconditional TTC or PIT).
        lgd: Loss given default as fraction [0, 1]. Should be downturn LGD
             for AIRB (CRE36.83).
        correlation: Asset correlation R. If None, uses Basel prescribed
                     corporate correlation function (CRE31.44).
        confidence_level: VaR confidence level. Basel standard = 0.999.
        maturity: Effective maturity M in years [1, 5].
                  Basel AIRB default = 2.5 years (CRE32.5).
                  FIRB uses a fixed M = 2.5 for all exposures.
        apply_maturity_adjustment: If True (default), applies MA per Basel
                  AIRB spec. Set False to recover the bare Vasicek kernel
                  (useful for educational comparison).

    Returns:
        K: Capital ratio as fraction of EAD. Non-negative.

    Note on maturity adjustment:
        Omitting MA (apply_maturity_adjustment=False) underestimates K by
        roughly 20–50% for typical corporate maturities (2–5 years).
        The MA is not optional in a spec-compliant corporate AIRB model.
    """
    pd = validate_pd(np.asarray(pd, dtype=float))
    lgd = validate_lgd(np.asarray(lgd, dtype=float))

    if correlation is None:
        R = basel_correlation(pd)
    else:
        R = np.clip(np.asarray(correlation, dtype=float), 1e-6, 0.9999)

    # Vasicek conditional loss quantile (the kernel)
    numerator = normal_inv_cdf(pd) + np.sqrt(R) * normal_inv_cdf(confidence_level)
    K_raw = lgd * normal_cdf(numerator / np.sqrt(1.0 - R)) - pd * lgd
    K_raw = np.maximum(K_raw, 0.0)

    if not apply_maturity_adjustment:
        return K_raw

    # Apply Basel maturity adjustment (CRE32.4)
    MA = maturity_adjustment(pd, maturity)
    return K_raw * MA


def irb_capital_requirement(
    pd: float | np.ndarray,
    lgd: float | np.ndarray,
    ead: float | np.ndarray,
    correlation: float | np.ndarray | None = None,
    confidence_level: float = 0.999,
    maturity: float | np.ndarray = 2.5,
    tier1_ratio: float = 0.08,
    apply_maturity_adjustment: bool = True,
) -> dict:
    """
    Compute full Basel III AIRB capital stack for a loan or portfolio.

    Args:
        pd: Probability of default.
        lgd: Loss given default (downturn LGD for AIRB).
        ead: Exposure at default ($).
        correlation: Asset correlation R. If None, uses Basel function.
        confidence_level: VaR confidence level (default 99.9%).
        maturity: Effective maturity in years (default 2.5, Basel AIRB).
        tier1_ratio: Minimum capital ratio applied to RWA (default 8%).
        apply_maturity_adjustment: Apply CRE32.4 maturity adjustment.

    Returns:
        dict with keys:
            K               — Capital ratio (EAD-fraction, maturity-adjusted)
            K_no_ma         — Capital ratio without maturity adjustment (kernel only)
            MA              — Maturity adjustment factor applied
            capital         — Dollar capital = EAD × K
            rwa             — Risk-weighted assets = 12.5 × capital
            req_capital     — Required capital = tier1_ratio × RWA
            el              — Expected loss = PD × LGD × EAD
    """
    pd_arr = np.asarray(pd, dtype=float)
    lgd_arr = np.asarray(lgd, dtype=float)
    ead_arr = np.asarray(ead, dtype=float)

    K = irb_capital_ratio(
        pd_arr, lgd_arr, correlation, confidence_level, maturity,
        apply_maturity_adjustment=apply_maturity_adjustment,
    )
    K_no_ma = irb_capital_ratio(
        pd_arr, lgd_arr, correlation, confidence_level, maturity,
        apply_maturity_adjustment=False,
    )
    MA = maturity_adjustment(validate_pd(pd_arr), maturity)

    capital = ead_arr * K
    rwa = 12.5 * capital
    req_capital = tier1_ratio * rwa
    el = validate_pd(pd_arr) * validate_lgd(lgd_arr) * ead_arr

    return {
        "K":            K,
        "K_no_ma":      K_no_ma,
        "MA":           MA,
        "capital":      capital,
        "rwa":          rwa,
        "req_capital":  req_capital,
        "el":           el,
    }


def downturn_lgd(lgd: float | np.ndarray, shock: float = 0.20) -> float | np.ndarray:
    """
    Apply a downturn LGD stress (additive percentage shock).

    Basel AIRB requires downturn LGD estimates that reflect conditions
    in periods of high credit losses (CRE36.83). Downturn LGD is typically
    estimated as TTC LGD plus a cyclical add-on derived from stressed
    recovery rate studies.

    Args:
        lgd: Through-the-cycle LGD.
        shock: Additive shock in percentage points (default +20pp).

    Returns:
        Downturn LGD, capped at 100%.
    """
    return validate_lgd(np.asarray(lgd, dtype=float) + shock)
