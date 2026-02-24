"""
Tests for the Basel III IRB Capital Formula (Basel-complete with maturity adjustment).
"""

import numpy as np
import pytest
from src.models.irb_capital import (
    irb_capital_ratio,
    irb_capital_requirement,
    basel_correlation,
    maturity_b,
    maturity_adjustment,
    downturn_lgd,
)


class TestBaselCorrelation:
    def test_range(self):
        pds = np.linspace(0.0003, 0.10, 100)
        R = basel_correlation(pds)
        assert np.all(R >= 0.119)
        assert np.all(R <= 0.241)

    def test_decreasing_in_pd(self):
        pds = np.array([0.001, 0.01, 0.05, 0.10])
        R = basel_correlation(pds)
        assert np.all(np.diff(R) < 0)

    def test_low_pd_approaches_0_24(self):
        assert abs(basel_correlation(0.0001) - 0.24) < 0.01

    def test_high_pd_approaches_0_12(self):
        assert abs(basel_correlation(0.99) - 0.12) < 0.01


class TestMaturityAdjustment:
    def test_b_non_negative(self):
        """b(PD) must be non-negative for all valid PD."""
        pds = np.linspace(0.0003, 0.99, 100)
        assert np.all(maturity_b(pds) >= 0)

    def test_b_decreasing_in_pd(self):
        """b(PD) decreases as PD increases — IG firms have higher b."""
        pds = np.array([0.001, 0.01, 0.05, 0.10])
        b = maturity_b(pds)
        assert np.all(np.diff(b) < 0), "b should decrease with PD"

    def test_ma_at_one_year_equals_one(self):
        """At M=1, MA should equal 1 (no maturity premium)."""
        pds = np.array([0.01, 0.03, 0.07])
        MA = maturity_adjustment(pds, maturity=1.0)
        np.testing.assert_allclose(MA, 1.0, atol=1e-10)

    def test_ma_at_default_maturity_gt_one(self):
        """At M=2.5 (Basel default), MA > 1 for all valid PD."""
        pds = np.linspace(0.0003, 0.30, 50)
        MA = maturity_adjustment(pds, maturity=2.5)
        assert np.all(MA > 1.0), "MA at 2.5yr should exceed 1.0"

    def test_ma_increasing_in_maturity(self):
        """Longer maturities require more capital (MA increases with M)."""
        pd = 0.02
        M_vals = np.array([1.0, 2.0, 2.5, 3.0, 5.0])
        MA = maturity_adjustment(pd, M_vals)
        assert np.all(np.diff(MA) > 0), "MA should increase with maturity"

    def test_ma_clamps_maturity(self):
        """Maturity outside [1, 5] should be clamped."""
        ma_low  = maturity_adjustment(0.02, 0.5)   # below 1 → clamped to 1
        ma_high = maturity_adjustment(0.02, 10.0)  # above 5 → clamped to 5
        ma_one  = maturity_adjustment(0.02, 1.0)
        ma_five = maturity_adjustment(0.02, 5.0)
        assert abs(ma_low - ma_one) < 1e-10
        assert abs(ma_high - ma_five) < 1e-10

    def test_known_b_value(self):
        """Spot-check b(PD=1%) against the Basel formula."""
        pd = 0.01
        expected = (0.11852 - 0.05478 * np.log(pd)) ** 2
        assert abs(maturity_b(pd) - expected) < 1e-12


class TestIRBCapitalRatio:
    def test_non_negative(self):
        pds = np.linspace(0.001, 0.10, 50)
        K = irb_capital_ratio(pds, 0.45)
        assert np.all(K >= 0)

    def test_less_than_lgd(self):
        K = irb_capital_ratio(0.50, 0.80, confidence_level=0.9999)
        assert K < 0.80

    def test_monotone_in_lgd(self):
        lgds = np.array([0.20, 0.30, 0.45, 0.60])
        K = irb_capital_ratio(0.02, lgds, 0.15)
        assert np.all(np.diff(K) > 0)

    def test_monotone_in_confidence(self):
        confs = np.array([0.990, 0.995, 0.999, 0.9999])
        K = irb_capital_ratio(0.02, 0.45, 0.15, confs)
        assert np.all(np.diff(K) > 0)

    def test_monotone_in_maturity(self):
        """K should increase with maturity (MA effect)."""
        maturities = np.array([1.0, 2.0, 2.5, 3.0, 5.0])
        K = irb_capital_ratio(0.02, 0.45, 0.15, maturity=maturities)
        assert np.all(np.diff(K) > 0), "K should increase with maturity"

    def test_maturity_adjustment_raises_k(self):
        """K with MA must exceed K without MA (for M > 1)."""
        K_with = irb_capital_ratio(0.02, 0.45, 0.15, maturity=2.5, apply_maturity_adjustment=True)
        K_no   = irb_capital_ratio(0.02, 0.45, 0.15, maturity=2.5, apply_maturity_adjustment=False)
        assert K_with > K_no, "Maturity adjustment must increase K above bare kernel"

    def test_no_ma_at_one_year(self):
        """At M=1, K with MA == K without MA."""
        K_with = irb_capital_ratio(0.02, 0.45, 0.15, maturity=1.0, apply_maturity_adjustment=True)
        K_no   = irb_capital_ratio(0.02, 0.45, 0.15, maturity=1.0, apply_maturity_adjustment=False)
        assert abs(K_with - K_no) < 1e-10, "At M=1, MA=1 so K should be identical"

    def test_known_approximation(self):
        """
        K(PD=1%, LGD=45%, M=2.5) should be ~8-15% after maturity adjustment.
        Pre-MA it was ~6-9%; MA at M=2.5 lifts it materially.
        """
        K = irb_capital_ratio(0.01, 0.45, maturity=2.5)
        assert 0.07 < K < 0.18, f"Expected 7–18% after MA, got {K:.4f}"


class TestIRBCapitalRequirement:
    def test_full_stack_keys(self):
        result = irb_capital_requirement(0.02, 0.45, 10_000_000, correlation=0.15)
        required = {"K", "K_no_ma", "MA", "capital", "rwa", "req_capital", "el"}
        assert required.issubset(set(result.keys()))

    def test_k_ge_k_no_ma(self):
        """K (with MA) must be >= K_no_ma for M=2.5."""
        result = irb_capital_requirement(0.02, 0.45, 10_000_000, correlation=0.15, maturity=2.5)
        assert result["K"] >= result["K_no_ma"]

    def test_ma_gt_one_at_default_maturity(self):
        result = irb_capital_requirement(0.02, 0.45, 10_000_000, correlation=0.15, maturity=2.5)
        assert np.all(result["MA"] > 1.0)

    def test_rwa_relationship(self):
        result = irb_capital_requirement(0.02, 0.45, 10_000_000, 0.15)
        assert abs(result["rwa"] - 12.5 * result["capital"]) < 1.0

    def test_el_formula(self):
        pd, lgd, ead = 0.02, 0.45, 10_000_000
        result = irb_capital_requirement(pd, lgd, ead, 0.15)
        assert abs(result["el"] - pd * lgd * ead) < 1.0


class TestDownturnLGD:
    def test_default_shock(self):
        lgd = np.array([0.40, 0.45, 0.50])
        np.testing.assert_allclose(downturn_lgd(lgd), np.array([0.60, 0.65, 0.70]))

    def test_cap_at_one(self):
        assert downturn_lgd(0.90, shock=0.20) == pytest.approx(1.0)

    def test_custom_shock(self):
        assert downturn_lgd(0.40, shock=0.10) == pytest.approx(0.50)

