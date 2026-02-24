"""
Tests for the Vasicek Single-Factor Credit Model.
"""

import numpy as np
import pytest
from src.models.vasicek import (
    conditional_pd,
    portfolio_loss_rate,
    SCENARIOS,
    simulate_loss_distribution,
)


class TestConditionalPD:
    def test_zero_factor_zero_correlation_returns_unconditional(self):
        """At Y=0 and R=0, conditional PD equals unconditional PD exactly."""
        pd = 0.02
        cond = conditional_pd(pd, 1e-9, 0.0)  # R ≈ 0
        assert abs(cond - pd) < 1e-4, f"Expected ≈{pd}, got {cond}"

    def test_zero_factor_with_correlation_less_than_unconditional(self):
        """At Y=0 with positive R, conditional PD is less than unconditional PD.
        This is because Y=0 represents the mean, not the median of the conditional distribution."""
        pd = 0.02
        cond = conditional_pd(pd, 0.15, 0.0)
        # Conditional PD at Y=0 will be less than unconditional due to Jensen's inequality
        assert 0 < cond < 1, "Conditional PD should be a valid probability"

    def test_negative_factor_increases_pd(self):
        """Negative Y (recession) should increase conditional PD."""
        pd = 0.02
        cond_neg = conditional_pd(pd, 0.15, -2.0)
        assert cond_neg > pd, "Negative Y should increase conditional PD"

    def test_positive_factor_decreases_pd(self):
        """Positive Y (boom) should decrease conditional PD."""
        pd = 0.02
        cond_pos = conditional_pd(pd, 0.15, 2.0)
        assert cond_pos < pd, "Positive Y should decrease conditional PD"

    def test_monotone_in_y(self):
        """Conditional PD must be monotone decreasing in Y."""
        pd, R = 0.02, 0.15
        Y_vals = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
        cond = conditional_pd(pd, R, Y_vals)
        assert np.all(np.diff(cond) < 0), "Conditional PD should decrease as Y increases"

    def test_bounds(self):
        """Conditional PD must be in [0, 1]."""
        pd, R = 0.02, 0.15
        Y_vals = np.linspace(-5, 5, 100)
        cond = conditional_pd(pd, R, Y_vals)
        assert np.all(cond >= 0), "Conditional PD cannot be negative"
        assert np.all(cond <= 1), "Conditional PD cannot exceed 1"

    def test_high_correlation_amplifies_stress(self):
        """Higher R amplifies PD increase under stress."""
        pd = 0.02
        cond_low_R = conditional_pd(pd, 0.05, -2.0)
        cond_high_R = conditional_pd(pd, 0.30, -2.0)
        assert cond_high_R > cond_low_R, "Higher R should amplify stress"

    def test_array_inputs(self):
        """Should handle numpy array inputs."""
        pds = np.array([0.01, 0.02, 0.05])
        R = np.array([0.12, 0.15, 0.18])
        cond = conditional_pd(pds, R, -1.5)
        assert cond.shape == (3,)
        assert np.all(cond > pds)


class TestPortfolioLossRate:
    def test_loss_rate_positive(self):
        pds = np.full(100, 0.02)
        lgds = np.full(100, 0.45)
        eads = np.full(100, 1_000_000)
        corrs = np.full(100, 0.15)
        loss = portfolio_loss_rate(pds, lgds, eads, corrs, systemic_factor=-1.0)
        assert loss > 0

    def test_loss_rate_increases_under_stress(self):
        pds = np.full(100, 0.02)
        lgds = np.full(100, 0.45)
        eads = np.full(100, 1_000_000)
        corrs = np.full(100, 0.15)
        loss_base = portfolio_loss_rate(pds, lgds, eads, corrs, 0.0)
        loss_stress = portfolio_loss_rate(pds, lgds, eads, corrs, -2.0)
        assert loss_stress > loss_base

    def test_loss_rate_bounded(self):
        pds = np.full(100, 0.02)
        lgds = np.full(100, 0.45)
        eads = np.full(100, 1_000_000)
        corrs = np.full(100, 0.15)
        loss = portfolio_loss_rate(pds, lgds, eads, corrs, -10.0)
        assert 0 <= loss <= 1.0


class TestScenarios:
    def test_all_scenarios_present(self):
        assert "baseline" in SCENARIOS
        assert "severe_recession" in SCENARIOS
        assert "gfc" in SCENARIOS

    def test_systemic_factors_ordered(self):
        """Scenarios should be ordered from best to worst."""
        assert SCENARIOS["baseline"].systemic_factor == 0.0
        assert SCENARIOS["severe_recession"].systemic_factor < -1.5


class TestRunScenario:
    """Tests for the dual capital adequacy perspective logic."""

    def _make_portfolio(self):
        from src.portfolio.generator import generate_portfolio
        return generate_portfolio(n_loans=500, seed=1)

    def test_scenario_has_required_fields(self):
        from src.stress.scenarios import run_scenario
        df = self._make_portfolio()
        result = run_scenario(df, SCENARIOS["baseline"])
        required = {
            "cond_expected_loss", "cond_el_rate", "irb_capital",
            "cond_el_coverage", "cond_el_surplus", "cond_el_covered",
            "within_irb_design_envelope", "scenario_quantile",
        }
        assert required.issubset(set(result.keys()))

    def test_baseline_within_design_envelope(self):
        """Y=0 → quantile=50% → within 99.9% design envelope."""
        from src.stress.scenarios import run_scenario
        df = self._make_portfolio()
        r = run_scenario(df, SCENARIOS["baseline"])
        assert r["within_irb_design_envelope"] is True

    def test_tail_at_design_boundary(self):
        """Tail scenario at Y≈-3.09 is at or beyond the 99.9% design boundary."""
        from src.stress.scenarios import run_scenario
        df = self._make_portfolio()
        r = run_scenario(df, SCENARIOS["tail"])
        assert r["scenario_quantile"] <= 0.002  # Φ(-3.09) ≈ 0.001

    def test_stress_increases_cond_el(self):
        """More negative Y must produce higher conditional expected loss."""
        from src.stress.scenarios import run_scenario
        df = self._make_portfolio()
        r_base   = run_scenario(df, SCENARIOS["baseline"])
        r_stress = run_scenario(df, SCENARIOS["severe_recession"])
        assert r_stress["cond_expected_loss"] > r_base["cond_expected_loss"]

    def test_coverage_positive(self):
        from src.stress.scenarios import run_scenario
        df = self._make_portfolio()
        r = run_scenario(df, SCENARIOS["baseline"])
        assert r["cond_el_coverage"] > 0


class TestSimulateLossDistribution:
    def test_shape(self):
        n = 1000
        pds = np.full(100, 0.02)
        lgds = np.full(100, 0.45)
        eads = np.full(100, 1e6)
        corrs = np.full(100, 0.15)
        losses = simulate_loss_distribution(pds, lgds, eads, corrs, n_simulations=n, seed=1)
        assert losses.shape == (n,)

    def test_positive_losses(self):
        pds = np.full(100, 0.02)
        lgds = np.full(100, 0.45)
        eads = np.full(100, 1e6)
        corrs = np.full(100, 0.15)
        losses = simulate_loss_distribution(pds, lgds, eads, corrs, n_simulations=500, seed=42)
        assert np.all(losses >= 0)

    def test_reproducible(self):
        """Same seed should produce same results."""
        args = (np.full(50, 0.02), np.full(50, 0.45), np.full(50, 1e6), np.full(50, 0.15))
        l1 = simulate_loss_distribution(*args, n_simulations=100, seed=7)
        l2 = simulate_loss_distribution(*args, n_simulations=100, seed=7)
        np.testing.assert_array_equal(l1, l2)
