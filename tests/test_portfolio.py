"""
Tests for the synthetic portfolio generator and analytics module.
"""

import numpy as np
import pandas as pd
import pytest
from src.portfolio.generator import generate_portfolio, INDUSTRY_CONFIGS
from src.portfolio.analytics import portfolio_summary, industry_breakdown, pd_distribution_bins


REQUIRED_COLUMNS = {"loan_id", "industry", "pd", "lgd", "ead", "maturity", "correlation", "el", "K", "K_no_ma", "MA", "capital", "rwa", "req_capital"}


@pytest.fixture(scope="module")
def small_portfolio():
    return generate_portfolio(n_loans=500, seed=1)


@pytest.fixture(scope="module")
def full_portfolio():
    return generate_portfolio(n_loans=1_000, seed=42)


class TestPortfolioGenerator:
    def test_returns_dataframe(self, small_portfolio):
        assert isinstance(small_portfolio, pd.DataFrame)

    def test_correct_columns(self, small_portfolio):
        assert REQUIRED_COLUMNS.issubset(set(small_portfolio.columns))

    def test_correct_row_count(self):
        df = generate_portfolio(n_loans=200, seed=1)
        assert len(df) == 200

    def test_pd_bounds(self, small_portfolio):
        assert small_portfolio["pd"].between(0.0003, 0.10).all(), "PDs out of bounds"

    def test_lgd_bounds(self, small_portfolio):
        assert small_portfolio["lgd"].between(0.20, 0.60).all(), "LGDs out of bounds"

    def test_ead_positive(self, small_portfolio):
        assert (small_portfolio["ead"] > 0).all()

    def test_all_industries_present(self, full_portfolio):
        expected = {cfg.name for cfg in INDUSTRY_CONFIGS}
        present = set(full_portfolio["industry"].unique())
        assert expected == present, f"Missing industries: {expected - present}"

    def test_maturity_bounds(self, small_portfolio):
        """Maturity must be in Basel-specified range [1, 5] years."""
        assert small_portfolio["maturity"].between(1.0, 5.0).all()

    def test_k_ge_k_no_ma(self, small_portfolio):
        """K (maturity-adjusted) must be >= K_no_ma for all M >= 1."""
        assert (small_portfolio["K"] >= small_portfolio["K_no_ma"] - 1e-10).all()

    def test_ma_gt_one(self, small_portfolio):
        """Maturity adjustment must be > 1 for all maturities > 1 year."""
        loans_above_1yr = small_portfolio[small_portfolio["maturity"] > 1.01]
        assert (loans_above_1yr["MA"] > 1.0).all()
        assert (small_portfolio["capital"] >= 0).all()

    def test_el_formula(self, small_portfolio):
        """EL row values should all be positive and bounded above by EAD."""
        assert (small_portfolio["el"] > 0).all()
        assert (small_portfolio["el"] <= small_portfolio["ead"]).all()

    def test_rwa_relationship(self, small_portfolio):
        """RWA = 12.5 × Capital."""
        expected_rwa = 12.5 * small_portfolio["capital"]
        pd.testing.assert_series_equal(
            small_portfolio["rwa"].round(2),
            expected_rwa.round(2),
            check_names=False,
        )

    def test_reproducible(self):
        """Same seed → same portfolio."""
        df1 = generate_portfolio(n_loans=100, seed=99)
        df2 = generate_portfolio(n_loans=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_portfolio(n_loans=100, seed=1)
        df2 = generate_portfolio(n_loans=100, seed=2)
        assert not df1["pd"].equals(df2["pd"])


class TestPortfolioAnalytics:
    def test_summary_keys(self, full_portfolio):
        summary = portfolio_summary(full_portfolio)
        required_keys = {
            "n_loans", "total_ead", "total_el", "total_capital",
            "total_rwa", "el_rate", "capital_rate", "wtd_avg_pd",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_summary_n_loans(self, full_portfolio):
        summary = portfolio_summary(full_portfolio)
        assert summary["n_loans"] == len(full_portfolio)

    def test_el_rate_positive_small(self, full_portfolio):
        summary = portfolio_summary(full_portfolio)
        assert 0 < summary["el_rate"] < 0.05, "EL rate should be small but positive"

    def test_capital_rate_gt_el_rate(self, full_portfolio):
        summary = portfolio_summary(full_portfolio)
        assert summary["capital_rate"] > summary["el_rate"], "Capital should exceed EL"

    def test_industry_breakdown_index(self, full_portfolio):
        ind = industry_breakdown(full_portfolio)
        assert ind.index.name == "industry"

    def test_industry_breakdown_totals(self, full_portfolio):
        ind = industry_breakdown(full_portfolio)
        assert abs(ind["total_ead"].sum() - full_portfolio["ead"].sum()) < 1

    def test_pd_histogram_coverage(self, full_portfolio):
        hist = pd_distribution_bins(full_portfolio)
        assert hist["n_loans"].sum() == len(full_portfolio)
