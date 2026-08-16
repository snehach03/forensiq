"""
test_pipeline_integration.py

Phase 9: Integration test - verifies that Phase 3's ratio functions and
Phase 4's rules engine compose correctly when chained together, the way
build_modeling_dataset() actually chains them in the real pipeline.

Why this is different from the unit tests (test_ratios.py, test_rules_
engine.py):
    Unit tests check each function in ISOLATION - they don't catch bugs
    where one function's output doesn't match what the next function
    expects (e.g. a renamed column). This test runs the real chain of
    functions end-to-end on one shared dataset, so a mismatch between
    stages would surface here even if every individual function's own
    unit tests still pass.

Scope: this test does NOT hit the SEC EDGAR API or the database - it
    starts from raw wide-format financial data (the shape Phase 2's
    normalize.py produces) and runs it through the real feature +
    rules functions. Testing all the way from a live SEC API call
    would be slow, network-dependent, and not really "unit-testable" -
    a full end-to-end run is better verified by manually running the
    real pipeline scripts occasionally, not in the automated test suite.

TODO Sneha: fix these import paths to match your actual file locations.
"""

import pandas as pd
import pytest

from forensiq.features.ratio import (  # TODO: fix path
    add_altman_z_score,
    add_beneish_m_score,
    add_growth_metrics,
    add_liquidity_leverage_ratios,
    add_piotroski_f_score,
    add_profitability_ratios,
)
from forensiq.rules.rules_engine import apply_rules  # TODO: fix path


@pytest.fixture
def raw_financials_for_pipeline() -> pd.DataFrame:
    """
    Raw wide-format data (as Phase 2's normalize.py would produce) for
    TWO companies across 2 fiscal years each - deliberately unhealthy
    in Company 1's year 2 (falling margin, rising leverage, weak cash
    flow) so we can verify red flags actually surface after the FULL
    chain runs, not just after one isolated function.

    Note: we use 2 companies (not 1) specifically because pandas'
    groupby().apply() has a quirk where a SINGLE group can return a
    DataFrame instead of a Series, which breaks add_beneish_m_score()'s
    formula assignment. This never happens in the real pipeline (always
    8 companies), but a single-company fixture would trip this pandas
    edge case here - so Company 2 exists purely to keep group count
    realistic and avoid that.
    """
    return pd.DataFrame([
        # Company 1, 2020 - healthy baseline
        {
            "company_id": 1, "fiscal_year": 2020,
            "AssetsCurrent": 500_000, "LiabilitiesCurrent": 200_000,
            "InventoryNet": 50_000, "Liabilities": 800_000,
            "StockholdersEquity": 400_000, "GrossProfit": 400_000,
            "Revenue": 1_000_000, "OperatingIncomeLoss": 150_000,
            "RetainedEarningsAccumulatedDeficit": 300_000, "Assets": 1_200_000,
            "NetIncomeLoss": 100_000, "NetCashProvidedByUsedInOperatingActivities": 120_000,
            "AccountsReceivableNetCurrent": 80_000,
        },
        # Company 1, 2021 - deliberately deteriorating
        {
            "company_id": 1, "fiscal_year": 2021,
            "AssetsCurrent": 480_000, "LiabilitiesCurrent": 210_000,
            "InventoryNet": 55_000, "Liabilities": 1_100_000,  # leverage jump
            "StockholdersEquity": 400_000,
            "GrossProfit": 300_000,  # margin drop vs 2020's 400,000
            "Revenue": 1_050_000, "OperatingIncomeLoss": 90_000,
            "RetainedEarningsAccumulatedDeficit": 280_000, "Assets": 1_150_000,
            "NetIncomeLoss": 95_000,
            "NetCashProvidedByUsedInOperatingActivities": 40_000,  # cash << income
            "AccountsReceivableNetCurrent": 84_000,
        },
        # Company 2, 2020 - healthy baseline (separate company - exists
        # so groupby has 2 groups, avoiding the pandas single-group quirk)
        {
            "company_id": 2, "fiscal_year": 2020,
            "AssetsCurrent": 300_000, "LiabilitiesCurrent": 150_000,
            "InventoryNet": 40_000, "Liabilities": 500_000,
            "StockholdersEquity": 250_000, "GrossProfit": 200_000,
            "Revenue": 600_000, "OperatingIncomeLoss": 60_000,
            "RetainedEarningsAccumulatedDeficit": 100_000, "Assets": 750_000,
            "NetIncomeLoss": 40_000, "NetCashProvidedByUsedInOperatingActivities": 50_000,
            "AccountsReceivableNetCurrent": 45_000,
        },
        # Company 2, 2021 - stays healthy (control group, no red flags expected)
        {
            "company_id": 2, "fiscal_year": 2021,
            "AssetsCurrent": 320_000, "LiabilitiesCurrent": 155_000,
            "InventoryNet": 42_000, "Liabilities": 510_000,
            "StockholdersEquity": 260_000, "GrossProfit": 215_000,
            "Revenue": 650_000, "OperatingIncomeLoss": 65_000,
            "RetainedEarningsAccumulatedDeficit": 110_000, "Assets": 780_000,
            "NetIncomeLoss": 44_000, "NetCashProvidedByUsedInOperatingActivities": 60_000,
            "AccountsReceivableNetCurrent": 48_000,
        },
    ])


def _run_full_feature_and_rules_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the real order build_modeling_dataset() calls these in
    (same order as ratios.py's own __main__ block)."""
    df = add_liquidity_leverage_ratios(df)
    df = add_profitability_ratios(df)
    df = add_growth_metrics(df)
    df = add_altman_z_score(df)
    df = add_beneish_m_score(df)
    df = add_piotroski_f_score(df)
    df = apply_rules(df)
    return df


class TestFullChainRunsCleanly:
    def test_chain_runs_without_error(self, raw_financials_for_pipeline):
        result = _run_full_feature_and_rules_chain(raw_financials_for_pipeline)
        assert len(result) == 4

    def test_all_expected_columns_present_after_chain(self, raw_financials_for_pipeline):
        # If any function renamed/dropped a column the next stage
        # needs, this catches it immediately with a clear list of
        # what's missing, instead of a downstream KeyError deep in
        # apply_rules().
        result = _run_full_feature_and_rules_chain(raw_financials_for_pipeline)
        expected_columns = {
            "current_ratio", "quick_ratio", "debt_to_equity",
            "gross_margin", "operating_margin",
            "revenue_growth", "receivables_growth", "inventory_growth",
            "altman_z_score",
            "rule_cashflow_below_income", "rule_margin_deteriorating",
            "rule_leverage_spike", "rule_altman_distress",
            "red_flag_count", "risk_level",
        }
        missing = expected_columns - set(result.columns)
        assert missing == set(), f"Missing columns after full chain: {missing}"


class TestFullChainProducesCorrectFlags:
    """These re-derive the same red flags we already unit-tested
    individually, but now via the REAL chained functions rather than
    a hand-crafted 'already computed' fixture - confirming the actual
    composition works, not just each piece alone."""

    def test_unhealthy_year_triggers_cashflow_flag(self, raw_financials_for_pipeline):
        result = _run_full_feature_and_rules_chain(raw_financials_for_pipeline)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        # NetCash=40,000 < NetIncome=95,000
        assert row_2021["rule_cashflow_below_income"] == True  # noqa: E712

    def test_unhealthy_year_triggers_margin_deteriorating_flag(self, raw_financials_for_pipeline):
        result = _run_full_feature_and_rules_chain(raw_financials_for_pipeline)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        # gross_margin computed fresh by add_profitability_ratios(),
        # then compared by apply_rules() - both stages must agree.
        assert row_2021["rule_margin_deteriorating"] == True  # noqa: E712

    def test_unhealthy_year_triggers_leverage_spike_flag(self, raw_financials_for_pipeline):
        result = _run_full_feature_and_rules_chain(raw_financials_for_pipeline)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["rule_leverage_spike"] == True  # noqa: E712

    def test_healthy_baseline_year_has_low_red_flag_count(self, raw_financials_for_pipeline):
        result = _run_full_feature_and_rules_chain(raw_financials_for_pipeline)
        row_2020 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row_2020["red_flag_count"] <= 1

    def test_unhealthy_year_has_higher_red_flag_count_than_baseline(self, raw_financials_for_pipeline):
        # The core end-to-end promise: a deliberately deteriorating
        # company-year should score MORE red flags than a healthy one,
        # after flowing through the entire real chain.
        result = _run_full_feature_and_rules_chain(raw_financials_for_pipeline)
        row_2020 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["red_flag_count"] > row_2020["red_flag_count"]