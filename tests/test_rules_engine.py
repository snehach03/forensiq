"""
test_rules_engine.py

Phase 9: Unit tests for the rules engine (apply_rules) - Phase 4's
8 deterministic red-flag rules + aggregate red_flag_count + risk_level.

Testing approach same as test_ratios.py: fixture has hand-crafted
values where we know EXACTLY which rules should fire, so every
assertion is checkable by hand, not just "did it run without crashing".

TODO Sneha: fix this import to match your actual file location,
e.g. if the file is src/forensiq/rules/rules_engine.py:
"""

import pandas as pd

from forensiq.rules.rules_engine import apply_rules  # TODO: fix path


class TestIndividualRules:
    """Company 1 has a clean 2020 baseline (no flags) and a 2021 row
    where every rule was deliberately engineered to trigger (except
    inventory_outpacing_revenue) - each test below checks ONE rule in
    isolation, so a failure points straight at which rule broke."""

    def test_cashflow_below_income_true_when_cash_less_than_income(self, sample_rules_input_df):
        result = apply_rules(sample_rules_input_df)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["rule_cashflow_below_income"] == True  # noqa: E712

    def test_cashflow_below_income_false_when_cash_covers_income(self, sample_rules_input_df):
        result = apply_rules(sample_rules_input_df)
        row_2020 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row_2020["rule_cashflow_below_income"] == False  # noqa: E712

    def test_margin_deteriorating_true_when_gross_margin_drops(self, sample_rules_input_df):
        # Company 1: gross_margin 0.40 (2020) -> 0.35 (2021), a drop
        result = apply_rules(sample_rules_input_df)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["rule_margin_deteriorating"] == True  # noqa: E712

    def test_margin_deteriorating_false_on_first_year_no_prior_data(self, sample_rules_input_df):
        # No prior year to compare against -> diff() is NaN -> rule
        # should be False, not crash and not silently count as True.
        result = apply_rules(sample_rules_input_df)
        row_2020 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row_2020["rule_margin_deteriorating"] == False  # noqa: E712

    def test_leverage_spike_true_when_debt_to_equity_jumps_over_25pct(self, sample_rules_input_df):
        # Company 1: debt_to_equity 1.0 -> 1.5 = +50%, above the 25% threshold
        result = apply_rules(sample_rules_input_df)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["rule_leverage_spike"] == True  # noqa: E712

    def test_altman_distress_true_below_threshold(self, sample_rules_input_df):
        # 1.0 < 1.23 -> distress zone
        result = apply_rules(sample_rules_input_df)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["rule_altman_distress"] == True  # noqa: E712

    def test_altman_distress_false_above_threshold(self, sample_rules_input_df):
        # Company 1, 2020: altman_z_score=3.5, well above 1.23
        result = apply_rules(sample_rules_input_df)
        row_2020 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row_2020["rule_altman_distress"] == False  # noqa: E712

    def test_beneish_manipulation_flag_true_above_threshold(self, sample_rules_input_df):
        # -1.0 > -1.78 -> flagged
        result = apply_rules(sample_rules_input_df)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["rule_beneish_manipulation_flag"] == True  # noqa: E712

    def test_piotroski_weak_true_at_or_below_two(self, sample_rules_input_df):
        # piotroski_f_score=2, threshold is "<= 2"
        result = apply_rules(sample_rules_input_df)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["rule_piotroski_weak"] == True  # noqa: E712


class TestMissingDataHandling:
    """Real SEC data has gaps (e.g. beneish_m_score is ~78% missing per
    the project's own feature-engineering notes) - this is the most
    important edge case to get right, since a bug here would silently
    mis-score real companies, not just crash loudly."""

    def test_nan_feature_does_not_crash(self, sample_rules_input_df):
        # Should simply run without raising - Company 2 has
        # beneish_m_score = NaN.
        result = apply_rules(sample_rules_input_df)
        assert len(result) == len(sample_rules_input_df)

    def test_nan_feature_treated_as_flag_not_triggered(self, sample_rules_input_df):
        # Missing data means "we don't know", not "manipulation
        # detected" - the rule should resolve to False, not True,
        # when the underlying score is NaN.
        result = apply_rules(sample_rules_input_df)
        row = result[(result["company_id"] == 2) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["rule_beneish_manipulation_flag"] == False  # noqa: E712


class TestAggregateScoring:
    def test_red_flag_count_sums_correctly(self, sample_rules_input_df):
        # Company 1, 2021: 7 of 8 rules trigger (all except
        # inventory_outpacing_revenue) - hand-counted from the fixture.
        result = apply_rules(sample_rules_input_df)
        row_2021 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row_2021["red_flag_count"] == 7

    def test_red_flag_count_zero_for_clean_baseline(self, sample_rules_input_df):
        result = apply_rules(sample_rules_input_df)
        row_2020 = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row_2020["red_flag_count"] == 0

    def test_risk_level_low_for_zero_flags(self, sample_rules_input_df):
        result = apply_rules(sample_rules_input_df)
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["risk_level"] == "Low"

    def test_risk_level_medium_for_two_flags(self, sample_rules_input_df):
        # Company 3 was crafted to trigger exactly 2 flags -> Medium bucket
        result = apply_rules(sample_rules_input_df)
        row = result[(result["company_id"] == 3) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["red_flag_count"] == 2
        assert row["risk_level"] == "Medium"

    def test_risk_level_high_for_many_flags(self, sample_rules_input_df):
        result = apply_rules(sample_rules_input_df)
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row["risk_level"] == "High"


class TestDataIntegrity:
    def test_does_not_mutate_input_df(self, sample_rules_input_df):
        original_columns = set(sample_rules_input_df.columns)
        apply_rules(sample_rules_input_df)
        assert set(sample_rules_input_df.columns) == original_columns

    def test_rules_evaluated_per_company_not_across_companies(self, sample_rules_input_df):
        # Company 2 and Company 3 each have only 1 fiscal year, but
        # they're different companies - diff/pct_change-based rules
        # (margin_deteriorating, leverage_spike) must NOT compare
        # Company 2's row against Company 3's row. Both should show
        # False (no prior year within their own company).
        result = apply_rules(sample_rules_input_df)
        company_2_row = result[result["company_id"] == 2].iloc[0]
        company_3_row = result[result["company_id"] == 3].iloc[0]
        assert company_2_row["rule_margin_deteriorating"] == False  # noqa: E712
        assert company_3_row["rule_margin_deteriorating"] == False  # noqa: E712