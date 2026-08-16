"""
test_ratios.py

Phase 9: Unit tests for ratios.py (Phase 3's financial ratio calculations).

Testing philosophy:
    Har function ke liye hum EXACT expected numbers hand-calculate karte
    hain (fixture ke known inputs se), phir compare karte hain function
    ke actual output se. Ye "just check it doesn't crash" se zyada
    strong hai - agar formula mein kabhi typo/sign-error aa jaaye
    (jaise + ki jagah - ho jaaye), ye turant fail hoga.

    Edge cases (zero denominator, missing data) alag se test karte hain
    kyunki ye wahi jagah hai jahan real bugs chhup te hain - normal case
    to aksar sahi hota hai, edge case mein hi crash/wrong-number aata hai.
"""

import numpy as np
import pandas as pd
import pytest

# TODO Sneha: apne actual module path ke hisaab se ye import fix karo,
# jaise agar file src/forensiq/features/ratios.py hai to:
from forensiq.features.ratio import (
    _safe_divide,
    add_altman_z_score,
    add_growth_metrics,
    add_liquidity_leverage_ratios,
    add_profitability_ratios,
)


class TestSafeDivide:
    """_safe_divide sabse zyada reused helper hai - baaki sab ratios isi
    pe depend karte hain, isliye isko sabse thoroughly test karte hain."""

    def test_normal_division(self):
        numerator = pd.Series([10.0, 20.0])
        denominator = pd.Series([2.0, 4.0])
        result = _safe_divide(numerator, denominator)
        assert result.tolist() == [5.0, 5.0]

    def test_zero_denominator_returns_na_not_crash(self):
        # Ye sabse important edge case hai - agar ye handle na ho,
        # asli SEC data mein kisi company ka koi liability zero hone
        # pe pura pipeline crash ho jaayega.
        numerator = pd.Series([10.0])
        denominator = pd.Series([0.0])
        result = _safe_divide(numerator, denominator)
        assert pd.isna(result.iloc[0])

    def test_missing_numerator_propagates_na(self):
        numerator = pd.Series([np.nan])
        denominator = pd.Series([5.0])
        result = _safe_divide(numerator, denominator)
        assert pd.isna(result.iloc[0])


class TestProfitabilityRatios:
    def test_gross_margin_calculation(self, sample_financials_df):
        result = add_profitability_ratios(sample_financials_df)
        # Company 1, 2020: GrossProfit=400,000 / Revenue=1,000,000 = 0.4
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["gross_margin"] == pytest.approx(0.4)

    def test_operating_margin_calculation(self, sample_financials_df):
        result = add_profitability_ratios(sample_financials_df)
        # Company 1, 2020: OperatingIncomeLoss=150,000 / Revenue=1,000,000 = 0.15
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["operating_margin"] == pytest.approx(0.15)

    def test_does_not_mutate_input_df(self, sample_financials_df):
        # ratios.py har function mein df.copy() karta hai - ye confirm
        # karta hai ki wo promise actually honor ho raha hai. Agar
        # accidentally copy() hata diya jaaye kabhi, ye test fail
        # hoga aur turant pata chal jaayega (silent bug avoid).
        original_columns = set(sample_financials_df.columns)
        add_profitability_ratios(sample_financials_df)
        assert set(sample_financials_df.columns) == original_columns


class TestLiquidityLeverageRatios:
    def test_current_ratio(self, sample_financials_df):
        result = add_liquidity_leverage_ratios(sample_financials_df)
        # Company 1, 2020: AssetsCurrent=500,000 / LiabilitiesCurrent=200,000 = 2.5
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["current_ratio"] == pytest.approx(2.5)

    def test_quick_ratio_excludes_inventory(self, sample_financials_df):
        result = add_liquidity_leverage_ratios(sample_financials_df)
        # Company 1, 2020: (500,000 - 50,000) / 200,000 = 2.25
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["quick_ratio"] == pytest.approx(2.25)

    def test_debt_to_equity(self, sample_financials_df):
        result = add_liquidity_leverage_ratios(sample_financials_df)
        # Company 1, 2020: Liabilities=800,000 / StockholdersEquity=400,000 = 2.0
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["debt_to_equity"] == pytest.approx(2.0)

    def test_quick_ratio_handles_missing_inventory_as_zero(self, sample_financials_df):
        # ratios.py mein InventoryNet.fillna(0) hai quick_ratio mein -
        # matlab agar inventory data missing hai, wo 0 treat hota hai
        # (not the whole ratio becoming NaN). Ye deliberate design
        # decision hai, test se lock kar rahe hain taaki accidentally
        # na badle.
        df = sample_financials_df.copy()
        df.loc[0, "InventoryNet"] = np.nan
        result = add_liquidity_leverage_ratios(df)
        row = result.iloc[0]
        # (500,000 - 0) / 200,000 = 2.5, same as current_ratio when
        # inventory is treated as missing/zero
        assert row["quick_ratio"] == pytest.approx(2.5)


class TestGrowthMetrics:
    def test_first_year_growth_is_nan(self, sample_financials_df):
        # Pehle available year ke liye "previous year" hai hi nahi,
        # isliye growth NaN hona chahiye, 0 nahi (0 ek galat signal
        # dega - "no growth" vs "no data" alag cheezein hain).
        result = add_growth_metrics(sample_financials_df)
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert pd.isna(row["revenue_growth"])

    def test_revenue_growth_calculation(self, sample_financials_df):
        result = add_growth_metrics(sample_financials_df)
        # Company 1: 1,000,000 -> 1,100,000 = +10%
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row["revenue_growth"] == pytest.approx(0.10)

    def test_receivables_outpacing_revenue_flag_true(self, sample_financials_df):
        # Company 2: revenue +10%, receivables +62.5% -> 62.5% > 1.5*10%=15%
        # aur revenue_growth > 0 - dono conditions true, flag TRUE expected.
        # Ye red-flag rule ka core fraud-signal hai (fake revenue via
        # inflated receivables) - isliye especially important test hai.
        result = add_growth_metrics(sample_financials_df)
        row = result[(result["company_id"] == 2) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row["receivables_outpacing_revenue"] == True  # noqa: E712

    def test_receivables_outpacing_revenue_flag_false_for_healthy_company(self, sample_financials_df):
        # Company 1: revenue +10%, receivables +10% - proportionate
        # growth, flag FALSE expected.
        result = add_growth_metrics(sample_financials_df)
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2021)].iloc[0]
        assert row["receivables_outpacing_revenue"] == False  # noqa: E712

    def test_growth_calculated_per_company_not_across_companies(self, sample_financials_df):
        # Critical bug-prevention test: agar groupby("company_id") kahin
        # accidentally hat jaaye, growth Company 1's 2021 aur Company 2's
        # 2020 ke beech calculate ho jaayega - completely meaningless
        # number, lekin silently wrong (crash nahi hoga, bas galat
        # answer aayega). Ye test isko catch karega.
        result = add_growth_metrics(sample_financials_df)
        company_2_first_year = result[
            (result["company_id"] == 2) & (result["fiscal_year"] == 2020)
        ].iloc[0]
        assert pd.isna(company_2_first_year["revenue_growth"])


class TestAltmanZScore:
    def test_altman_z_score_matches_hand_calculation(self, sample_financials_df):
        # Company 1, 2020 ke liye hand-calculated expected value:
        # working_capital = 500,000 - 200,000 = 300,000
        # Z = 0.717*(300000/1200000) + 0.847*(300000/1200000)
        #   + 3.107*(150000/1200000) + 0.420*(400000/800000)
        #   + 0.998*(1000000/1200000)
        #   = 0.17925 + 0.21175 + 0.388375 + 0.21 + 0.831667
        #   ≈ 1.82104
        result = add_altman_z_score(sample_financials_df)
        row = result[(result["company_id"] == 1) & (result["fiscal_year"] == 2020)].iloc[0]
        assert row["altman_z_score"] == pytest.approx(1.82104, abs=0.001)