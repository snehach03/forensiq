"""
conftest.py

Pytest ye file automatically discover karta hai - yahan defined koi bhi
@pytest.fixture wala function saare test files mein available hota hai
bina import kiye, bas function parameter mein fixture ka naam likh do.

Fixture ka data shape ratios.py ke actual column names match karta hai
(AssetsCurrent, GrossProfit, etc - jo SEC XBRL tags hain), taaki tests
real functions ke against directly chalein.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_financials_df() -> pd.DataFrame:
    """
    2 companies x 2 fiscal years ka fake wide-format financial data -
    exactly wahi shape jo Phase 2's normalize.py produce karta hai.

    2 years isliye rakhe kyunki growth metrics (revenue_growth, etc.)
    pct_change() use karte hain - ek company ke liye kam se kam 2 rows
    chahiye taaki growth calculate ho sake.

    Numbers is tarah chune hain ki:
    - Company 1: healthy growth (revenue_growth=10%, receivables bhi
      thoda badhein lekin proportionate) -> koi red flag nahi
    - Company 2: revenue 10% badha, lekin receivables 62.5% badhe ->
      receivables_outpacing_revenue flag TRUE hona chahiye (classic
      "fake revenue via receivables inflation" pattern)
    """
    return pd.DataFrame([
        # Company 1 - healthy
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
        {
            "company_id": 1, "fiscal_year": 2021,
            "AssetsCurrent": 550_000, "LiabilitiesCurrent": 210_000,
            "InventoryNet": 55_000, "Liabilities": 780_000,
            "StockholdersEquity": 430_000, "GrossProfit": 450_000,
            "Revenue": 1_100_000, "OperatingIncomeLoss": 160_000,
            "RetainedEarningsAccumulatedDeficit": 340_000, "Assets": 1_260_000,
            "NetIncomeLoss": 110_000, "NetCashProvidedByUsedInOperatingActivities": 130_000,
            "AccountsReceivableNetCurrent": 88_000,  # +10% - proportionate to revenue
        },
        # Company 2 - receivables outpacing revenue (potential red flag)
        {
            "company_id": 2, "fiscal_year": 2020,
            "AssetsCurrent": 300_000, "LiabilitiesCurrent": 150_000,
            "InventoryNet": 40_000, "Liabilities": 500_000,
            "StockholdersEquity": 250_000, "GrossProfit": 200_000,
            "Revenue": 600_000, "OperatingIncomeLoss": 60_000,
            "RetainedEarningsAccumulatedDeficit": 100_000, "Assets": 750_000,
            "NetIncomeLoss": 40_000, "NetCashProvidedByUsedInOperatingActivities": 50_000,
            "AccountsReceivableNetCurrent": 80_000,
        },
        {
            "company_id": 2, "fiscal_year": 2021,
            "AssetsCurrent": 320_000, "LiabilitiesCurrent": 160_000,
            "InventoryNet": 42_000, "Liabilities": 520_000,
            "StockholdersEquity": 260_000, "GrossProfit": 210_000,
            "Revenue": 660_000,  # +10% revenue growth
            "OperatingIncomeLoss": 62_000,
            "RetainedEarningsAccumulatedDeficit": 105_000, "Assets": 770_000,
            "NetIncomeLoss": 42_000, "NetCashProvidedByUsedInOperatingActivities": 48_000,
            "AccountsReceivableNetCurrent": 130_000,  # +62.5% - way outpacing revenue
        },
    ])


@pytest.fixture
def sample_rules_input_df() -> pd.DataFrame:
    """
    Pre-computed feature columns (as if Phase 3's ratio functions
    already ran) - apply_rules() only READS these, doesn't compute
    them, so we can hand-craft exact values to trigger specific rules.

    Company 1: 2 fiscal years - 2020 is a clean "no red flags" baseline
      (also lets diff/pct_change-based rules naturally be False since
      there's no prior year), 2021 deliberately triggers ALL 8 rules
      so we can verify each one independently AND the aggregate count.
    Company 2: 1 year, with beneish_m_score = NaN (simulating the ~78%
      real-world missingness) - tests that missing data doesn't crash
      and doesn't get miscounted as a triggered flag.
    Company 3: 1 year, deliberately red_flag_count=2, to test the
      "Medium" risk_level bucket (Company 1's rows only give us
      Low and High examples).
    """
    return pd.DataFrame([
        # Company 1, 2020 - clean baseline, zero flags expected
        {
            "company_id": 1, "fiscal_year": 2020,
            "NetCashProvidedByUsedInOperatingActivities": 150_000, "NetIncomeLoss": 100_000,
            "gross_margin": 0.40, "debt_to_equity": 1.0,
            "altman_z_score": 3.5, "beneish_m_score": -3.0, "piotroski_f_score": 7,
            "receivables_outpacing_revenue": False, "inventory_outpacing_revenue": False,
        },
        # Company 1, 2021 - every single rule deliberately triggered except inventory
        {
            "company_id": 1, "fiscal_year": 2021,
            "NetCashProvidedByUsedInOperatingActivities": 90_000, "NetIncomeLoss": 100_000,  # cash < income
            "gross_margin": 0.35,  # down from 0.40 -> deteriorating
            "debt_to_equity": 1.5,  # +50% vs 1.0 -> leverage spike (>25%)
            "altman_z_score": 1.0,  # < 1.23 -> distress
            "beneish_m_score": -1.0,  # > -1.78 -> manipulation flag
            "piotroski_f_score": 2,  # <= 2 -> weak
            "receivables_outpacing_revenue": True, "inventory_outpacing_revenue": False,
        },
        # Company 2, 2020 - single year, beneish_m_score missing (NaN)
        {
            "company_id": 2, "fiscal_year": 2020,
            "NetCashProvidedByUsedInOperatingActivities": 80_000, "NetIncomeLoss": 100_000,  # cash < income
            "gross_margin": 0.30, "debt_to_equity": 2.0,
            "altman_z_score": 0.9,  # distress
            "beneish_m_score": float("nan"),  # missing data - should NOT count as a flag
            "piotroski_f_score": 1,  # weak
            "receivables_outpacing_revenue": True, "inventory_outpacing_revenue": True,
        },
        # Company 3, 2020 - exactly 2 flags triggered, for testing "Medium" risk_level
        {
            "company_id": 3, "fiscal_year": 2020,
            "NetCashProvidedByUsedInOperatingActivities": 50_000, "NetIncomeLoss": 100_000,  # cash < income
            "gross_margin": 0.50, "debt_to_equity": 0.8,
            "altman_z_score": 1.0,  # distress
            "beneish_m_score": -3.0, "piotroski_f_score": 6,
            "receivables_outpacing_revenue": False, "inventory_outpacing_revenue": False,
        },
    ])


@pytest.fixture
def sample_llm_snapshot() -> dict:
    """
    A fake company snapshot dict, shaped like what build_company_snapshot()
    (Phase 7) produces - used to test the hallucination guard's
    _allowed_base_values() / _is_close_to_any() / _flag_unrecognized_numbers().

    Numbers deliberately include a mix of a ratio (0.65), a count (5),
    SHAP contributions (small decimals), and a rule-text with an
    embedded number (25% threshold) - covering every source
    _allowed_base_values() pulls from.
    """
    return {
        "company_name": "Test Co",
        "fiscal_year": 2021,
        "risk_score": 0.65,
        "red_flag_count": 5,
        "shap_top_drivers": [
            {"feature": "debt_to_equity", "contribution": 0.23},
            {"feature": "altman_z_score", "contribution": 0.15},
        ],
        "shap_top_reducers": [
            {"feature": "current_ratio", "contribution": -0.10},
        ],
        "triggered_rules": [
            "Leverage spiked more than 25% YoY",
            "Altman Z-Score below 1.23 (distress zone)",
        ],
    }