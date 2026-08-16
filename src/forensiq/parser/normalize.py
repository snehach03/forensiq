"""
Converts financial_facts (long format: one row per concept per period)
into a wide format (one row per company-period, concepts as columns)
that Phase 3 feature engineering can directly use for calculations.
"""

import pandas as pd
from forensiq.db.base import engine

# Balance Sheet concepts are "instant" (a snapshot at period_end) —
# they have no period_start, so a duration filter doesn't apply to them.
# Income Statement / Cash Flow concepts are "duration" (measured over
# period_start to period_end) — these DO need the duration filter to
# exclude quarterly figures mislabeled as annual.
INSTANT_CONCEPTS = {
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "AccountsReceivableNetCurrent",
    "InventoryNet",
    "PropertyPlantAndEquipmentNet",
    "RetainedEarningsAccumulatedDeficit",
}


def get_wide_financials(form_type: str = "10-K", min_fiscal_year: int = 2016) -> pd.DataFrame:
    """
    Pulls all financial_facts from the database and pivots them into
    wide format: one row per (company_id, fiscal_year), one column
    per financial concept.

    min_fiscal_year defaults to 2016: XBRL tagging was inconsistent
    across companies before then, so we restrict to years where data
    quality is reliable.
    """
    query = f"""
        SELECT company_id, concept, fiscal_year, fiscal_period, value, period_start, period_end
        FROM financial_facts
        WHERE form_type = '{form_type}' AND fiscal_year >= {min_fiscal_year}
    """
    df = pd.read_sql(query, engine)

    is_instant = df["concept"].isin(INSTANT_CONCEPTS)

    df["duration_days"] = (
        pd.to_datetime(df["period_end"]) - pd.to_datetime(df["period_start"])
    ).dt.days

    df = df[is_instant | (df["duration_days"] > 300)]

    wide_df = df.pivot_table(
        index=["company_id", "fiscal_year"],
        columns="concept",
        values="value",
        aggfunc="first"
    )

    wide_df = wide_df.reset_index()

    if "CostOfGoodsAndServicesSold" in wide_df.columns:
        wide_df = wide_df.drop(columns=["CostOfGoodsAndServicesSold"])

    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
    ]
    existing_revenue_tags = [tag for tag in revenue_tags if tag in wide_df.columns]

    if existing_revenue_tags:
        wide_df["Revenue"] = wide_df[existing_revenue_tags[0]]
        for tag in existing_revenue_tags[1:]:
            wide_df["Revenue"] = wide_df["Revenue"].fillna(wide_df[tag])
        wide_df = wide_df.drop(columns=existing_revenue_tags)

    return wide_df


if __name__ == "__main__":
    result = get_wide_financials()
    print(result.head(10))
    print("\nShape:", result.shape)
    print("\nColumns:", list(result.columns))
    print("\nMissing data per column:\n", result.isna().sum())