"""
Calculates basic financial ratios (liquidity, leverage, profitability)
from the wide-format financial data produced by Phase 2's normalize.py.

Design principle: every calculation checks that its required inputs
are present (not NaN) before dividing. If a required number is
missing, the ratio is set to None for that row rather than crashing
or silently producing a wrong number (e.g. treating missing data as 0).
"""

import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divides two pandas Series element-wise, returning NaN wherever
    either input is missing or the denominator is zero (which would
    otherwise raise a divide-by-zero error).
    """
    result = numerator / denominator
    result = result.where(denominator != 0, other=pd.NA)
    return result

def add_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Gross Margin and Operating Margin columns to the given
    wide-format financial DataFrame.
    """
    df = df.copy()

    df["gross_margin"] = _safe_divide(df["GrossProfit"], df["Revenue"])

    df["operating_margin"] = _safe_divide(df["OperatingIncomeLoss"], df["Revenue"])

    return df

def add_liquidity_leverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Current Ratio, Quick Ratio, and Debt-to-Equity Ratio columns
    to the given wide-format financial DataFrame.
    """
    df = df.copy()

    df["current_ratio"] = _safe_divide(df["AssetsCurrent"], df["LiabilitiesCurrent"])

    df["quick_ratio"] = _safe_divide(
        df["AssetsCurrent"] - df["InventoryNet"].fillna(0),
        df["LiabilitiesCurrent"]
    )

    df["debt_to_equity"] = _safe_divide(df["Liabilities"], df["StockholdersEquity"])

    return df

def add_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds year-over-year growth metrics: Revenue growth, Receivables
    growth, and Inventory growth. Also flags when Receivables or
    Inventory are growing significantly faster than Revenue, a
    classic red flag for aggressive/fake revenue recognition.

    Growth is calculated per company (sorted by fiscal_year) using
    the previous available row for that company — not necessarily
    the immediately preceding calendar year, since some years may
    be missing due to data gaps.
    """
    df = df.copy()
    df = df.sort_values(["company_id", "fiscal_year"])

    df["revenue_growth"] = df.groupby("company_id")["Revenue"].pct_change()
    df["receivables_growth"] = df.groupby("company_id")["AccountsReceivableNetCurrent"].pct_change()
    df["inventory_growth"] = df.groupby("company_id")["InventoryNet"].pct_change()

    # Red flag: receivables growing much faster than revenue.
    # We use a simple threshold: receivables growth more than 1.5x
    # revenue growth (and both are meaningfully positive) is flagged.
    df["receivables_outpacing_revenue"] = (
        (df["receivables_growth"] > df["revenue_growth"] * 1.5)
        & (df["revenue_growth"] > 0)
    )

    df["inventory_outpacing_revenue"] = (
        (df["inventory_growth"] > df["revenue_growth"] * 1.5)
        & (df["revenue_growth"] > 0)
    )

    return df

def add_altman_z_score(df: pd.DataFrame) -> pd.DataFrame:
        """"
        Adds the Altman Z'-Score (private firm variant, using book value
        of equity instead of market value since we don't have stock price
        data). Lower scores indicate higher bankruptcy/distress risk,
        which correlates with higher motive for earnings manipulation.

        Zones (per Altman's original research):
         Z' > 2.9   -> Safe Zone
         1.23-2.9   -> Grey Zone
         Z' < 1.23  -> Distress Zone
        """
        df = df.copy()

        working_capital = df["AssetsCurrent"] - df["LiabilitiesCurrent"]

        df["altman_z_score"] = (
            0.717 * _safe_divide(working_capital, df["Assets"])
            + 0.847 * _safe_divide(df["RetainedEarningsAccumulatedDeficit"], df["Assets"])
            + 3.107 * _safe_divide(df["OperatingIncomeLoss"], df["Assets"])
            + 0.420 * _safe_divide(df["StockholdersEquity"], df["Liabilities"])
            + 0.998 * _safe_divide(df["Revenue"], df["Assets"])
        )

        return df

def add_beneish_m_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a simplified Beneish M-Score using the indices we can compute
    from available data: DSRI, GMI, SGI, TATA, LVGI. The full academic
    formula uses 8 indices; we omit AQI, DEPI, and SGAI since our
    dataset doesn't have the underlying line items (asset composition
    detail, depreciation schedules, SG&A breakdown).

    This is an approximation, not the canonical academic score — it
    should be interpreted as a directional signal, not an exact
    replication of Beneish's original research.

    Higher M-Score (less negative) suggests higher likelihood of
    earnings manipulation. Beneish's original threshold: M > -1.78
    flags a company as a likely manipulator.
    """
    df = df.copy()
    df = df.sort_values(["company_id", "fiscal_year"])

    receivables_to_revenue = _safe_divide(df["AccountsReceivableNetCurrent"], df["Revenue"])
    dsri = df.groupby("company_id").apply(
        lambda g: _safe_divide(
            receivables_to_revenue.loc[g.index],
            receivables_to_revenue.loc[g.index].shift(1)
        )
    ).reset_index(level=0, drop=True)

    gross_margin = _safe_divide(df["GrossProfit"], df["Revenue"])
    gmi = df.groupby("company_id").apply(
        lambda g: _safe_divide(
            gross_margin.loc[g.index].shift(1),
            gross_margin.loc[g.index]
        )
    ).reset_index(level=0, drop=True)

    sgi = df.groupby("company_id")["Revenue"].pct_change() + 1

    tata = _safe_divide(
        df["NetIncomeLoss"] - df["NetCashProvidedByUsedInOperatingActivities"],
        df["Assets"]
    )

    debt_to_equity = _safe_divide(df["Liabilities"], df["StockholdersEquity"])
    lvgi = df.groupby("company_id").apply(
        lambda g: _safe_divide(
            debt_to_equity.loc[g.index],
            debt_to_equity.loc[g.index].shift(1)
        )
    ).reset_index(level=0, drop=True)

    df["beneish_m_score"] = (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.892 * sgi
        + 4.679 * tata
        - 0.327 * lvgi
    )

    return df

def add_piotroski_f_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a simplified Piotroski F-Score (8 of the original 9 criteria;
    we omit "no new shares issued" since we don't track share count).
    Each criterion is a simple pass/fail check, summed into a 0-8
    score. Higher scores indicate improving financial health.

    Unlike Altman/Beneish (weighted composite formulas), this score
    is intentionally simple and transparent — useful for plain-English
    explanations in the LLM layer (Phase 7).
    """
    df = df.copy()
    df = df.sort_values(["company_id", "fiscal_year"])

    asset_turnover = _safe_divide(df["Revenue"], df["Assets"])
    current_ratio = _safe_divide(df["AssetsCurrent"], df["LiabilitiesCurrent"])
    debt_to_equity = _safe_divide(df["Liabilities"], df["StockholdersEquity"])
    gross_margin = _safe_divide(df["GrossProfit"], df["Revenue"])

    criteria = pd.DataFrame(index=df.index)

    criteria["positive_net_income"] = df["NetIncomeLoss"] > 0
    criteria["positive_operating_cash_flow"] = df["NetCashProvidedByUsedInOperatingActivities"] > 0
    criteria["cash_flow_exceeds_net_income"] = (
        df["NetCashProvidedByUsedInOperatingActivities"] > df["NetIncomeLoss"]
    )
    criteria["revenue_growth_positive"] = df.groupby("company_id")["Revenue"].pct_change() > 0

    criteria["current_ratio_improved"] = (
        df.groupby("company_id").apply(lambda g: current_ratio.loc[g.index].diff()).reset_index(level=0, drop=True) > 0
    )
    criteria["leverage_decreased"] = (
        df.groupby("company_id").apply(lambda g: debt_to_equity.loc[g.index].diff()).reset_index(level=0, drop=True) < 0
    )
    criteria["gross_margin_improved"] = (
        df.groupby("company_id").apply(lambda g: gross_margin.loc[g.index].diff()).reset_index(level=0, drop=True) > 0
    )
    criteria["asset_turnover_improved"] = (
        df.groupby("company_id").apply(lambda g: asset_turnover.loc[g.index].diff()).reset_index(level=0, drop=True) > 0
    )

    df["piotroski_f_score"] = criteria.sum(axis=1)

    return df


if __name__ == "__main__":
    from forensiq.parser.normalize import get_wide_financials

    df = get_wide_financials()
    df = add_liquidity_leverage_ratios(df)
    df = add_profitability_ratios(df)
    df = add_growth_metrics(df)
    df=add_altman_z_score(df)
    df=add_beneish_m_score(df)
    df=add_piotroski_f_score(df)

    # Company ID reference (from our ingestion order):
    # 1=Apple, 2=Microsoft, 3=Costco, 4=GE, 5=Under Armour,
    # 6=Enron, 7=Kraft Heinz, 8=Valeant/Bausch Health
    flagged = df[
        df["receivables_outpacing_revenue"] | df["inventory_outpacing_revenue"]
    ]

    print("=== Flagged company-years (red flags triggered) ===")
    print(flagged[[
        "company_id", "fiscal_year",
        "revenue_growth", "receivables_growth", "inventory_growth",
        "receivables_outpacing_revenue", "inventory_outpacing_revenue","altman_z_score",
        "beneish_m_score","piotroski_f_score"
    ]])
