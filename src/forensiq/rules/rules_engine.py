"""
Deterministic rules engine: combines individual financial signals
(ratios, growth metrics, composite scores from Phase 3) into a set of
boolean red-flag rules and an aggregate risk score per company-year.

Deliberately NOT using ML or LLM here — every rule is a fixed,
explainable threshold check, so results are fully reproducible and
auditable (a requirement stated in the project brief).
"""

import pandas as pd


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds individual rule-violation boolean columns plus an aggregate
    red_flag_count and risk_level, based on features already present
    in df (must have run all Phase 3 feature functions first).
    """
    df = df.copy()
    df = df.sort_values(["company_id", "fiscal_year"])

    # Rule 3: reported profit isn't backed by actual cash.
    df["rule_cashflow_below_income"] = (
        df["NetCashProvidedByUsedInOperatingActivities"] < df["NetIncomeLoss"]
    )

    # Rule 4: gross margin fell versus the prior year.
    gross_margin_diff = df.groupby("company_id")["gross_margin"].diff()
    df["rule_margin_deteriorating"] = gross_margin_diff < 0

    # Rule 5: leverage (Debt-to-Equity) jumped more than 25% YoY.
    debt_to_equity_pct_change = df.groupby("company_id")["debt_to_equity"].pct_change()
    df["rule_leverage_spike"] = debt_to_equity_pct_change > 0.25

    # Rule 6: Altman Z-Score signals distress.
    df["rule_altman_distress"] = df["altman_z_score"] < 1.23

    # Rule 7: Beneish M-Score above the manipulation-likely threshold.
    df["rule_beneish_manipulation_flag"] = df["beneish_m_score"] > -1.78

    # Rule 8: Piotroski F-Score indicates weak overall financial health.
    df["rule_piotroski_weak"] = df["piotroski_f_score"] <= 2

    rule_columns = [
        "receivables_outpacing_revenue",
        "inventory_outpacing_revenue",
        "rule_cashflow_below_income",
        "rule_margin_deteriorating",
        "rule_leverage_spike",
        "rule_altman_distress",
        "rule_beneish_manipulation_flag",
        "rule_piotroski_weak",
    ]

    # Count only rules that could actually be evaluated (ignore NaN,
    # which means "insufficient data", not "rule passed").
    df["red_flag_count"] = df[rule_columns].fillna(False).sum(axis=1)

    df["risk_level"] = pd.cut(
        df["red_flag_count"],
        bins=[-1, 1, 3, 100],
        labels=["Low", "Medium", "High"]
    )

    return df



if __name__ == "__main__":
    from forensiq.parser.normalize import get_wide_financials
    from forensiq.features.ratio import (
        add_liquidity_leverage_ratios,
        add_profitability_ratios,
        add_growth_metrics,
        add_altman_z_score,
        add_beneish_m_score,
        add_piotroski_f_score,
    )

    df = get_wide_financials()
    df = add_liquidity_leverage_ratios(df)
    df = add_profitability_ratios(df)
    df = add_growth_metrics(df)
    df = add_altman_z_score(df)
    df = add_beneish_m_score(df)
    df = add_piotroski_f_score(df)
    df = apply_rules(df)

    print("=== Risk level distribution ===")
    print(df["risk_level"].value_counts())

    print("\n=== Average red_flag_count per company ===")
    print(df.groupby("company_id")["red_flag_count"].mean().sort_values(ascending=False))

    print("\n=== High risk company-years ===")
    print(df[df["risk_level"] == "High"][["company_id", "fiscal_year", "red_flag_count"]])