"""
build_dataset.py

Orchestrates the full feature pipeline into one final modeling-ready dataframe:
normalize -> ratios (all 6 functions) -> rules engine -> labels

WHY a separate module (not inline in a notebook):
- Reusable: both the ML training script AND the Streamlit dashboard (Phase 8)
  will need this exact same dataset later. Building it once here avoids
  duplicating pipeline logic in two places.
"""

import pandas as pd

from forensiq.parser.normalize import get_wide_financials
from forensiq.features.ratio import (
    add_liquidity_leverage_ratios,
    add_profitability_ratios,
    add_growth_metrics,
    add_altman_z_score,
    add_beneish_m_score,
    add_piotroski_f_score,
)
from forensiq.rules.rules_engine import apply_rules
from forensiq.ml.labels import add_labels


def build_modeling_dataset(form_type: str = "10-K", min_fiscal_year: int = 2016) -> pd.DataFrame:
    """
    Returns the final, fully-featured, labeled dataframe ready for ML.
    One row = one company-fiscal_year.
    """
    df = get_wide_financials(form_type=form_type, min_fiscal_year=min_fiscal_year)

    df = add_liquidity_leverage_ratios(df)
    df = add_profitability_ratios(df)
    df = add_growth_metrics(df)
    df = add_altman_z_score(df)
    df = add_beneish_m_score(df)
    df = add_piotroski_f_score(df)

    df = apply_rules(df)
    df = add_labels(df)

    return df


if __name__ == "__main__":
    # Quick manual sanity check when running this file directly:
    # python -m forensiq.ml.build_dataset
    dataset = build_modeling_dataset()

    print(f"Shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns)}")
    print(f"\nLabel distribution:\n{dataset['label'].value_counts()}")
    print(f"\nMissing values per column:\n{dataset.isna().sum().sort_values(ascending=False)}")