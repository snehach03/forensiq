"""
preprocessing.py

Defines which columns become ML features (ratios/scores/flags only —
NOT raw dollar amounts, since scale reflects company size, not fraud risk)
and builds a leakage-safe imputation pipeline.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


NUMERIC_FEATURES = [
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "gross_margin",
    "operating_margin",
    "revenue_growth",
    "receivables_growth",
    "inventory_growth",
    "altman_z_score",
    "piotroski_f_score",
]

BINARY_FEATURES = [
    "receivables_outpacing_revenue",
    "inventory_outpacing_revenue",
    "rule_cashflow_below_income",
    "rule_margin_deteriorating",
    "rule_leverage_spike",
    "rule_altman_distress",
    "rule_beneish_manipulation_flag",
    "rule_piotroski_weak",
    "red_flag_count",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric_impute", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("binary_passthrough", "passthrough", BINARY_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )