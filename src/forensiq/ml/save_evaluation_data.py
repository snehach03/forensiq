"""
save_evaluation_data.py

Phase 8 ke Model Evaluation view (ROC Curve, Precision/Recall,
Confusion Matrix) ke liye zaroori raw data generate karta hai.

Kyun dobara LOOCV chalana padta hai:
    forensiq_logreg_metadata.json mein sirf AGGREGATE metrics hain
    (recall=0.60, precision=0.136, etc) - ye already-averaged numbers
    hain. ROC curve ke liye har row ka (true_label, predicted_probability)
    pair chahiye - jo Phase 5 mein kahin persist nahi hua tha, isliye
    yahan dobara compute karke save kar rahe hain.

Kyun same preprocessing pipeline reuse kiya:
    save_final_model.py mein jo ColumnTransformer + LogisticRegression
    structure use hua tha, wahi yahan hai - taaki results consistent
    rahein jo pehle evaluate kiya gaya tha (koi accidental drift na ho
    features ya preprocessing mein).
"""

import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline

from forensiq.ml.build_dataset import build_modeling_dataset

MODEL_DIR = Path("models")
OUTPUT_PATH = MODEL_DIR / "loocv_predictions.json"

# save_final_model.py se exactly wahi feature lists - consistency ke liye.
NUMERIC_FEATURES = [
    "current_ratio", "quick_ratio", "debt_to_equity",
    "gross_margin", "operating_margin",
    "revenue_growth", "receivables_growth", "inventory_growth",
    "altman_z_score", "piotroski_f_score",
]

BINARY_FLAG_FEATURES = [
    "receivables_outpacing_revenue", "inventory_outpacing_revenue",
    "rule_cashflow_below_income", "rule_margin_deteriorating",
    "rule_leverage_spike", "rule_altman_distress",
    "rule_beneish_manipulation_flag", "rule_piotroski_weak",
    "red_flag_count",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FLAG_FEATURES
TARGET_COLUMN = "label"


def run_loocv_and_save():
    df = build_modeling_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("flags", "passthrough", BINARY_FLAG_FEATURES),
        ]
    )

    loo = LeaveOneOut()
    results = []

    print(f"Running LOOCV on {len(df)} rows...")

    for i, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ])
        pipeline.fit(X_train, y_train)

        predicted_proba = pipeline.predict_proba(X_test)[0][1]
        row = df.iloc[test_idx[0]]

        results.append({
            "company_id": int(row["company_id"]),
            "fiscal_year": int(row["fiscal_year"]),
            "y_true": int(y_test.iloc[0]),
            "y_pred_proba": round(float(predicted_proba), 4),
        })

        if i % 10 == 0 or i == len(df):
            print(f"  {i}/{len(df)} rows done")

    MODEL_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved {len(results)} LOOCV predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_loocv_and_save()