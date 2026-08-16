"""
save_final_model.py

Phase 5 final step: Train the winning model config (Logistic Regression,
threshold=0.40) on the FULL 72-row dataset and persist it to disk.

Why full dataset (not LOOCV)?
    LOOCV was used only to EVALUATE which model/threshold generalizes best.
    Now that we've picked the winner, we retrain on all available data so
    the deployed model has seen everything we know about — same idea as
    a student studying the full syllabus after mock-tests proved the
    method works.

Why save the pipeline (imputer + model) together?
    Downstream consumers (Phase 6 SHAP, Phase 8 Streamlit) will feed in
    raw feature rows that may contain missing values. Bundling the
    imputer with the model means they never have to reimplement or
    accidentally mismatch the preprocessing logic.
"""

import joblib
import json
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from forensiq.ml.build_dataset import build_modeling_dataset  # your Phase 5 orchestrator


# ---- 1. Load the full modeling-ready dataset (all 72 rows) ----
df =build_modeling_dataset()
# TEMPORARY — run once to see real column names, then remove
print(df.columns.tolist())

NUMERIC_FEATURES = [
    "current_ratio", "quick_ratio", "debt_to_equity",
    "gross_margin", "operating_margin",
    "revenue_growth", "receivables_growth", "inventory_growth",
    "altman_z_score", "piotroski_f_score",
    # NOTE: beneish_m_score deliberately EXCLUDED — 78% missing (Phase 5, Step 2).
    # Only its derived binary flag is used below.
]

BINARY_FLAG_FEATURES = [
    "receivables_outpacing_revenue", "inventory_outpacing_revenue",
    "rule_cashflow_below_income", "rule_margin_deteriorating",
    "rule_leverage_spike", "rule_altman_distress",
    "rule_beneish_manipulation_flag", "rule_piotroski_weak",
    "red_flag_count",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FLAG_FEATURES
TARGET_COLUMN = "label"   # confirmed from your df.columns output


X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ---- 2. Rebuild the SAME leakage-safe preprocessing pipeline ----
# (identical structure to what you used inside LOOCV in train_models.py)
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("flags", "passthrough", BINARY_FLAG_FEATURES),
    ]
)

final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
])

# ---- 3. Fit on ALL 72 rows (no train/test split — this is the deployment model) ----
final_pipeline.fit(X, y)

# ---- 4. Save the pipeline ----
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

model_path = MODEL_DIR / "forensiq_logreg_pipeline.joblib"
joblib.dump(final_pipeline, model_path)

# ---- 5. Save metadata alongside it (threshold is a business decision, not part of the model object) ----
metadata = {
    "model_type": "LogisticRegression",
    "decision_threshold": 0.40,
    "selection_metric": "F2-score (beta=2, prioritizes Recall over Precision)",
    "feature_columns": FEATURE_COLUMNS,
    "n_training_rows": len(df),
    "n_positive_labels": int(y.sum()),
    "trained_on": datetime.now().isoformat(),
    "evaluation_summary": {
        "method": "LOOCV (n=72)",
        "recall": 0.60,
        "precision": 0.136,
        "f2_score": 0.357,
        "roc_auc": 0.672,
    },
    "notes": (
        "Small-sample proof-of-concept model (5 positive labels). "
        "Metrics are realistic for this dataset size, not production-grade. "
        "Threshold=0.40 chosen for fraud-forensics domain where False "
        "Negatives (missed fraud) are far costlier than False Positives."
    ),
}

metadata_path = MODEL_DIR / "forensiq_logreg_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Metadata saved to: {metadata_path}")
print(f"   Trained on {len(df)} rows ({int(y.sum())} fraud-labeled)")