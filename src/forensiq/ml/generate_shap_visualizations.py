"""
generate_shap_visualizations.py

Phase 6 (part 2): Global feature-importance bar chart + SHAP summary plot.
Checklist requirement: "Feature importance visualizations"
"""

import joblib
import json
import shap
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from forensiq.ml.build_dataset import build_modeling_dataset

MODEL_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1. Saved model + metadata reuse (retrain nahi karna, Phase 5 se load) ----
pipeline = joblib.load(MODEL_DIR / "forensiq_logreg_pipeline.joblib")
with open(MODEL_DIR / "forensiq_logreg_metadata.json") as f:
    metadata = json.load(f)

FEATURE_COLUMNS = metadata["feature_columns"]

# ---- 2. Data load + preprocess (same as compute_shap_explanations.py) ----
df = build_modeling_dataset()
X = df[FEATURE_COLUMNS]

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]
X_transformed = preprocessor.transform(X)
X_transformed = X_transformed.astype(float) 
X_transformed_df = pd.DataFrame(X_transformed, columns=FEATURE_COLUMNS, index=df.index)

# ---- 3. SHAP values recompute (same LinearExplainer as before) ----
explainer = shap.LinearExplainer(classifier, X_transformed_df)
shap_values = explainer(X_transformed_df)

# ---- 4. Global Feature Importance — Bar Chart ----
global_importance = (
    pd.DataFrame(shap_values.values, columns=FEATURE_COLUMNS)
    .abs()
    .mean()
    .sort_values(ascending=True)  # ascending taaki barh() sabse important ko top pe dikhaye
)

plt.figure(figsize=(9, 7))
plt.barh(global_importance.index, global_importance.values, color="#4C72B0")
plt.xlabel("Mean |SHAP value| (average impact on fraud-risk prediction)")
plt.title("ForensIQ — Global Feature Importance (Logistic Regression)")
plt.tight_layout()
bar_chart_path = FIGURES_DIR / "global_feature_importance.png"
plt.savefig(bar_chart_path, dpi=150)
plt.close()

# ---- 5. SHAP Summary (Beeswarm) Plot ----
plt.figure()
shap.summary_plot(shap_values, X_transformed_df, show=False)
plt.title("ForensIQ — SHAP Summary Plot")
plt.tight_layout()
summary_plot_path = FIGURES_DIR / "shap_summary_plot.png"
plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Bar chart saved: {bar_chart_path}")
print(f"✅ SHAP summary plot saved: {summary_plot_path}")