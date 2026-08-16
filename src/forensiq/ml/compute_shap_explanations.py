"""
compute_shap_explanations.py

Phase 6: Har company-year prediction ke liye SHAP values nikalte hain,
taaki "model ne isse risky kyun bola" ka answer mile — sirf probability
nahi, balki feature-level reasoning.
"""

import joblib
import json
import shap
import pandas as pd
from pathlib import Path

from forensiq.ml.build_dataset import build_modeling_dataset


# ---- 1. Saved model + metadata load karo (Phase 5 se, retrain nahi karna) ----
MODEL_DIR = Path("models")
pipeline = joblib.load(MODEL_DIR / "forensiq_logreg_pipeline.joblib")

with open(MODEL_DIR / "forensiq_logreg_metadata.json") as f:
    metadata = json.load(f)

FEATURE_COLUMNS = metadata["feature_columns"]
THRESHOLD = metadata["decision_threshold"]

# ---- 2. Poora dataset dobara load karo (same 72 rows) ----
df = build_modeling_dataset()
X = df[FEATURE_COLUMNS]

# ---- 3. Pipeline ke andar se preprocessor aur model alag nikalo ----
# Kyun: SHAP ko clean/imputed numeric matrix chahiye, raw NaN-wale data nahi.
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

X_transformed = preprocessor.transform(X)
X_transformed_df = pd.DataFrame(X_transformed, columns=FEATURE_COLUMNS, index=df.index)

# ---- 4. SHAP Explainer banao ----
# LinearExplainer kyun: hamara final model Logistic Regression hai (linear),
# isliye ye exact aur computationally cheap explainer hai — koi approximation
# ki zaroorat nahi jaisi TreeExplainer/KernelExplainer mein hoti hai.
explainer = shap.LinearExplainer(classifier, X_transformed_df)
shap_values = explainer(X_transformed_df)

# ---- 5. Har company-year ke liye top contributing features nikalo ----
results = []
for i, row in df.iterrows():
    row_shap = shap_values[df.index.get_loc(i)].values
    feature_contributions = dict(zip(FEATURE_COLUMNS, row_shap))

    # Sabse zyada positive-contribute karne wale (risk badhane wale) top-3
    top_risk_drivers = sorted(
        feature_contributions.items(), key=lambda x: x[1], reverse=True
    )[:3]

    # Sabse zyada negative-contribute karne wale (risk kam karne wale) top-3
    top_risk_reducers = sorted(
        feature_contributions.items(), key=lambda x: x[1]
    )[:3]

    predicted_proba = classifier.predict_proba(
        X_transformed_df.loc[[i]]
    )[0][1]

    results.append({
        "company_id": row["company_id"],
        "fiscal_year": row["fiscal_year"],
        "predicted_risk_probability": round(float(predicted_proba), 3),
        "flagged_as_risky": bool(predicted_proba >= THRESHOLD),
        "top_risk_drivers": [
            {"feature": f, "contribution": round(float(v), 4)} for f, v in top_risk_drivers
        ],
        "top_risk_reducers": [
            {"feature": f, "contribution": round(float(v), 4)} for f, v in top_risk_reducers
        ],
    })

# ---- 6. Per-row explanations save karo ----
output_path = MODEL_DIR / "shap_explanations.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

# ---- 7. Global feature importance bhi save karo (poore dataset ka average) ----
# Kyun: Ye batata hai "overall, kaunse features model ke liye sabse zyada
# matter karte hain" — chahe kisi specific company ka case ho ya na ho.
global_importance = (
    pd.DataFrame(shap_values.values, columns=FEATURE_COLUMNS)
    .abs()
    .mean()
    .sort_values(ascending=False)
)

global_importance_path = MODEL_DIR / "shap_global_importance.json"
global_importance.to_json(global_importance_path, indent=2)

print(f"✅ Per-company explanations saved to: {output_path}")
print(f"✅ Global feature importance saved to: {global_importance_path}")
print("\nTop 5 globally important features:")
print(global_importance.head())