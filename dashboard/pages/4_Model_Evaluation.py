"""
4_Model_Evaluation.py

Phase 8, view 4: Model Evaluation.

Design decision - ROC AUR Precision-Recall curve, sirf ROC nahi:
    Humara dataset bahut imbalanced hai (5 fraud / 72 total, ~7% positive
    rate). ROC curve imbalanced data pe kabhi optimistic tasveer dikha
    sakta hai (kyunki True Negatives bahut zyada hain). Precision-Recall
    curve is imbalance ke against zyada honest signal deta hai - isliye
    dono dikha rahe hain, taaki koi ek metric misleading na lage.

Design decision - confusion matrix fixed threshold (0.40) pe:
    Ye wahi threshold hai jo Phase 5 mein choose kiya gaya tha
    (F2-score/Recall priority ke liye) - saved model isi threshold pe
    deploy hota hai, isliye confusion matrix bhi isi pe dikhana
    consistent hai actual model behavior ke saath.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.utils.data_loader import load_loocv_predictions  # noqa: E402

st.set_page_config(page_title="Model Evaluation - ForensIQ", page_icon="📉", layout="wide")
st.title("📉 Model Evaluation")

st.caption(
    "Evaluated via Leave-One-Out Cross-Validation (LOOCV) on 72 company-years "
    "(5 fraud-labeled). Small-sample proof-of-concept metrics - not production-grade."
)

predictions = load_loocv_predictions()

if not predictions:
    st.warning(
        "No LOOCV predictions found. Run `python -m forensiq.ml.save_evaluation_data` "
        "first to generate `models/loocv_predictions.json`."
    )
    st.stop()

y_true = [p["y_true"] for p in predictions]
y_proba = [p["y_pred_proba"] for p in predictions]

DECISION_THRESHOLD = 0.40

st.divider()

# ---- ROC Curve + Precision-Recall Curve side by side ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
        line=dict(dash="dash", color="gray"),
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.5, y=0.05),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col2:
    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall"))
    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    st.plotly_chart(fig_pr, use_container_width=True)

st.divider()

# ---- Confusion Matrix at decision threshold ----
st.subheader(f"Confusion Matrix (threshold = {DECISION_THRESHOLD})")

y_pred = [1 if p >= DECISION_THRESHOLD else 0 for p in y_proba]
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

cm_labels = ["Healthy (0)", "Fraud (1)"]
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=[f"Predicted: {label}" for label in cm_labels],
    y=[f"Actual: {label}" for label in cm_labels],
    text=cm,
    texttemplate="%{text}",
    colorscale="Blues",
))
fig_cm.update_layout(width=500)

col_cm, col_notes = st.columns([1, 1])
with col_cm:
    st.plotly_chart(fig_cm, use_container_width=True)
with col_notes:
    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"""
    - **True Positives:** {tp} (fraud correctly flagged)
    - **False Negatives:** {fn} (fraud missed - costliest error in this domain)
    - **False Positives:** {fp} (healthy company incorrectly flagged)
    - **True Negatives:** {tn} (healthy correctly cleared)

    Threshold={DECISION_THRESHOLD} was chosen to prioritize Recall
    (catching fraud) over Precision, since missed fraud (False Negatives)
    is considered costlier than false alarms in this domain.
    """)

st.divider()

# ---- Raw predictions table, for transparency ----
with st.expander("View raw LOOCV predictions"):
    df = pd.DataFrame(predictions)
    df["predicted_label"] = (df["y_pred_proba"] >= DECISION_THRESHOLD).astype(int)
    st.dataframe(df, use_container_width=True)