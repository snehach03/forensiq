"""
6_Correlation_Heatmap.py

Phase 8, view 6: Correlation Heatmap.

Design decision - dataset-level view, not per-company:
    Unlike the other pages (Company Explorer, Fraud Risk & Narrative,
    Financial Trends, Peer Comparison), this page looks at relationships
    between FEATURES across the entire dataset, not at one company's
    values. It answers "how related are our 11 ratios/scores to each
    other", which is useful for spotting redundant features and
    sanity-checking that unrelated metrics aren't suspiciously correlated
    (a possible sign of a calculation bug or data leakage).

Design decision - diverging colorscale (RdBu), not "Blues" like the
    confusion matrix:
    Correlation ranges from -1 to +1 and negative vs positive have
    fundamentally different meanings (move together vs move opposite).
    A sequential colorscale (like Blues, used for the confusion matrix's
    non-negative counts) would not visually distinguish a strong negative
    correlation from a near-zero one. A diverging scale centered at 0
    makes both directions immediately readable.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.utils.data_loader import load_modeling_dataset  # noqa: E402

st.set_page_config(page_title="Correlation Heatmap - ForensIQ", page_icon="🧮", layout="wide")
st.title("🧮 Correlation Heatmap")
st.caption(
    "How the model's input features relate to each other across the "
    "full dataset (all companies, all fiscal years combined)."
)

full_dataset = load_modeling_dataset()

# Same metric set used across Financial Trends / Peer Comparison, for
# consistency - these are the features that actually feed the model.
METRIC_OPTIONS = {
    "Revenue Growth (%)": "revenue_growth",
    "Receivables Growth (%)": "receivables_growth",
    "Inventory Growth (%)": "inventory_growth",
    "Gross Margin": "gross_margin",
    "Operating Margin": "operating_margin",
    "Current Ratio": "current_ratio",
    "Quick Ratio": "quick_ratio",
    "Debt-to-Equity": "debt_to_equity",
    "Altman Z-Score": "altman_z_score",
    "Piotroski F-Score": "piotroski_f_score",
    "Red Flag Count": "red_flag_count",
}

available_cols = [col for col in METRIC_OPTIONS.values() if col in full_dataset.columns]
missing_cols = [col for col in METRIC_OPTIONS.values() if col not in full_dataset.columns]

if missing_cols:
    st.caption(f"Note: these columns weren't found in the dataset and are excluded: {missing_cols}")

st.divider()

# Reverse-lookup for readable axis labels instead of raw column names.
label_lookup = {v: k for k, v in METRIC_OPTIONS.items()}
corr_matrix = full_dataset[available_cols].corr()
corr_matrix.index = [label_lookup[c] for c in corr_matrix.index]
corr_matrix.columns = [label_lookup[c] for c in corr_matrix.columns]

fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig.update_layout(
    title="Feature Correlation Matrix (all companies, all years)",
    height=650,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "🔵 Blue = positive correlation (move together)   "
    "🔴 Red = negative correlation (move opposite)   "
    "White ≈ 0 = little to no linear relationship"
)

st.divider()

# ---- Highlight the strongest pairwise correlations, for a quick read ----
st.subheader("Strongest Relationships")

corr_pairs = (
    corr_matrix.where(~corr_matrix.abs().gt(0.999))  # drop the diagonal (self-correlation = 1.0)
    .unstack()
    .dropna()
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

# Each pair appears twice (A-B and B-A) - keep only one direction.
seen = set()
top_pairs = []
for (feat_a, feat_b), value in corr_pairs.items():
    pair_key = tuple(sorted([feat_a, feat_b]))
    if pair_key in seen:
        continue
    seen.add(pair_key)
    top_pairs.append({"Feature A": feat_a, "Feature B": feat_b, "Correlation": round(value, 3)})
    if len(top_pairs) >= 5:
        break

st.dataframe(pd.DataFrame(top_pairs), use_container_width=True, hide_index=True)