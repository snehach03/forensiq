"""
5_Peer_Comparison.py

Phase 8, view 5: Peer Comparison.

Design decision - isolated numbers vs peer comparison:
    Ek company ka ratio (jaise debt_to_equity=1.8) apne aap mein
    meaningless hai jab tak koi baseline na ho. Forensic accounting
    mein industry/peer comparison standard practice hai - "is number
    ka matlab hai bura hai" sirf tab pata chalta hai jab similar
    companies se compare karo. Isliye ye page absolute thresholds pe
    depend nahi karta, relative comparison dikhata hai.

Design decision - latest-year snapshot bar chart, timeline nahi:
    Per-company timeline already page 2 (Fraud Risk & Narrative) mein
    hai. Yahan primary question "abhi kaun zyada risky/healthy hai
    doosron ke comparison mein" - isliye grouped bar chart (ek time
    point, multiple companies) zyada useful hai yahan.

Design decision - same METRIC_OPTIONS jo Financial Trends page mein
    hain:
    Consistency ke liye - user ko dono pages pe same metric names/
    ordering milegi, cognitive load kam hota hai.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.utils.data_loader import get_company_list, load_modeling_dataset  # noqa: E402

st.set_page_config(page_title="Peer Comparison - ForensIQ", page_icon="⚖️", layout="wide")
st.title("⚖️ Peer Comparison")
st.caption(
    "Compare companies side-by-side on the same metric. A ratio only "
    "means something relative to peers - this view provides that context."
)

companies_df = get_company_list()
full_dataset = load_modeling_dataset()

st.divider()

# ---- Company multi-select (minimum 2 needed for a comparison) ----
default_selection = companies_df["name"].tolist()[:3]
selected_names = st.multiselect(
    "Select companies to compare (choose at least 2)",
    options=companies_df["name"].tolist(),
    default=default_selection,
)

if len(selected_names) < 2:
    st.info("Please select at least 2 companies to compare.")
    st.stop()

selected_ids = companies_df[companies_df["name"].isin(selected_names)]["id"].tolist()

# ---- Metric dropdown - same options as Financial Trends page, for consistency ----
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

selected_metric_label = st.selectbox("Choose a metric to compare", options=list(METRIC_OPTIONS.keys()))
selected_metric_col = METRIC_OPTIONS[selected_metric_label]

st.divider()

# ---- Build latest-year-per-company comparison table ----
comparison_rows = []
for company_id in selected_ids:
    company_rows = full_dataset[full_dataset["company_id"] == company_id]
    if company_rows.empty:
        continue
    latest_row = company_rows.sort_values("fiscal_year").iloc[-1]
    company_name = companies_df[companies_df["id"] == company_id].iloc[0]["name"]
    comparison_rows.append(
        {
            "Company": company_name,
            "Fiscal Year": int(latest_row["fiscal_year"]),
            selected_metric_label: latest_row[selected_metric_col],
        }
    )

if not comparison_rows:
    st.warning("No modeling data available for the selected companies.")
    st.stop()

comparison_df = pd.DataFrame(comparison_rows)

# ---- Grouped bar chart ----
st.subheader(f"{selected_metric_label} — Latest Available Year")

fig = px.bar(
    comparison_df,
    x="Company",
    y=selected_metric_label,
    color="Company",
    text=selected_metric_label,
    hover_data=["Fiscal Year"],
)
fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Note: each company's most recent available fiscal year is shown - "
    "these years may differ across companies depending on data availability."
)

st.divider()

# ---- Raw data table ----
st.subheader("Raw Data")
st.dataframe(comparison_df.set_index("Company"), use_container_width=True)