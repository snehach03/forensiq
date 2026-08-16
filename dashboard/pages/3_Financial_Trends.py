"""
3_Financial_Trends.py

Phase 8, view 3: Financial Trends.

Design decision - Plotly, matplotlib nahi:
    Plotly ke charts interactive hain (hover pe values dikhte hain, zoom
    kar sakte ho) - dashboard mein ye matplotlib ke static images se
    kaafi better user-experience deta hai, aur Streamlit ke saath
    first-class integration hai (st.plotly_chart).

Design decision - ek dropdown se metric choose karna, sab metrics ek
saath nahi dikhana:
    Financial ratios ke units alag-alag hain (kuch percentage, kuch
    ratio, kuch score) - sabko ek hi chart mein daalne se scale
    mismatch ho jaata (jaise debt_to_equity 0-2 range mein, revenue_growth
    -50 se +50% range mein) - graph unreadable ho jaata. Isliye user
    khud choose karta hai kaunsa metric dekhna hai.
"""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.utils.data_loader import get_company_list, load_modeling_dataset  # noqa: E402

st.set_page_config(page_title="Financial Trends - ForensIQ", page_icon="📈", layout="wide")
st.title("📈 Financial Trends")

companies_df = get_company_list()
full_dataset = load_modeling_dataset()

# Session state se company selection reuse karo (Company Explorer se), warna
# fallback dropdown.
if "selected_company_id" not in st.session_state:
    selected_name = st.selectbox("Select a company", options=companies_df["name"].tolist())
    selected_company_id = int(companies_df[companies_df["name"] == selected_name].iloc[0]["id"])
else:
    selected_company_id = st.session_state["selected_company_id"]
    selected_name = st.session_state["selected_company_name"]
    st.caption(f"Showing: **{selected_name}** (change selection in Company Explorer)")

company_history = full_dataset[full_dataset["company_id"] == selected_company_id].sort_values("fiscal_year")

if company_history.empty or len(company_history) < 2:
    st.info(
        "Not enough multi-year data to plot a trend for this company "
        "(need at least 2 fiscal years)."
    )
    st.stop()

st.divider()

# Readable labels for the metric dropdown - raw column names avoid karte
# hain UI mein, jaisa humne Phase 7 mein bhi kiya tha LLM prompt ke liye.
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

selected_metric_label = st.selectbox("Choose a metric to plot", options=list(METRIC_OPTIONS.keys()))
selected_metric_col = METRIC_OPTIONS[selected_metric_label]

fig = px.line(
    company_history,
    x="fiscal_year",
    y=selected_metric_col,
    markers=True,
    title=f"{selected_metric_label} — {selected_name}",
    labels={"fiscal_year": "Fiscal Year", selected_metric_col: selected_metric_label},
)
fig.update_layout(hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Raw Data")
display_cols = ["fiscal_year"] + list(METRIC_OPTIONS.values())
st.dataframe(
    company_history[display_cols].set_index("fiscal_year"),
    use_container_width=True,
)