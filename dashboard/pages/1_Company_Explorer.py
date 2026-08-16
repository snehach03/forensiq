"""
1_Company_Explorer.py

Phase 8, view 1: Company Explorer.

Design decision - session_state mein selected company store karna:
    User ek baar company select kare, toh baaki pages (jab tak Streamlit
    multi-page navigation use ho) usi selection ko yaad rakhein - taaki
    har page pe dobara dropdown se company chunni na pade. Ye standard
    Streamlit pattern hai multi-page apps mein "shared selection" ke liye.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.utils.data_loader import get_company_list, get_latest_snapshot_row  # noqa: E402

st.set_page_config(page_title="Company Explorer - ForensIQ", page_icon="🏢", layout="wide")
st.title("🏢 Company Explorer")

companies_df = get_company_list()

if companies_df.empty:
    st.warning("No companies found in the database.")
    st.stop()

# Dropdown se company select karo - session_state mein save hoga taaki
# baaki pages bhi isi selection ko access kar sakein.
selected_name = st.selectbox(
    "Select a company",
    options=companies_df["name"].tolist(),
    index=0,
)

selected_row = companies_df[companies_df["name"] == selected_name].iloc[0]
selected_company_id = int(selected_row["id"])

# Session state mein save - baaki pages "st.session_state.selected_company_id"
# se isi ko read kar sakenge, dropdown dobara nahi dikhana padega.
st.session_state["selected_company_id"] = selected_company_id
st.session_state["selected_company_name"] = selected_name

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Company Info")
    st.write(f"**Name:** {selected_row['name']}")
    st.write(f"**Ticker:** {selected_row['ticker'] or 'N/A'}")
    st.write(f"**CIK:** {selected_row['cik']}")

with col2:
    st.subheader("Latest Snapshot")
    snapshot_row = get_latest_snapshot_row(selected_company_id)

    if snapshot_row is None:
        st.info(
            "This company has no modeling data available (e.g. pre-XBRL "
            "era filings, like Enron, are excluded from the ML dataset)."
        )
    else:
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            st.metric("Fiscal Year", int(snapshot_row["fiscal_year"]))
        with risk_col2:
            st.metric("Red Flags", int(snapshot_row["red_flag_count"]))
        with risk_col3:
            risk_level = snapshot_row.get("risk_level", "N/A")
            st.metric("Risk Level", str(risk_level))

st.divider()
st.info(
    "👈 Select **Fraud Risk & Narrative** from the sidebar to see this "
    "company's ML risk score, SHAP drivers, and the plain-English summary."
)