"""
2_Fraud_Risk_and_Narrative.py

Phase 8, view 2: Fraud Risk & Narrative.

Design decision - session_state se company selection reuse karna:
    Page 1 (Company Explorer) mein user ne jo company choose ki thi,
    wahi selection yahan bhi use hoga - dobara dropdown nahi dikhega.
    Agar session_state khaali hai (user seedha is page pe aaya, Company
    Explorer visit kiye bina), fallback dropdown dikhayenge.

Design decision - Fraud Risk Timeline ke liye get_company_risk_history()
    (Phase 8 addition):
    Single-year snapshot ek number dikhata hai, lekin fraud aksar
    "gradual buildup" hota hai (jaise Kraft Heinz 2016-2018). Multi-year
    line chart ye trajectory dikhata hai jo ek static metric nahi dikha
    sakta. get_company_risk_history() already SHAP explanations se ye
    data nikaal deta hai - dobara model call karne ki zaroorat nahi.

Design decision - SHAP ko bar chart mein convert karna (Phase 8 addition):
    Pehle sirf text bullets the (feature : contribution). Bar chart mein
    bar ki length hi impact ka size dikhati hai, aur color (red=risk
    driver, green=risk reducer) instant visual signal deta hai - text
    list se zyada interpretable hai non-technical viewer ke liye bhi.
"""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.utils.data_loader import (  # noqa: E402
    get_company_list,
    get_company_risk_history,
    get_latest_snapshot_row,
    get_narrative_for_company,
    load_shap_explanations,
)

st.set_page_config(page_title="Fraud Risk & Narrative - ForensIQ", page_icon="⚠️", layout="wide")
st.title("⚠️ Fraud Risk & Narrative")

companies_df = get_company_list()

# Session state se company selection lo; agar khaali hai, fallback dropdown do.
if "selected_company_id" not in st.session_state:
    st.info("No company selected yet - pick one below (or visit Company Explorer first).")
    selected_name = st.selectbox("Select a company", options=companies_df["name"].tolist())
    selected_company_id = int(companies_df[companies_df["name"] == selected_name].iloc[0]["id"])
else:
    selected_company_id = st.session_state["selected_company_id"]
    selected_name = st.session_state["selected_company_name"]
    st.caption(f"Showing: **{selected_name}** (change selection in Company Explorer)")

st.divider()

snapshot_row = get_latest_snapshot_row(selected_company_id)

if snapshot_row is None:
    st.warning(
        "This company has no modeling data available (e.g. pre-XBRL era "
        "filings are excluded from the ML dataset)."
    )
    st.stop()

latest_year = int(snapshot_row["fiscal_year"])
narrative_entry = get_narrative_for_company(selected_company_id)

# ---- SHAP explanation ke andar hi risk_score bhi hai (predicted_risk_probability) ----
all_shap = load_shap_explanations()
shap_entry = next(
    (e for e in all_shap if e["company_id"] == selected_company_id and e["fiscal_year"] == latest_year),
    None,
)

# ---- Section 1: ML risk score ----
st.subheader("🤖 ML Model Output")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Fiscal Year", latest_year)
with col2:
    st.metric("Red Flag Count", int(snapshot_row["red_flag_count"]))
with col3:
    st.metric("Risk Level", str(snapshot_row.get("risk_level", "N/A")))
with col4:
    if shap_entry:
        prob_pct = f"{shap_entry['predicted_risk_probability'] * 100:.1f}%"
        st.metric("Risk Probability", prob_pct)
    else:
        st.metric("Risk Probability", "N/A")

st.divider()

# ---- Section 1b: Fraud Risk Timeline (NEW - Phase 8) ----
st.subheader("📅 Fraud Risk Timeline")

risk_history = get_company_risk_history(selected_company_id)

if len(risk_history) < 2:
    st.info(
        "Not enough multi-year data to plot a risk timeline for this "
        "company (need at least 2 fiscal years with SHAP explanations)."
    )
else:
    timeline_fig = px.line(
        x=[e["fiscal_year"] for e in risk_history],
        y=[e["predicted_risk_probability"] for e in risk_history],
        markers=True,
        labels={"x": "Fiscal Year", "y": "Predicted Risk Probability"},
        title=f"Risk Probability Over Time — {selected_name}",
    )
    # 0.40 decision threshold ki reference line - Phase 5 mein jo threshold
    # chosen hua tha, wahi yahan bhi dikhana consistent hai (jaisa Model
    # Evaluation page mein bhi isi threshold ka use hota hai).
    timeline_fig.add_hline(
        y=0.40,
        line_dash="dash",
        line_color="red",
        annotation_text="Decision threshold (0.40)",
        annotation_position="bottom right",
    )
    timeline_fig.update_yaxes(range=[0, 1])
    timeline_fig.update_layout(hovermode="x unified")
    st.plotly_chart(timeline_fig, use_container_width=True)

st.divider()

# ---- Section 2: SHAP drivers/reducers (Phase 6 output) ----
st.subheader("🔍 SHAP Feature Explanations")

if shap_entry is None:
    st.info("No SHAP explanation available for this company-year.")
else:
    # Drivers (risk badhane wale, positive contribution) aur reducers (risk
    # ghatane wale, negative contribution) - dono ko ek hi horizontal bar
    # chart mein combine kar rahe hain, sorted by contribution magnitude,
    # taaki sabse impactful feature sabse upar/neeche dikhe.
    drivers = shap_entry["top_risk_drivers"]
    reducers = shap_entry["top_risk_reducers"]

    features = [d["feature"] for d in drivers] + [r["feature"] for r in reducers]
    contributions = [d["contribution"] for d in drivers] + [r["contribution"] for r in reducers]
    colors = ["#d62728" if c >= 0 else "#2ca02c" for c in contributions]

    # Sort magnitude ke hisaab se (sabse chhota top pe, taaki bar chart mein
    # sabse bada impact sabse upar dikhe - Plotly horizontal bars neeche se
    # upar draw karta hai).
    sorted_data = sorted(zip(features, contributions, colors), key=lambda x: x[1])
    features_sorted, contributions_sorted, colors_sorted = zip(*sorted_data)

    shap_fig = go.Figure(
        go.Bar(
            x=list(contributions_sorted),
            y=list(features_sorted),
            orientation="h",
            marker_color=list(colors_sorted),
        )
    )
    shap_fig.update_layout(
        xaxis_title="SHAP Contribution (+ = increases risk, - = decreases risk)",
        yaxis_title="",
        title="Top Risk Drivers & Reducers",
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    st.caption("🔴 Red = pushes risk score up   🟢 Green = pushes risk score down")

st.divider()

# ---- Section 3: LLM Narrative (Phase 7 output) ----
st.subheader("📝 Plain-English Risk Summary")

if narrative_entry is None:
    st.warning(
        "No narrative available for this company. It may have failed during "
        "batch generation - check `models/narratives.json` for details."
    )
else:
    if narrative_entry.get("validation_warning"):
        st.warning(narrative_entry["validation_warning"])
    st.write(narrative_entry["narrative"])

st.divider()
st.caption(
    "This summary is LLM-generated from pre-computed rule violations, "
    "ML scores, and SHAP values only - the LLM does not perform its own "
    "calculations (see Phase 7 design notes)."
)