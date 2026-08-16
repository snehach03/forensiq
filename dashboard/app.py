"""
app.py

Dashboard ka entry point. Ye "Home" page hai - project ka overview
dikhata hai. Actual company-specific views alag pages/ files mein
hain (Streamlit automatically inko sidebar mein list karta hai).

Chalane ka tareeka (project root se):
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# dashboard/ folder se chalne par bhi "forensiq" package import ho sake,
# isliye src/ ko path mein add kar rahe hain. Project root bhi add kar rahe
# hain kyunki Streamlit sirf app.py ke apne folder ko path mein daalta hai -
# "dashboard" package ko khud import karne ke liye uska parent (root) bhi
# chahiye hota hai.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.utils.data_loader import get_company_list, load_narratives  # noqa: E402

st.set_page_config(
    page_title="ForensIQ - Financial Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 ForensIQ")
st.subheader("AI-Powered Financial Forensics Platform")

st.markdown("""
ForensIQ analyzes SEC filings to detect accounting anomalies and potential
fraud signals using three layered techniques:

1. **Deterministic Rules Engine** — fixed, explainable red-flag checks
2. **Machine Learning Layer** — Logistic Regression risk scoring
3. **LLM Explanation Layer** — plain-English narrative reports (Groq)

Use the sidebar to navigate between views.
""")

st.divider()

# Quick summary stats - pehli nazar mein dashboard "alive" lage,
# khaali page na dikhe.
companies_df = get_company_list()
narratives_data = load_narratives()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Companies Tracked", len(companies_df))
with col2:
    st.metric("Narratives Generated", narratives_data.get("successful", 0))
with col3:
    st.metric("Failed Generations", narratives_data.get("failed", 0))

st.divider()
st.caption(
    "Proof-of-concept project — model metrics reflect a small, imbalanced "
    "dataset (72 company-years, 5 fraud-labeled) and are not production-grade."
)