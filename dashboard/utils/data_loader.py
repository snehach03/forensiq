"""
data_loader.py

Saare Streamlit pages isi file se data lenge - koi bhi page apna
khud ka DB query ya file-read logic nahi likhega. Ye "single source
of truth" pattern hai: agar kal data-source badle (DB schema, file
location), sirf ye ek file update karni padegi.

@st.cache_data ka istemal:
    Streamlit har user-interaction (jaise dropdown change) pe poori
    script re-run karta hai. Bina caching ke, har interaction pe DB
    query / file-read dobara chalta - slow aur wasteful. cache_data
    result ko memory mein rakhta hai jab tak input params same hain.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from forensiq.ml.build_dataset import build_modeling_dataset
from forensiq.db.base import get_session
from forensiq.db.models import Company

MODELS_DIR = Path("models")
NARRATIVES_PATH = MODELS_DIR / "narratives.json"
SHAP_EXPLANATIONS_PATH = MODELS_DIR / "shap_explanations.json"
SHAP_GLOBAL_IMPORTANCE_PATH = MODELS_DIR / "shap_global_importance.json"
LOOCV_PREDICTIONS_PATH = MODELS_DIR / "loocv_predictions.json"


@st.cache_data(ttl=3600)
def load_modeling_dataset() -> pd.DataFrame:
    """Poora modeling dataset (rules + features + labels) - Phase 5/6
    dono isi dataset pe based hain."""
    return build_modeling_dataset()


@st.cache_data(ttl=3600)
def get_company_list() -> pd.DataFrame:
    """Saari companies (id, name, ticker) - dropdown/selector ke liye.
    DB se aata hai kyunki naam sirf wahin available hai (dataset mein
    sirf company_id hota hai, naam nahi)."""
    session = get_session()
    companies = session.query(Company).order_by(Company.name).all()
    return pd.DataFrame([
        {"id": c.id, "name": c.name, "ticker": c.ticker, "cik": c.cik}
        for c in companies
    ])


@st.cache_data(ttl=3600)
def load_narratives() -> dict:
    """Phase 7 ke LLM narratives - poora batch output (metadata + list)."""
    if not NARRATIVES_PATH.exists():
        return {"narratives": [], "failures": []}
    with open(NARRATIVES_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def get_narrative_for_company(company_id: int) -> dict | None:
    """Ek specific company ka narrative dhoondhta hai. None agar nahi mila
    (jaise agar batch generation ke time wo company fail hui thi)."""
    data = load_narratives()
    for n in data.get("narratives", []):
        if n["company_id"] == company_id:
            return n
    return None


@st.cache_data(ttl=3600)
def load_shap_explanations() -> list[dict]:
    """Phase 6 ka per-company-year SHAP output - drivers/reducers."""
    if not SHAP_EXPLANATIONS_PATH.exists():
        return []
    with open(SHAP_EXPLANATIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_shap_global_importance() -> dict:
    """Phase 6 ka overall feature-importance (poore dataset ka average)."""
    if not SHAP_GLOBAL_IMPORTANCE_PATH.exists():
        return {}
    with open(SHAP_GLOBAL_IMPORTANCE_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_loocv_predictions() -> list[dict]:
    """Phase 8 ke Model Evaluation view ke liye - har row ka
    (true_label, predicted_probability) pair, LOOCV se."""
    if not LOOCV_PREDICTIONS_PATH.exists():
        return []
    with open(LOOCV_PREDICTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_company_risk_history(company_id: int) -> list[dict]:
    """Ek company ke saare available fiscal-years ka risk probability -
    Fraud Risk Timeline ke liye. shap_explanations.json mein har
    company-year ka predicted_risk_probability already hai (Phase 6
    output), isliye alag se dobara model call karne ki zaroorat nahi."""
    all_explanations = load_shap_explanations()
    company_history = [e for e in all_explanations if e["company_id"] == company_id]
    return sorted(company_history, key=lambda e: e["fiscal_year"])


def get_latest_snapshot_row(company_id: int) -> pd.Series | None:
    """Ek company ka sabse recent fiscal-year row - modeling dataset se.
    (Note: caching yahan nahi lagi kyunki ye poore cached DataFrame se
    hi filter karta hai - already fast hai, extra cache layer ki zaroorat
    nahi.)"""
    df = load_modeling_dataset()
    company_rows = df[df["company_id"] == company_id]
    if company_rows.empty:
        return None
    return company_rows.sort_values("fiscal_year").iloc[-1]