"""
snapshot_builder.py

Phase 7 ka pehla hissa: "adapter" jo teen alag sources se data leke ek
standard, LLM-ready dictionary banata hai.

Kyun adapter pattern:
    narrative-generator (agla module) ko kabhi ye pata nahi hona chahiye
    ki data DataFrame se aaya ya JSON file se. Agar kal Phase 4/5 ka
    storage format badle, sirf ye file badalni padegi - LLM prompt logic
    bilkul touch nahi karna padega.

Data sources:
    1. build_modeling_dataset() -> company_id, fiscal_year, saare rule
       flags (rule_*), red_flag_count. (Phase 4 rules yahin se aate hain,
       kyunki rules_engine.apply_rules() ka output hi model ke features
       ban jaata hai - alag se stored nahi hai.)
    2. models/shap_explanations.json -> predicted_risk_probability,
       flagged_as_risky, top_risk_drivers, top_risk_reducers (Phase 5 score
       + Phase 6 SHAP dono yahi ek file mein hain).
"""

import json
from pathlib import Path

from forensiq.ml.build_dataset import build_modeling_dataset
from forensiq.db.base import get_session
from forensiq.db.models import Company

MODEL_DIR = Path("models")
SHAP_EXPLANATIONS_PATH = MODEL_DIR / "shap_explanations.json"

# Human-readable naam har rule flag column ke liye - ye isliye alag rakha
# hai kyunki LLM ko raw column name (jaise "rule_leverage_spike") dena
# thoda robotic lagta; readable label dena narrative ko natural banata hai.
RULE_LABELS = {
    "receivables_outpacing_revenue": "Receivables growing faster than revenue",
    "inventory_outpacing_revenue": "Inventory growing faster than revenue",
    "rule_cashflow_below_income": "Operating cash flow below reported net income",
    "rule_margin_deteriorating": "Gross margin declined year-over-year",
    "rule_leverage_spike": "Debt-to-Equity spiked more than 25% YoY",
    "rule_altman_distress": "Altman Z-Score signals financial distress",
    "rule_beneish_manipulation_flag": "Beneish M-Score above manipulation-likely threshold",
    "rule_piotroski_weak": "Piotroski F-Score indicates weak financial health",
}


def _resolve_company_name(company_id) -> str:
    """company_id (Company.id) ko DB se query karke actual naam laata hai.
    Company na milne par company_id hi wapas kar deta hai (fail-safe
    fallback), taaki narrative generation crash na ho."""
    session = get_session()
    company = session.get(Company, company_id)
    return company.name if company else str(company_id)


def _load_shap_explanations() -> list[dict]:
    with open(SHAP_EXPLANATIONS_PATH) as f:
        return json.load(f)


def build_company_snapshot(company_id) -> dict:
    """
    Ek company ke liye "overall" (latest fiscal year) snapshot banata hai,
    jisme rules + ML score + SHAP explanation sab combine ho.

    Overall = us company ka sabse recent fiscal_year, kyunki fraud-risk
    ke liye "abhi company kaisi dikh rahi hai" sabse relevant hai
    (purane saal ka data dilute kar deta agar hum average nikalte).
    """
    # ---- Rules data: build_modeling_dataset() se ----
    df = build_modeling_dataset()
    company_rows = df[df["company_id"] == company_id]

    if company_rows.empty:
        raise ValueError(f"company_id '{company_id}' dataset mein nahi mila.")

    latest_row = company_rows.sort_values("fiscal_year").iloc[-1]
    latest_year = int(latest_row["fiscal_year"])

    triggered_rules = [
        RULE_LABELS[col]
        for col in RULE_LABELS
        if bool(latest_row.get(col, False))
    ]

    # ---- Score + SHAP data: shap_explanations.json se ----
    all_explanations = _load_shap_explanations()
    matching = [
        e for e in all_explanations
        if e["company_id"] == company_id and e["fiscal_year"] == latest_year
    ]

    if not matching:
        raise ValueError(
            f"'{company_id}' / FY{latest_year} ke liye SHAP explanation nahi mili. "
            "compute_shap_explanations.py dobara chalao?"
        )

    explanation = matching[0]

    return {
        "company_name": _resolve_company_name(company_id),
        "fiscal_year": latest_year,
        "risk_score": explanation["predicted_risk_probability"],
        "flagged_as_risky": explanation["flagged_as_risky"],
        "red_flag_count": int(latest_row["red_flag_count"]),
        "triggered_rules": triggered_rules,
        "shap_top_drivers": [
            {"feature": d["feature"], "contribution": d["contribution"]}
            for d in explanation["top_risk_drivers"]
        ],
        "shap_top_reducers": [
            {"feature": d["feature"], "contribution": d["contribution"]}
            for d in explanation["top_risk_reducers"]
        ],
    }


if __name__ == "__main__":
    # Quick manual test - apna ek company_id daal ke chalao
    import sys
    raw_id = sys.argv[1] if len(sys.argv) > 1 else None
    if raw_id is None:
        print("Usage: python snapshot_builder.py <company_id>")
    else:
        # company_id dataset mein integer hai (Company.id, primary key),
        # lekin sys.argv hamesha string deta hai - isliye convert zaroori hai,
        # warna "1" == 1 comparison False aayega aur row kabhi match nahi hoga.
        test_id = int(raw_id)
        snapshot = build_company_snapshot(test_id)
        print(json.dumps(snapshot, indent=2))