"""
batch_generate_narratives.py

Phase 7 ka final step: saari companies ke liye narratives ek baar
generate karke models/narratives.json mein save karta hai.

Kyun batch + persist (live generation nahi):
    Phase 8 (Streamlit dashboard) ko har page-load pe Groq call nahi
    karni chahiye - slow hoga, unnecessary cost/rate-limit risk hoga,
    aur agar Groq down ho toh dashboard bhi crash ho jaayega. Isliye
    narratives ek baar generate karke disk pe save karte hain; dashboard
    sirf is JSON ko read karega.

Kyun per-company error handling:
    Ek company ka data missing/incomplete ho (jaisa humne dekha - company_id
    6 dataset mein nahi hai) toh poora batch fail nahi hona chahiye. Har
    company independently try hota hai; success ya failure dono record
    hote hain taaki baad mein pata chale kya missing hai.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from forensiq.ml.build_dataset import build_modeling_dataset
from forensiq.llm.generate_narrative import generate_narrative

MODEL_DIR = Path("models")
OUTPUT_PATH = MODEL_DIR / "narratives.json"

# Groq free-tier rate limits ka dhyan rakhne ke liye chhota gap - abhi
# 7 companies ke liye zaroori nahi, but future-proofing ke liye achhi habit.
DELAY_BETWEEN_CALLS_SECONDS = 1


def get_all_company_ids() -> list[int]:
    """Dataset mein jitni bhi companies hain, unke IDs nikalta hai -
    hardcoded list rakhne ki jagah, taaki naya company add hone par
    ye script khud-ba-khud usse bhi include kar le."""
    df = build_modeling_dataset()
    return sorted(df["company_id"].unique().tolist())


def run_batch() -> dict:
    company_ids = get_all_company_ids()
    print(f"Found {len(company_ids)} companies: {company_ids}")

    results = []
    failures = []

    for i, company_id in enumerate(company_ids, start=1):
        print(f"[{i}/{len(company_ids)}] Generating narrative for company_id={company_id}...")
        try:
            narrative_result = generate_narrative(company_id)
            results.append({"company_id": company_id, **narrative_result})

            if narrative_result.get("validation_warning"):
                print(f"  ⚠️  Validation warning: {narrative_result['validation_warning']}")
            else:
                print(f"  ✅ Success")

        except Exception as e:
            # Ek company fail ho toh poora batch mat roko - error record
            # karke aage badho.
            print(f"  ❌ Failed: {e}")
            failures.append({"company_id": company_id, "error": str(e)})

        if i < len(company_ids):
            time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_companies": len(company_ids),
        "successful": len(results),
        "failed": len(failures),
        "narratives": results,
        "failures": failures,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"✅ {len(results)}/{len(company_ids)} narratives generated successfully")
    if failures:
        print(f"❌ {len(failures)} failed: {[f['company_id'] for f in failures]}")
    print(f"Saved to: {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    run_batch()