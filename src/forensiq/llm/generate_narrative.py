"""
generate_narrative.py

Phase 7 ka final piece: snapshot -> Groq API call -> validated narrative.

Model choice: openai/gpt-oss-20b
    Task reasoning-heavy nahi hai (translate/summarize hi karna hai,
    calculate kuch nahi karna), isliye lightweight/fast model kaafi hai -
    bade model ka cost/latency overhead yahan justify nahi hota.
"""

import os
import re
from dotenv import load_dotenv
from groq import Groq

from forensiq.llm.snapshot_builder import build_company_snapshot
from forensiq.llm.narrative_prompt import SYSTEM_PROMPT, build_user_prompt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"


def _extract_numbers(text: str) -> list[float]:
    """Text mein se saare numeric tokens nikal ke float list return karta
    hai (string ki jagah float - taaki numeric comparison kar sakein,
    formatting variations se independent)."""
    raw = re.findall(r"-?\d+\.?\d*", text)
    return [float(n) for n in raw if n not in ("", "-", ".")]


def _allowed_base_values(snapshot: dict) -> set[float]:
    """Snapshot ke andar jitne bhi numbers legitimately hain, unko float
    set mein collect karta hai - "base" values, formatting-independent."""
    allowed = {
        float(snapshot["fiscal_year"]),
        snapshot["risk_score"],
        float(snapshot["red_flag_count"]),
        0.40,  # decision threshold
    }

    for d in snapshot["shap_top_drivers"] + snapshot["shap_top_reducers"]:
        allowed.add(d["contribution"])
        allowed.add(abs(d["contribution"]))

    # Rule-violation labels (jaise "spiked more than 25% YoY") mein khud
    # humare diye gaye numbers chhupe ho sakte hain - LLM inhe wapas quote
    # kar sakta hai, ye hallucination nahi hai.
    for rule_text in snapshot["triggered_rules"]:
        allowed.update(float(n) for n in re.findall(r"-?\d+\.?\d*", rule_text))

    return allowed


def _is_close_to_any(value: float, allowed: set[float], tolerance: float = 0.05) -> bool:
    """LLM formatting ke liye flexible: value ko base value se, ya uske
    x100/÷100 (percentage conversion) se compare karta hai. Isse "0.40"
    vs "40%" ya "0.203" vs "20.3%" jaise cases dono legitimate maane
    jaate hain, bina har combination manually predict kiye."""
    for base in allowed:
        for candidate in (base, base * 100, base / 100 if base else 0):
            if abs(value - candidate) <= tolerance:
                return True
    return False


def _flag_unrecognized_numbers(text: str, snapshot: dict) -> list[float]:
    """Hallucination guard: generated text mein jo bhi numbers hain,
    check karta hai ki wo snapshot se (formatting-independent) match
    karte hain ya LLM ne khud bana diye. Chote counting numbers (0-9,
    jaise 'top 3 features') ko ignore karte hain."""
    allowed = _allowed_base_values(snapshot)
    found = _extract_numbers(text)

    suspicious = []
    for num in found:
        if abs(num) <= 9:
            continue  # chote counting numbers, false-positive avoid karo
        if not _is_close_to_any(num, allowed):
            suspicious.append(num)

    return suspicious


def generate_narrative(company_id) -> dict:
    """Poora pipeline: snapshot build -> Groq call -> validate -> return."""
    snapshot = build_company_snapshot(company_id)
    user_prompt = build_user_prompt(snapshot)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # kam temperature = zyada deterministic/factual,
                          # creative writing yahan goal nahi hai
    )

    narrative_text = completion.choices[0].message.content
    suspicious_numbers = _flag_unrecognized_numbers(narrative_text, snapshot)

    return {
        "company_name": snapshot["company_name"],
        "fiscal_year": snapshot["fiscal_year"],
        "narrative": narrative_text,
        "validation_warning": (
            f"⚠️ Unrecognized numbers in output: {suspicious_numbers} - review before trusting."
            if suspicious_numbers else None
        ),
    }


if __name__ == "__main__":
    import sys
    import json

    raw_id = sys.argv[1] if len(sys.argv) > 1 else None
    if raw_id is None:
        print("Usage: python generate_narrative.py <company_id>")
    else:
        # snapshot_builder.py mein jo bug fix kiya tha (sys.argv string
        # deta hai, company_id integer hai) - wahi yahan bhi zaroori hai.
        test_id = int(raw_id)
        result = generate_narrative(test_id)
        print(json.dumps(result, indent=2))