"""
narrative_prompt.py

Prompt ko logic se alag rakha hai (separate file) taaki prompt-tuning
(jo LLM work mein normal hai - bahut baar edit hota hai) baaki code
ko touch na kare.

Design principle (already discuss kiya): LLM sirf "translator" hai.
System prompt mein isi constraint ko explicit aur strict banaya gaya hai.
"""

SYSTEM_PROMPT = """You are a financial forensics analyst writing a plain-English \
risk summary for a fraud-detection dashboard.

STRICT RULES - follow these without exception:
1. Use ONLY the numbers, percentages, feature names, and facts given to you \
in the data below. Never introduce a number, statistic, or fact that is not \
explicitly present in the input.
2. Do not speculate about intent, legality, or make accusations of actual \
fraud. Describe patterns and risk signals only ("this pattern is consistent \
with X" not "this company committed X").
3. If the input data is thin or inconclusive, say so explicitly rather than \
filling gaps with assumptions.
4. Write 2-3 short paragraphs, professional tone, as if for a financial \
analyst reviewing many companies quickly.
5. Reference the rule violations and SHAP drivers naturally in prose - do \
not just list them mechanically."""


def build_user_prompt(snapshot: dict) -> str:
    """Snapshot dictionary ko ek clean, structured text block mein
    convert karta hai jo LLM ko diya jaayega. JSON dump isliye nahi
    kiya raw - structured labeled text LLM ke liye parse karna aasan
    hota hai aur output quality better aati hai."""

    rules_text = (
        "\n".join(f"  - {r}" for r in snapshot["triggered_rules"])
        if snapshot["triggered_rules"]
        else "  - None triggered"
    )

    drivers_text = "\n".join(
        f"  - {d['feature']} (contribution: {d['contribution']:+.4f}, increases risk)"
        for d in snapshot["shap_top_drivers"]
    )

    reducers_text = "\n".join(
        f"  - {d['feature']} (contribution: {d['contribution']:+.4f}, decreases risk)"
        for d in snapshot["shap_top_reducers"]
    )

    return f"""Company: {snapshot['company_name']}
Fiscal Year: {snapshot['fiscal_year']}

ML MODEL OUTPUT:
  Fraud risk probability: {snapshot['risk_score']:.3f}
  Flagged as risky (threshold 0.40): {snapshot['flagged_as_risky']}

DETERMINISTIC RULE VIOLATIONS ({snapshot['red_flag_count']} triggered):
{rules_text}

TOP SHAP RISK DRIVERS (features pushing risk score up):
{drivers_text}

TOP SHAP RISK REDUCERS (features pushing risk score down):
{reducers_text}

Write the risk summary now, following the system instructions."""