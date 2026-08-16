"""
diagnose_growth_missingness.py
Confirms whether growth-metric NaNs come from (a) genuine first-year gaps
or (b) underlying concept missingness compounding across two years.
Run: python -m forensiq.ml.diagnose_growth_missingness
"""

from forensiq.ml.build_dataset import build_modeling_dataset

df = build_modeling_dataset()

print("Per-company fiscal years present + revenue_growth NaN pattern:\n")
for cid, group in df.groupby("company_id"):
    group_sorted = group.sort_values("fiscal_year")
    years = group_sorted["fiscal_year"].tolist()
    revenue_missing = group_sorted["Revenue"].isna().tolist()
    growth_missing = group_sorted["revenue_growth"].isna().tolist()
    print(f"company_id={cid}")
    print(f"  fiscal_years         : {years}")
    print(f"  Revenue missing?     : {revenue_missing}")
    print(f"  revenue_growth NaN?  : {growth_missing}")