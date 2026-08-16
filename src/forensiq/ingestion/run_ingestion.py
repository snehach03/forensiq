"""
Orchestration script: runs the full ingestion pipeline (company info +
filings + XBRL facts) for every company in our project scope.

Why a single script instead of running each service manually per company:
Running 8 companies through 3 services by hand in a Python shell is
repetitive and error-prone (easy to typo a CIK or skip a step). This
script is the single source of truth for "which companies are in our
dataset" and guarantees every company goes through the same pipeline.
"""

import time

from forensiq.db.base import get_session
from forensiq.db.models import Company
from forensiq.ingestion.company_service import register_company
from forensiq.ingestion.filing_service import sync_filings
from forensiq.ingestion.xbrl_service import sync_financial_facts

# Our fixed dataset: known fraud/scrutiny cases + healthy comparables.
COMPANIES = {
    "Apple": "320193",
    "Microsoft": "789019",
    "Costco": "909832",
    "General Electric": "40545",
    "Under Armour": "1336917",
    "Enron": "1024401",
    "Kraft Heinz": "1637459",
    "Valeant/Bausch Health": "885590",
}


def run_full_ingestion():
    for name, cik in COMPANIES.items():
        print(f"\n=== Processing {name} (CIK: {cik}) ===")

        try:
            company = register_company(cik)
            print(f"  Company registered: {company.name} (id={company.id})")

            filings = sync_filings(company)
            print(f"  Filings synced: {len(filings)}")

            fact_count = sync_financial_facts(company)
            print(f"  New financial facts saved: {fact_count}")

        except Exception as e:
            print(f"  ERROR processing {name}: {e}")

        time.sleep(1)  # small pause between companies, extra courtesy to SEC's servers

    print("\n=== Ingestion complete for all companies ===")


if __name__ == "__main__":
    run_full_ingestion()