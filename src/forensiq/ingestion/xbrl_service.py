"""
Service layer for syncing XBRL structured financial facts from
SEC EDGAR into our database.

Why we whitelist specific concepts instead of saving everything:
SEC's companyfacts API returns 100+ XBRL concepts per company, most
of which we'll never use (obscure disclosures, footnote-level detail).
Saving only the concepts our feature engineering (Phase 3) actually
needs keeps the database lean and avoids parsing noise we don't need.
"""

from datetime import datetime

from forensiq.db.base import get_session
from forensiq.db.models import Company, FinancialFact
from forensiq.ingestion.edgar_client import EdgarClient

# Core concepts needed for Beneish M-Score, Altman Z-Score,
# Piotroski F-Score, and standard financial ratios (Phase 3).
RELEVANT_CONCEPTS = {
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "AccountsReceivableNetCurrent",
    "InventoryNet",
    "PropertyPlantAndEquipmentNet",
    "NetCashProvidedByUsedInOperatingActivities",
    "RetainedEarningsAccumulatedDeficit",
}


def sync_financial_facts(company: Company) -> int:
    """
    Fetches XBRL company facts from SEC EDGAR, extracts our whitelisted
    concepts, and saves each (concept, period) data point to the
    financial_facts table. Returns the count of NEW facts saved.
    """
    client = EdgarClient()
    data = client.get_company_facts(company.cik)

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    session = get_session()
    saved_count = 0

    try:
        for concept_name in RELEVANT_CONCEPTS:
            concept_data = us_gaap.get(concept_name)
            if not concept_data:
                continue  # this company may not report this concept

            units = concept_data.get("units", {})
            usd_values = units.get("USD", [])

            for entry in usd_values:
                fiscal_period = entry.get("fp")
                form_type = entry.get("form")
                end_date_str = entry.get("end")
                start_date_str = entry.get("start")
                value = entry.get("val")

                if not all([fiscal_period, form_type, end_date_str, value is not None]):
                    continue  # skip incomplete entries

                if form_type not in ("10-K", "10-Q"):
                    continue

                period_end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                period_start = (
                    datetime.strptime(start_date_str, "%Y-%m-%d").date()
                    if start_date_str
                    else None
                )

                fiscal_year = period_end.year

                existing = (
                    session.query(FinancialFact)
                    .filter_by(
                        company_id=company.id,
                        concept=concept_name,
                        fiscal_year=fiscal_year,
                        fiscal_period=fiscal_period,
                        form_type=form_type,
                    )
                    .first()
                )
                if existing:
                    continue

                fact = FinancialFact(
                    company_id=company.id,
                    filing_id=None,
                    concept=concept_name,
                    unit="USD",
                    value=value,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    period_start=period_start,
                    period_end=period_end,
                    form_type=form_type,
                )
                session.add(fact)
                saved_count += 1

        session.commit()
        return saved_count

    finally:
        session.close()