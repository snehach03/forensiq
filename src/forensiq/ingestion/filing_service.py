"""
Service layer for syncing a company's 10-K/10-Q filing list from
SEC EDGAR into our database.

Why this is separate from company_service.py:
Company metadata and filing history are different concerns with
different update frequencies (a company's name rarely changes, but
new filings show up every quarter). Keeping them in separate services
means each one stays focused and easy to test independently.
"""

from datetime import datetime

from forensiq.db.base import get_session
from forensiq.db.models import Company, Filing
from forensiq.ingestion.edgar_client import EdgarClient

RELEVANT_FORM_TYPES = {"10-K", "10-Q"}


def sync_filings(company: Company) -> list[Filing]:
    """
    Fetches the full filing history for a company from SEC EDGAR,
    filters to 10-K/10-Q only, and saves new ones to the database.
    Already-saved filings (matched by accession_number) are skipped.
    """
    client = EdgarClient()
    data = client.get_company_submissions(company.cik)

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])

    session = get_session()
    saved_filings = []

    try:
        for i in range(len(forms)):
            form_type = forms[i]
            if form_type not in RELEVANT_FORM_TYPES:
                continue

            accession_number = accession_numbers[i]

            existing = (
                session.query(Filing)
                .filter_by(accession_number=accession_number)
                .first()
            )
            if existing:
                saved_filings.append(existing)
                continue

            filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d").date()
            period_of_report = (
                datetime.strptime(report_dates[i], "%Y-%m-%d").date()
                if report_dates[i]
                else None
            )

            accession_no_dashes = accession_number.replace("-", "")
            primary_doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(company.cik)}/{accession_no_dashes}/{primary_docs[i]}"
            )

            fiscal_year = period_of_report.year if period_of_report else filing_date.year
            fiscal_period = "FY" if form_type == "10-K" else _infer_quarter(period_of_report)

            filing = Filing(
                company_id=company.id,
                accession_number=accession_number,
                form_type=form_type,
                filing_date=filing_date,
                period_of_report=period_of_report,
                primary_doc_url=primary_doc_url,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
            )
            session.add(filing)
            saved_filings.append(filing)

        session.commit()
        for f in saved_filings:
            session.refresh(f)
        return saved_filings

    finally:
        session.close()


def _infer_quarter(period_of_report) -> str:
    """Rough quarter inference from the report's period-end month."""
    if period_of_report is None:
        return "Q1"
    month = period_of_report.month
    if month <= 3:
        return "Q1"
    elif month <= 6:
        return "Q2"
    elif month <= 9:
        return "Q3"
    return "Q4"