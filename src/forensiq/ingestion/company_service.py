"""
Service layer for syncing company metadata from SEC EDGAR into our database.

Why a separate service layer (not just calling EdgarClient directly
from wherever we need data):
This keeps "how to fetch from SEC" (edgar_client.py) separate from
"how to save it to our database" (this file). If tomorrow we needed to
fetch from a different source, only this layer would change — the
database logic and models stay untouched.
"""

from forensiq.db.base import get_session
from forensiq.db.models import Company
from forensiq.ingestion.edgar_client import EdgarClient


def register_company(cik: str) -> Company:
    """
    Fetches a company's metadata from SEC EDGAR and saves it to the
    database. If the company already exists (same CIK), updates it
    instead of creating a duplicate (idempotent).
    """
    client = EdgarClient()
    data = client.get_company_submissions(cik)

    padded_cik = cik.zfill(10)
    session = get_session()

    try:
        existing = session.query(Company).filter_by(cik=padded_cik).first()

        if existing:
            existing.name = data.get("name")
            existing.ticker = (data.get("tickers") or [None])[0]
            existing.sic_code = data.get("sic")
            existing.sic_description = data.get("sicDescription")
            company = existing
        else:
            company = Company(
                cik=padded_cik,
                name=data.get("name"),
                ticker=(data.get("tickers") or [None])[0],
                sic_code=data.get("sic"),
                sic_description=data.get("sicDescription"),
            )
            session.add(company)

        session.commit()
        session.refresh(company)
        return company

    finally:
        session.close()