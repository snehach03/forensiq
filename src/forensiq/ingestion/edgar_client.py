"""
Thin HTTP client for SEC EDGAR APIs.

Why this file exists on its own:
All SEC-specific HTTP concerns (required headers, rate limiting, retries)
are isolated here so that the rest of the ingestion code (company_service,
filing_service, xbrl_service) doesn't need to know HOW to talk to SEC —
it just calls methods like get_company_submissions(cik).

SEC's rules we must follow:
  - Every request needs a descriptive User-Agent header (name + email),
    or SEC blocks the request.
  - Rate limit: max ~10 requests/second. We stay well under that.
"""

import time
import requests

from forensiq.config import settings


class EdgarClient:
    BASE_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    BASE_XBRL_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.sec_user_agent})

    def _get(self, url: str) -> dict:
        """Makes a GET request with basic rate-limiting and error handling."""
        response = self.session.get(url)
        time.sleep(0.15)  # stay comfortably under SEC's 10 req/sec limit

        if response.status_code == 404:
            raise ValueError(f"Not found on SEC EDGAR: {url}")
        response.raise_for_status()
        return response.json()

    def get_company_submissions(self, cik: str) -> dict:
        """
        Fetches company metadata + list of all filings for a given CIK.
        CIK must be zero-padded to 10 digits, e.g. '0000320193' for Apple.
        """
        padded_cik = cik.zfill(10)
        url = f"{self.BASE_SUBMISSIONS_URL}/CIK{padded_cik}.json"
        return self._get(url)

    def get_company_facts(self, cik: str) -> dict:
        """
        Fetches all XBRL structured financial facts for a given CIK
        (Revenue, NetIncomeLoss, Assets, etc. across all reported periods).
        """
        padded_cik = cik.zfill(10)
        url = f"{self.BASE_XBRL_FACTS_URL}/CIK{padded_cik}.json"
        return self._get(url)