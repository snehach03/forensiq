"""
SQLAlchemy ORM models — these define our database tables as Python classes.

Design note: financial_facts is stored in LONG format (one row per
concept per period) rather than WIDE format (one row per filing with
50+ columns). Reason: not every company reports every XBRL concept,
so a wide table would be sparse and painful to alter whenever a new
concept shows up. The feature engineering layer (Phase 3) will pivot
this into wide format per company-period when needed.
"""

from datetime import date, datetime
from sqlalchemy import (
    String, Integer, BigInteger, Date, DateTime, Numeric,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(10))
    sic_code: Mapped[str | None] = mapped_column(String(10))
    sic_description: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    filings: Mapped[list["Filing"]] = relationship(back_populates="company")


class Filing(Base):
    __tablename__ = "filings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)
    accession_number: Mapped[str] = mapped_column(String(25), unique=True, nullable=False)
    form_type: Mapped[str] = mapped_column(String(10), nullable=False)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False)
    period_of_report: Mapped[date | None] = mapped_column(Date)
    primary_doc_url: Mapped[str | None] = mapped_column(String(500))
    fiscal_year: Mapped[int | None] = mapped_column(Integer)
    fiscal_period: Mapped[str | None] = mapped_column(String(4))
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    company: Mapped["Company"] = relationship(back_populates="filings")

    __table_args__ = (
        Index("idx_company_form", "company_id", "form_type"),
    )


class FinancialFact(Base):
    __tablename__ = "financial_facts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)
    filing_id: Mapped[int | None] = mapped_column(ForeignKey("filings.id"))
    concept: Mapped[str] = mapped_column(String(100), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    value: Mapped[float] = mapped_column(Numeric(20, 2), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_period: Mapped[str] = mapped_column(String(4), nullable=False)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    form_type: Mapped[str] = mapped_column(String(10), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "company_id", "concept", "fiscal_year", "fiscal_period", "form_type",
            name="uq_fact"
        ),
        Index("idx_company_concept_period", "company_id", "concept", "fiscal_year", "fiscal_period"),
    )