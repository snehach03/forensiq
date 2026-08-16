"""
migrate_to_aiven.py

One-time script: copies all data from the LOCAL MySQL database to the
Aiven-hosted MySQL database, so the Render-deployed dashboard (which
can't reach your laptop) has its own copy of the data to read from.

Why table order matters:
    filings.company_id and financial_facts.company_id/filing_id are
    foreign keys - if we insert filings before companies exist on the
    Aiven side, MySQL will reject the insert. So we copy in dependency
    order: companies -> filings -> financial_facts.

Why we create tables via Base.metadata.create_all() instead of copying
    schema manually:
    We already have the exact schema defined once, in models.py (the
    single source of truth) - reusing it here means the Aiven tables
    are guaranteed identical in structure to the local ones, no risk
    of a typo'd column type causing subtle bugs later.

Why SSL is handled explicitly for the Aiven connection (but not local):
    Aiven requires SSL for all connections (visible as "SSL mode:
    REQUIRED" in their console) - local MySQL typically doesn't. This
    is passed via connect_args, not the connection string itself,
    because pymysql expects SSL options as a dict, not URL params.

Run this ONCE from the project root (venv activated):
    python migrate_to_aiven.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from forensiq.db.models import Base

load_dotenv()

# ---- Source: your local MySQL (existing DB_* variables in .env) ----
LOCAL_DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
source_engine = create_engine(LOCAL_DB_URL)

# ---- Target: Aiven MySQL (new AIVEN_* variables in .env) ----
AIVEN_DB_URL = (
    f"mysql+pymysql://{os.getenv('AIVEN_DB_USER')}:{os.getenv('AIVEN_DB_PASSWORD')}"
    f"@{os.getenv('AIVEN_DB_HOST')}:{os.getenv('AIVEN_DB_PORT')}/{os.getenv('AIVEN_DB_NAME')}"
)
target_engine = create_engine(
    AIVEN_DB_URL,
    connect_args={"ssl": {"ca": "ca.pem"}},
)

# Dependency order - parents before children, so foreign keys resolve.
TABLES_IN_ORDER = ["companies", "filings", "financial_facts"]


def migrate():
    print("Creating tables on Aiven (schema from models.py)...")
    Base.metadata.create_all(bind=target_engine)

    for table_name in TABLES_IN_ORDER:
        print(f"\nCopying table: {table_name}")
        df = pd.read_sql(f"SELECT * FROM {table_name}", source_engine)
        print(f"  Read {len(df)} rows from local DB")

        if df.empty:
            print("  Nothing to copy, skipping.")
            continue

        df.to_sql(table_name, target_engine, if_exists="append", index=False)
        print(f"  Inserted {len(df)} rows into Aiven")

    print("\n✅ Migration complete.")


if __name__ == "__main__":
    migrate()