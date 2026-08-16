"""
Database engine + session management, plus a helper to create all
tables from our ORM models (Phase 1 approach — no Alembic yet).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from forensiq.config import settings
from forensiq.db.models import Base

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session() -> Session:
    return SessionLocal()


def init_db() -> None:
    """Creates the `forensiq` database if missing, then creates all tables."""
    import pymysql

    conn = pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
    )
    with conn.cursor() as cursor:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {settings.db_name}")
    conn.close()

    Base.metadata.create_all(bind=engine)
    print("Database and tables created successfully.")


if __name__ == "__main__":
    init_db()