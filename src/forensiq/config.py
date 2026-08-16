"""
Central configuration for ForensIQ.

Why this file exists:
Every module (ingestion, db, api, llm) needs settings like DB credentials
or API keys. Instead of each module calling os.getenv() separately
(error-prone, no validation, fails silently if a var is missing),
we load everything once here through Pydantic Settings. This gives us:
  - Type validation (e.g. DB_PORT must be an int, not a random string)
  - A single source of truth
  - A loud, early failure at startup if a required variable is missing,
    instead of a confusing error three layers deep at runtime.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    db_host: str
    db_port: int = 3306
    db_user: str
    db_password: str
    db_name: str
    groq_api_key :str
    # SEC EDGAR
    # SEC requires a descriptive User-Agent on every request (name + email).
    # Requests without one, or with a generic one, get blocked.
    sec_user_agent: str

    # LLM (used later, Phase 7)
    gemini_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def database_url(self) -> str:
        """Builds the SQLAlchemy connection string from individual DB settings."""
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


# Loaded once, imported everywhere else as `from forensiq.config import settings`
settings = Settings()