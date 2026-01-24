import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import quote_plus


class Settings(BaseSettings):
    # Pydantic will look for these names in your .env file
    DATABASE_URL: Optional[str] = None
    APP_NAME: str = "FastAPI App"
    DEBUG: bool = False

    @property
    def assemble_db_url(self) -> str:
        """
        Logic to handle both External URLs and Devbox Sockets.
        """
        # 1. Use DATABASE_URL if provided in .env
        if self.DATABASE_URL:
            # Fix for common 'postgresql://' vs 'postgresql+asyncpg://' issue
            if self.DATABASE_URL.startswith("postgresql://"):
                return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
            return self.DATABASE_URL

        # 2. Fallback to Devbox Socket environment variables
        pg_host = os.getenv("PGHOST")
        if pg_host:
            pg_user = os.getenv("USER", "postgres")
            return f"postgresql+asyncpg://{pg_user}@/postgres?host={quote_plus(pg_host)}"

        raise RuntimeError("No database configuration found in .env or Devbox environment.")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()