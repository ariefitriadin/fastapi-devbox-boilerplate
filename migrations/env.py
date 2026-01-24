import asyncio
import os
from logging.config import fileConfig
from urllib.parse import quote_plus
from dotenv import load_dotenv

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from app.core.config import settings
from alembic import context

from app.models.base import Base
from app.models.user import User

# --- IMPORT YOUR MODELS HERE ---
# from app.models.base import Base  # Example import
# target_metadata = Base.metadata
target_metadata = Base.metadata
# -------------------------------

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_url():
    # ...
    pg_host = os.getenv("PGHOST")  # e.g., /home/ari3/.../postgresql
    pg_user = os.getenv("USER", "ari3")
    pg_db = "postgres"
    pg_port = "5433"

    if pg_host:
        # We leave the hostname EMPTY (between @ and /)
        # but keep the port and the ?host query parameter.
        return f"postgresql+asyncpg://{pg_user}@:{pg_port}/{pg_db}?host={quote_plus(pg_host)}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.
    """

    # Get the configuration from alembic.ini
    section = config.get_section(config.config_ini_section, {})

    # Inject our dynamic Devbox URL
    section["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()