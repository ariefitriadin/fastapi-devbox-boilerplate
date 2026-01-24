from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings
from typing import AsyncGenerator

# 1. Create the Engine
# echo=True allows you to see the SQL queries in your terminal (great for dev)
engine = create_async_engine(
    settings.assemble_db_url,
    echo=settings.DEBUG,
    future=True
)

# 2. Create a Session Factory
# expire_on_commit=False is important for async to prevent accidental lazy loading errors
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# 3. Dependency to get DB session in FastAPI routes
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()