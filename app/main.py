from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints import ai
from app.core.config import settings
from app.core.database import get_db

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Production-ready FastAPI boilerplate with Devbox, Postgres, and AI/ML",
)

# Include AI endpoints
app.include_router(ai.router, prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Service Health Check
    Verifies:
    1. API is responsive
    2. Database connection via Unix Socket is active
    """
    try:
        # Check database connectivity
        await db.execute(text("SELECT 1"))
        return {
            "status": "online",
            "database": "connected",
            "environment": "devbox",
            "version": "0.1.0",
        }
    except Exception as e:
        return {"status": "degraded", "database": "error", "detail": str(e)}


@app.get("/", tags=["System"])
def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}
