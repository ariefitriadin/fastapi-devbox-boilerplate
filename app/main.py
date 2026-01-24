from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.database import get_db
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Production-ready FastAPI boilerplate with Devbox and Postgres"
)

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
            "version": "0.1.0"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database": "error",
            "detail": str(e)
        }

@app.get("/", tags=["System"])
def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}