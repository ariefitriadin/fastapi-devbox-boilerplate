from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """
    Base class for all database models.
    Provides the 'metadata' object that Alembic needs.
    """
    pass