"""Database schema initialization for FlowCast."""

from sqlalchemy.engine import Engine

from src.day20.database import Base
from src.day20.models import TrafficObservation  # noqa: F401


def initialize_schema(engine: Engine) -> None:
    """Create missing tables registered with the SQLAlchemy metadata."""
    Base.metadata.create_all(bind=engine)
