"""Database engine and session configuration for FlowCast."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


def build_engine(database_url: str) -> Engine:
    """Build a SQLAlchemy engine using explicit connection settings."""
    if not database_url or not database_url.strip():
        raise ValueError("database_url must not be empty.")

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
    )


def build_engine_from_environment() -> Engine:
    """Build the database engine using DATABASE_URL from the environment."""
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Configure it in your .env file."
        )

    return build_engine(database_url)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a reusable SQLAlchemy session factory."""
    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(
    factory: sessionmaker[Session],
) -> Iterator[Session]:
    """Provide a transactional session with automatic cleanup."""
    with factory.begin() as session:
        yield session
