"""Shared PostgreSQL connection helper.

Reads DATABASE_URL from the environment (.env), so every script gets the
same connection without duplicating credentials.
"""
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load .env from the project root regardless of which script imports this.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import os

DATABASE_URL = os.environ.get("DATABASE_URL")


def _normalize(url: str) -> str:
    """Many managed Postgres providers (Supabase, Render, Heroku-style
    hosts) hand out connection strings starting with "postgres://", which
    SQLAlchemy 1.4+ no longer accepts (it needs the exact dialect name,
    "postgresql://"). Fixed once here rather than requiring every
    deployment target to hand-edit its own URL.
    """
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def get_engine():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    return create_engine(_normalize(DATABASE_URL))
