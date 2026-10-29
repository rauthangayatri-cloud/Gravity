"""Explicit, side-effect-free configuration for the new database layer."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATABASE = PROJECT_ROOT / "api" / "interventions.sqlite3"


class ConfigurationError(ValueError):
    """Report invalid configuration without disclosing connection credentials."""


@dataclass(frozen=True)
class DatabaseConfig:
    """Require an explicit new target; never fall back to the legacy database."""

    url: URL = field(repr=False)
    mode: str = "local"

    def __post_init__(self) -> None:
        """Validate direct construction as well as environment-based configuration."""
        url = self.url
        if not isinstance(url, URL):
            raise ConfigurationError("DatabaseConfig requires a parsed SQLAlchemy URL.")
        if self.mode not in {"local", "shared"}:
            raise ConfigurationError("GRAVITY_MODE must be local or shared.")
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        if url.drivername not in {"sqlite", "sqlite+pysqlite", "postgresql+psycopg"}:
            raise ConfigurationError("Use SQLite locally or PostgreSQL with psycopg for shared deployment.")
        if self.mode == "shared" and url.get_backend_name() != "postgresql":
            raise ConfigurationError("Shared deployment requires PostgreSQL.")
        if not url.database:
            raise ConfigurationError("DATABASE_URL must name a database.")
        if url.get_backend_name() == "sqlite":
            if url.query or url.username or url.password or url.host or url.port:
                raise ConfigurationError("SQLite URLs must be plain file paths, without URI/query overrides.")
            if url.database != ":memory:":
                target = Path(url.database).resolve()
                if target == LEGACY_DATABASE.resolve():
                    raise ConfigurationError("The legacy SQLite file is protected; choose a new database path.")
                url = url.set(database=str(target))
        object.__setattr__(self, "url", url)

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "DatabaseConfig":
        """Read process settings only; importing this module never loads files."""
        values = os.environ if environ is None else environ
        raw = values.get("DATABASE_URL", "").strip()
        if not raw:
            raise ConfigurationError("DATABASE_URL must explicitly identify a new database.")
        try:
            url = make_url(raw)
        except (ArgumentError, ValueError):
            raise ConfigurationError("DATABASE_URL is not a valid database URL.") from None
        return cls(url=url, mode=values.get("GRAVITY_MODE", "local").strip().lower())
