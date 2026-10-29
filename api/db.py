"""Connection/transaction helpers; never initialize or migrate at runtime."""

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import StaticPool

from api.config import DatabaseConfig

SCHEMA_REVISION = "0001_identity_and_datasets"
LEGACY_TABLES = {"runtime_datasets", "runtime_state"}


class DatabaseSafetyError(RuntimeError):
    """Refuse legacy, uninitialized, or incompatible databases without altering them."""


def _check_sqlite_file(path: Path, allow_create: bool) -> None:
    """Inspect existing files read-only before SQLAlchemy opens a connection."""
    if not path.parent.is_dir():
        raise DatabaseSafetyError("The chosen SQLite parent directory must already exist.")
    if not path.exists():
        if not allow_create:
            raise DatabaseSafetyError("Database does not exist; explicit schema setup is required.")
        return
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_schema WHERE type IN ('table', 'view')")
            if not row[0].startswith("sqlite_")
        }
    if names & LEGACY_TABLES or (names and "alembic_version" not in names):
        raise DatabaseSafetyError("Existing unversioned/legacy data is protected; choose a new database.")


def build_engine(config: DatabaseConfig, *, allow_create: bool = False) -> Engine:
    """Construct a lazy engine; file creation is opt-in for explicit migrations/tests."""
    if config.url.get_backend_name() == "sqlite":
        in_memory = config.url.database == ":memory:"
        if not in_memory:
            _check_sqlite_file(Path(str(config.url.database)), allow_create)
        kwargs = {"poolclass": StaticPool} if in_memory else {}
        if not in_memory:
            # mode=rw prevents an ordinary read from recreating a removed file.
            sqlite_uri = Path(str(config.url.database)).as_uri() + ("?mode=rwc" if allow_create else "?mode=rw")
            kwargs["creator"] = lambda: sqlite3.connect(sqlite_uri, uri=True, check_same_thread=False, timeout=10)
        engine = create_engine(
            config.url,
            hide_parameters=True,
            connect_args={"check_same_thread": False, "timeout": 10},
            **kwargs,
        )

        @event.listens_for(engine, "connect")
        def configure_sqlite(dbapi_connection, _connection_record):
            """Enable FKs on every connection and opt out of legacy transaction control."""
            dbapi_connection.isolation_level = None
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        @event.listens_for(engine, "begin")
        def begin_sqlite(connection):
            """Make reads, savepoints, and DDL participate in real transactions."""
            connection.exec_driver_sql("BEGIN")

        return engine
    return create_engine(
        config.url,
        hide_parameters=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": 10},
    )


def check_migration_target(connection: Connection) -> None:
    """Permit only an empty database/schema or the recognized versioned schema."""
    from api.models import metadata

    inspector = inspect(connection)
    names = set(inspector.get_table_names())
    if inspector.get_view_names():
        raise DatabaseSafetyError("Target contains existing views; choose a dedicated new database/schema.")
    if names & LEGACY_TABLES or (names and "alembic_version" not in names):
        raise DatabaseSafetyError("Refusing schema changes to unversioned/legacy data.")
    if names - set(metadata.tables) - {"alembic_version"}:
        raise DatabaseSafetyError("Target contains unrelated tables; choose a dedicated new database/schema.")
    if "alembic_version" in names:
        revisions = connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
        if revisions != [SCHEMA_REVISION]:
            raise DatabaseSafetyError("Target has an unknown or incomplete schema revision.")
        if not set(metadata.tables).issubset(names):
            raise DatabaseSafetyError("Explicit schema setup is incomplete; automatic repair is not allowed.")
        for table in metadata.tables.values():
            actual = {column["name"] for column in inspector.get_columns(table.name)}
            if actual != set(table.c.keys()):
                raise DatabaseSafetyError("Schema columns do not match the declared revision; repair requires review.")


def require_schema(connection: Connection) -> None:
    """Validate the schema without performing any initialization or repair."""
    from api.models import metadata

    check_migration_target(connection)
    names = set(inspect(connection).get_table_names())
    if "alembic_version" not in names or not set(metadata.tables).issubset(names):
        raise DatabaseSafetyError("Explicit Alembic schema setup is required before database use.")


@contextmanager
def transaction(engine: Engine) -> Iterator[Connection]:
    """Commit an entire unit of work, or roll it back on any exception."""
    with engine.begin() as connection:
        require_schema(connection)
        yield connection
