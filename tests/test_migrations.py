"""Fresh-schema setup must be explicit, reproducible, and protective of legacy data."""

import hashlib
import io
import runpy
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations

from api.config import DatabaseConfig
from api.db import SCHEMA_REVISION, DatabaseSafetyError, build_engine, require_schema
from api.models import metadata


def test_migration_matches_models_and_is_repeatable(db_engine, alembic_config):
    """The frozen revision matches the model; a repeated upgrade does not change data."""
    with db_engine.begin() as connection:
        require_schema(connection)
        assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == SCHEMA_REVISION
        context = MigrationContext.configure(connection, opts={"compare_type": True, "compare_server_default": True})
        assert compare_metadata(context, metadata) == []
        alembic_config.attributes["connection"] = connection
        command.upgrade(alembic_config, "head")
        assert set(sa.inspect(connection).get_table_names()) == set(metadata.tables) | {"alembic_version"}
        for table in metadata.tables.values():
            assert connection.scalar(sa.select(sa.func.count()).select_from(table)) == 0


def test_migration_requires_confirmation_before_connecting(tmp_path, monkeypatch, alembic_config):
    """Even a supplied new target is not created without explicit schema consent."""
    target = tmp_path / "unapproved.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{target}")
    alembic_config.attributes.clear()
    with pytest.raises(RuntimeError, match="Explicit consent"):
        command.upgrade(alembic_config, "head")
    assert not target.exists()


def test_environment_migration_creates_only_new_target(tmp_path, monkeypatch, alembic_config):
    """Exercise actual environment settings without using project runtime files."""
    target = tmp_path / "approved-new.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{target}")
    monkeypatch.setenv("GRAVITY_MODE", "local")
    command.upgrade(alembic_config, "head")
    engine = build_engine(DatabaseConfig.from_environment())
    try:
        with engine.connect() as connection:
            require_schema(connection)
            assert connection.scalar(sa.text("SELECT count(*) FROM users")) == 0
            assert connection.scalar(sa.text("SELECT count(*) FROM datasets")) == 0
    finally:
        engine.dispose()


def test_legacy_connection_is_refused_without_changing_source(tmp_path, alembic_config):
    """The guard protects caller-supplied connections as well as configured targets."""
    target = tmp_path / "legacy-copy.sqlite3"
    with closing(sqlite3.connect(target)) as connection, connection:
        connection.execute("CREATE TABLE runtime_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO runtime_state VALUES ('active_dataset_id', 'synthetic-legacy-id')")
    before = hashlib.sha256(target.read_bytes()).digest()
    engine = sa.create_engine(f"sqlite:///{target}")
    try:
        with engine.connect() as connection:
            alembic_config.attributes["connection"] = connection
            with pytest.raises(DatabaseSafetyError, match="legacy"):
                command.upgrade(alembic_config, "head")
    finally:
        engine.dispose()
    assert hashlib.sha256(target.read_bytes()).digest() == before


def test_destructive_downgrade_is_disabled(db_engine, alembic_config):
    """A downgrade cannot silently remove tables or history."""
    with db_engine.connect() as connection:
        alembic_config.attributes["connection"] = connection
        with pytest.raises(RuntimeError, match="Destructive downgrade"):
            command.downgrade(alembic_config, "base")
    with db_engine.connect() as connection:
        require_schema(connection)


def test_unknown_revision_and_partial_schema_fail_closed(db_engine):
    """Runtime checks do not auto-upgrade, repair, or fall back to demo data."""
    with db_engine.connect() as connection:
        with connection.begin():
            connection.execute(sa.text("UPDATE alembic_version SET version_num = 'unknown-test-revision'"))
            with pytest.raises(DatabaseSafetyError, match="unknown"):
                require_schema(connection)
            connection.rollback()
        with connection.begin():
            connection.execute(sa.schema.DropTable(metadata.tables["user_dataset_selections"]))
            with pytest.raises(DatabaseSafetyError, match="schema setup"):
                require_schema(connection)
            connection.rollback()
        with connection.begin():
            connection.execute(sa.text("ALTER TABLE users ADD COLUMN unexpected_test_column INTEGER"))
            with pytest.raises(DatabaseSafetyError, match="columns"):
                require_schema(connection)
            connection.rollback()
        require_schema(connection)


def test_migration_failure_rolls_back_all_new_tables(tmp_path, monkeypatch, alembic_config):
    """SQLite DDL must be transactional, including Alembic's version table."""
    target = tmp_path / "failed-new.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{target}")
    monkeypatch.setenv("GRAVITY_MODE", "local")
    original = Operations.create_table

    def fail_midway(self, table_name, *args, **kwargs):
        """Inject failure after several tables have been created."""
        if table_name == "student_records":
            raise RuntimeError("injected DDL failure")
        return original(self, table_name, *args, **kwargs)

    monkeypatch.setattr(Operations, "create_table", fail_midway)
    with pytest.raises(RuntimeError, match="injected DDL"):
        command.upgrade(alembic_config, "head")
    with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)) as connection:
        assert connection.execute("SELECT name FROM sqlite_schema WHERE type='table'").fetchall() == []


def test_initial_revision_renders_portable_postgresql_ddl():
    """Inspect frozen-revision SQL without connecting to a PostgreSQL server."""
    revision_path = Path(__file__).resolve().parents[1] / "migrations" / "versions" / f"{SCHEMA_REVISION}.py"
    revision = runpy.run_path(str(revision_path))
    output = io.StringIO()
    context = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output})
    with Operations.context(context):
        revision["upgrade"]()
    sql = output.getvalue()
    assert sql.count("CREATE TABLE ") == len(metadata.tables)
    assert "BOOLEAN DEFAULT true" in sql
    assert "BOOLEAN DEFAULT false" in sql
    assert "BOOLEAN DEFAULT 1" not in sql
    assert "JSONB" in sql and "UUID" in sql
    assert "api.models" not in revision_path.read_text(encoding="utf-8")


def test_offline_migration_is_disabled(alembic_config):
    """Do not offer a shortcut that bypasses target inspection."""
    with pytest.raises(RuntimeError, match="Offline execution"):
        command.upgrade(alembic_config, "head", sql=True)
