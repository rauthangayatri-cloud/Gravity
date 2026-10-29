"""Explicit online migrations only; reject legacy databases before running DDL."""

from alembic import context

from api.config import DatabaseConfig
from api.db import build_engine, check_migration_target
from api.models import metadata

config = context.config


def run_migrations(connection):
    """Check the target inside the same transaction used by Alembic."""
    check_migration_target(connection)
    context.configure(connection=connection, target_metadata=metadata, compare_type=True, compare_server_default=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Offline execution is disabled because target safety cannot be inspected.")

confirmed = config.attributes.get("confirm_schema_change") is True
confirmed = confirmed or context.get_x_argument(as_dictionary=True).get("confirm_schema_change") == "yes"
if not confirmed:
    raise RuntimeError("Explicit consent required: alembic -x confirm_schema_change=yes upgrade head")

supplied_connection = config.attributes.get("connection")
if supplied_connection is not None:
    run_migrations(supplied_connection)
else:
    engine = build_engine(DatabaseConfig.from_environment(), allow_create=True)
    try:
        with engine.begin() as connection:
            run_migrations(connection)
    finally:
        engine.dispose()
