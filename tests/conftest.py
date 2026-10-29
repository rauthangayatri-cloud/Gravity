"""Opt-in foundation fixtures using temporary databases and synthetic records only."""

import os
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from api import models as m
from api.config import DatabaseConfig
from api.db import build_engine

ROOT = Path(__file__).resolve().parents[1]


def migration_configuration() -> Config:
    """Require explicit consent even for the test-owned target."""
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["confirm_schema_change"] = True
    return config


@pytest.fixture
def alembic_config():
    """Supply a new configuration for each migration test."""
    return migration_configuration()


@pytest.fixture(params=["sqlite", "postgresql"])
def db_engine(request, tmp_path):
    """Run both dialects; PostgreSQL requires an explicitly supplied disposable target."""
    admin_engine = None
    schema_name = None
    if request.param == "sqlite":
        settings = DatabaseConfig.from_environment({"DATABASE_URL": f"sqlite:///{tmp_path / 'foundation.sqlite3'}"})
        engine = build_engine(settings, allow_create=True)
    else:
        raw = os.environ.get("GRAVITY_TEST_POSTGRES_URL", "")
        if not raw:
            pytest.skip("PostgreSQL not run: GRAVITY_TEST_POSTGRES_URL is not configured.")
        settings = DatabaseConfig.from_environment({"DATABASE_URL": raw, "GRAVITY_MODE": "shared"})
        if (
            settings.url.host not in {"localhost", "127.0.0.1", "::1"}
            or not str(settings.url.database).startswith("gravity_test_")
            or settings.url.query
        ):
            pytest.fail("Use a disposable local gravity_test_* PostgreSQL database URL without query overrides.")
        schema_name = "gravity_test_" + uuid4().hex
        admin_engine = build_engine(settings)
        try:
            with admin_engine.begin() as connection:
                connection.execute(sa.schema.CreateSchema(schema_name))
        except Exception:
            admin_engine.dispose()
            raise
        scoped_url = settings.url.update_query_dict({"options": f"-csearch_path={schema_name}"})
        engine = build_engine(DatabaseConfig(url=scoped_url, mode="shared"))
    try:
        config = migration_configuration()
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield engine
    finally:
        engine.dispose()
        if admin_engine is not None:
            try:
                with admin_engine.begin() as connection:
                    # Only this fixture's unique schema is removed, never the database.
                    connection.execute(sa.schema.DropSchema(schema_name, cascade=True))
            finally:
                admin_engine.dispose()


@pytest.fixture
def dataset_graph(db_engine):
    """Two faculty in one institution plus a second institution, with overlapping IDs."""
    names = [
        "i1",
        "i2",
        "faculty_a",
        "faculty_b",
        "admin",
        "viewer",
        "outsider",
        "d1",
        "d2",
        "d3",
        "v1",
        "v2",
        "v3",
        "s1",
        "s2",
        "s3",
        "r1",
        "r2",
        "r3",
    ]
    ids = {name: UUID(int=index) for index, name in enumerate(names, start=1)}
    with db_engine.begin() as connection:
        for name in ("i1", "i2"):
            connection.execute(m.institutions.insert().values(id=ids[name], name="Synthetic " + name))
        for name, role, institution in (
            ("faculty_a", "faculty", "i1"),
            ("faculty_b", "faculty", "i1"),
            ("admin", "admin", "i1"),
            ("viewer", "viewer", "i1"),
            ("outsider", "faculty", "i2"),
        ):
            connection.execute(
                m.users.insert().values(
                    id=ids[name], username_key=name, password_hash="synthetic-schema-test-not-a-login-hash"
                )
            )
            connection.execute(
                m.memberships.insert().values(institution_id=ids[institution], user_id=ids[name], role=role)
            )
        for name in ("i1", "i2"):
            connection.execute(
                m.departments.insert().values(institution_id=ids[name], code="cse", name="Synthetic CSE")
            )
        for number, owner, institution in ((1, "faculty_a", "i1"), (2, "faculty_b", "i1"), (3, "outsider", "i2")):
            scope = {"institution_id": ids[institution], "dataset_id": ids[f"d{number}"]}
            connection.execute(
                m.membership_departments.insert().values(
                    institution_id=ids[institution], user_id=ids[owner], department_code="cse"
                )
            )
            connection.execute(
                m.datasets.insert().values(
                    id=scope["dataset_id"],
                    institution_id=scope["institution_id"],
                    owner_user_id=ids[owner],
                    label="Synthetic dataset",
                    kind="upload",
                )
            )
            connection.execute(m.dataset_departments.insert().values(**scope, department_code="cse"))
            connection.execute(
                m.dataset_versions.insert().values(
                    **scope,
                    id=ids[f"v{number}"],
                    created_by_user_id=ids[owner],
                    version_number=1,
                    student_count=1,
                    schema_version="1",
                    provenance={"synthetic": True},
                )
            )
            connection.execute(
                m.dataset_students.insert().values(
                    **scope, id=ids[f"s{number}"], external_id="001", external_id_key="001"
                )
            )
            connection.execute(
                m.student_records.insert().values(
                    **scope,
                    id=ids[f"r{number}"],
                    version_id=ids[f"v{number}"],
                    student_id=ids[f"s{number}"],
                    department_code="cse",
                    snapshot_month=date(2026, 6, 30),
                    success_score=60 + number,
                    raw={"student_id": "001"},
                    derived={"synthetic": True},
                )
            )
            connection.execute(
                m.user_dataset_selections.insert().values(**scope, user_id=ids[owner], version_id=ids[f"v{number}"])
            )
    return ids
