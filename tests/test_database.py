"""Phase 1 storage/configuration checks, not future endpoint authorization tests."""

import hashlib
import sqlite3
from contextlib import closing
from datetime import date
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from api import models as m
from api.config import LEGACY_DATABASE, ConfigurationError, DatabaseConfig
from api.db import DatabaseSafetyError, build_engine, require_schema, transaction
from api.repositories import DatasetRepository


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"DATABASE_URL": "not-a-url-secret"},
        {"DATABASE_URL": "sqlite:///demo.sqlite3", "GRAVITY_MODE": "shared"},
        {"DATABASE_URL": "mysql://example/db"},
        {"DATABASE_URL": "sqlite:///demo.sqlite3", "GRAVITY_MODE": "unknown"},
        {"DATABASE_URL": "sqlite:///"},
        {"DATABASE_URL": "sqlite:///demo.sqlite3?uri=true"},
        {"DATABASE_URL": f"sqlite:///{LEGACY_DATABASE}"},
    ],
)
def test_invalid_or_legacy_configuration_is_rejected(values):
    """No implicit target, unsupported driver, mode bypass, or known legacy path."""
    with pytest.raises(ConfigurationError) as error:
        DatabaseConfig.from_environment(values)
    assert "not-a-url-secret" not in str(error.value)


def test_direct_configuration_cannot_bypass_mode_validation():
    """Constructing a settings object directly has the same safety rules."""
    with pytest.raises(ConfigurationError, match="PostgreSQL"):
        DatabaseConfig(url=make_url("sqlite:///:memory:"), mode="shared")


def test_configuration_redacts_credentials_and_normalizes_postgres():
    """Configuration repr must not expose a supplied password."""
    config = DatabaseConfig.from_environment(
        {"DATABASE_URL": "postgresql://test_user:test-placeholder@localhost/newdb", "GRAVITY_MODE": "shared"}
    )
    assert config.url.drivername == "postgresql+psycopg"
    assert "test-placeholder" not in repr(config)


def test_engine_does_not_create_files_or_directories_implicitly(tmp_path):
    """Only an explicit migration/test may create a new SQLite file."""
    target = tmp_path / "new.sqlite3"
    config = DatabaseConfig.from_environment({"DATABASE_URL": f"sqlite:///{target}"})
    with pytest.raises(DatabaseSafetyError, match="does not exist"):
        build_engine(config)
    assert not target.exists()
    engine = build_engine(config, allow_create=True)
    assert not target.exists()
    engine.dispose()
    nested = tmp_path / "uncreated" / "new.sqlite3"
    with pytest.raises(DatabaseSafetyError, match="parent directory"):
        build_engine(DatabaseConfig.from_environment({"DATABASE_URL": f"sqlite:///{nested}"}), allow_create=True)
    assert not nested.parent.exists()


@pytest.mark.parametrize("table", ["runtime_state", "runtime_datasets", "interventions", "unrelated_records"])
def test_legacy_or_unversioned_files_are_preserved_at_any_path(tmp_path, table):
    """Synthetic database preservation must not depend on a filename."""
    target = tmp_path / "existing.sqlite3"
    with closing(sqlite3.connect(target)) as connection, connection:
        connection.execute(f"CREATE TABLE {table} (value TEXT)")
        connection.execute(f"INSERT INTO {table} VALUES (?)", ("synthetic retained value",))
    before = hashlib.sha256(target.read_bytes()).digest()
    with pytest.raises(DatabaseSafetyError, match="protected"):
        build_engine(DatabaseConfig.from_environment({"DATABASE_URL": f"sqlite:///{target}"}), allow_create=True)
    assert hashlib.sha256(target.read_bytes()).digest() == before


def test_uninitialized_database_is_not_repaired(tmp_path):
    """A read reports missing migrations instead of creating tables."""
    config = DatabaseConfig.from_environment({"DATABASE_URL": f"sqlite:///{tmp_path / 'empty.sqlite3'}"})
    engine = build_engine(config, allow_create=True)
    try:
        with engine.connect() as connection:
            with pytest.raises(DatabaseSafetyError, match="schema setup"):
                require_schema(connection)
            assert sa.inspect(connection).get_table_names() == []
    finally:
        engine.dispose()


def test_every_sqlite_connection_enforces_foreign_keys(tmp_path):
    """A pool with simultaneous connections must enable FKs on each."""
    config = DatabaseConfig.from_environment({"DATABASE_URL": f"sqlite:///{tmp_path / 'connections.sqlite3'}"})
    engine = build_engine(config, allow_create=True)
    try:
        with engine.connect() as first, engine.connect() as second:
            assert first.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
            assert second.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
    finally:
        engine.dispose()


def test_postgresql_ddl_compilation_and_constraint_names():
    """Compilation is distinct from real PostgreSQL execution; prevent name collisions."""
    dialect = postgresql.dialect()
    ddl = "\n".join(str(sa.schema.CreateTable(t).compile(dialect=dialect)) for t in m.metadata.sorted_tables)
    assert "JSONB" in ddl and "UUID" in ddl and "ON DELETE RESTRICT" in ddl
    for table in m.metadata.tables.values():
        names = [dialect.identifier_preparer.format_constraint(c) for c in table.constraints]
        assert len(names) == len(set(names))
        assert all(len(name.strip('"')) <= 63 for name in names)
        for index in table.indexes:
            assert "CREATE INDEX" in str(sa.schema.CreateIndex(index).compile(dialect=dialect))


def test_overlapping_ids_and_repository_scope(db_engine, dataset_graph):
    """Identical external IDs remain separate across datasets and institutions."""
    ids = dataset_graph
    with transaction(db_engine) as connection:
        assert connection.scalar(sa.select(sa.func.count()).select_from(m.dataset_students)) == 3
        repository = DatasetRepository(connection, ids["i1"])
        assert repository.dataset(ids["d1"])["owner_user_id"] == ids["faculty_a"]
        assert repository.dataset(ids["d3"]) is None
        assert repository.version(ids["d1"], ids["v1"])["version_number"] == 1
        assert repository.version(ids["d1"], ids["v2"]) is None
        assert repository.selection(ids["faculty_a"])["dataset_id"] == ids["d1"]
        assert repository.selection(ids["faculty_b"])["dataset_id"] == ids["d2"]
        assert repository.selection(ids["outsider"]) is None
        assert repository.record(ids["d1"], ids["v1"], ids["r1"])["raw"] == {"student_id": "001"}
        assert repository.record(ids["d1"], ids["v2"], ids["r1"]) is None
        assert repository.student_identity(ids["d1"], " 001 ")["id"] == ids["s1"]
        assert repository.student_identity(ids["d2"], "001")["id"] == ids["s2"]
        assert repository.student_identity(ids["d3"], "001") is None
        with pytest.raises(ValueError):
            DatasetRepository(connection, None)


def test_repository_insert_is_normalized_and_caller_transactional(db_engine, dataset_graph):
    """Identity creation commits only with its caller and keeps display text."""
    ids = dataset_graph
    with transaction(db_engine) as connection:
        identifier = DatasetRepository(connection, ids["i1"]).add_student_identity(ids["d1"], "  Student_A  ")
    with transaction(db_engine) as connection:
        record = DatasetRepository(connection, ids["i1"]).student_identity(ids["d1"], "STUDENT_a")
        assert record["id"] == identifier and record["external_id"] == "Student_A"
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        DatasetRepository(connection, ids["i1"]).add_student_identity(ids["d1"], "student_A")
    with pytest.raises(RuntimeError), transaction(db_engine) as connection:
        DatasetRepository(connection, ids["i1"]).add_student_identity(ids["d1"], "rolled-back")
        raise RuntimeError("injected failure")
    with transaction(db_engine) as connection:
        assert DatasetRepository(connection, ids["i1"]).student_identity(ids["d1"], "rolled-back") is None


@pytest.mark.parametrize("value", [None, 123, " ", "a" * 129])
def test_repository_requires_bounded_textual_ids(value):
    """Reject coercion and empty/oversized identities before persistence."""
    with pytest.raises(ValueError):
        DatasetRepository._student_key(value)


@pytest.mark.parametrize(
    "wrong_field,wrong_id",
    [
        ("institution_id", "i2"),
        ("version_id", "v2"),
        ("student_id", "s2"),
        ("dataset_id", "d2"),
    ],
)
def test_record_references_cannot_cross_scopes(db_engine, dataset_graph, wrong_field, wrong_id):
    """Composite FKs reject mixed institution/dataset/version/student references."""
    ids = dataset_graph
    values = dict(
        institution_id=ids["i1"],
        dataset_id=ids["d1"],
        version_id=ids["v1"],
        student_id=ids["s1"],
        department_code="cse",
        snapshot_month=date(2026, 6, 30),
    )
    values[wrong_field] = ids[wrong_id]
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(m.student_records.insert().values(**values))


def test_record_department_must_be_declared_by_dataset(db_engine, dataset_graph):
    """An institution department alone does not put it in the dataset scope."""
    ids = dataset_graph
    with transaction(db_engine) as connection:
        connection.execute(m.departments.insert().values(institution_id=ids["i1"], code="ece", name="Synthetic ECE"))
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(
            m.student_records.update().where(m.student_records.c.id == ids["r1"]).values(department_code="ece")
        )


def test_owners_and_grants_require_same_institution_membership(db_engine, dataset_graph):
    """Structural constraints reject references to another institution's member."""
    ids = dataset_graph
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(
            m.datasets.update().where(m.datasets.c.id == ids["d1"]).values(owner_user_id=ids["outsider"])
        )
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(
            m.dataset_grants.insert().values(
                institution_id=ids["i1"], dataset_id=ids["d1"], user_id=ids["outsider"], granted_by_user_id=ids["admin"]
            )
        )
    with transaction(db_engine) as connection:
        connection.execute(
            m.dataset_grants.insert().values(
                institution_id=ids["i1"], dataset_id=ids["d1"], user_id=ids["viewer"], granted_by_user_id=ids["admin"]
            )
        )


@pytest.mark.parametrize("values", [{"version_id": None}, {"dataset_id": None}, {"revision": -1}])
def test_selection_requires_complete_reference_and_valid_revision(db_engine, dataset_graph, values):
    """Nullable selections are all-or-nothing and cannot have negative revisions."""
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(
            m.user_dataset_selections.update()
            .where(m.user_dataset_selections.c.user_id == dataset_graph["faculty_a"])
            .values(**values)
        )


def test_selection_cannot_cross_institutions(db_engine, dataset_graph):
    """A valid foreign version UUID is not a valid selection in another institution."""
    ids = dataset_graph
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(
            m.user_dataset_selections.update()
            .where(m.user_dataset_selections.c.user_id == ids["faculty_a"])
            .values(dataset_id=ids["d3"], version_id=ids["v3"])
        )


def test_intervention_reference_and_history_are_restricted(db_engine, dataset_graph):
    """Interventions cannot point at a mismatched record or silently cascade away."""
    ids = dataset_graph
    identifier = uuid4()
    values = dict(
        id=identifier,
        institution_id=ids["i1"],
        dataset_id=ids["d1"],
        version_id=ids["v1"],
        student_id=ids["s1"],
        record_id=ids["r1"],
        created_by_user_id=ids["faculty_a"],
        type="Synthetic follow-up",
        start_date=date(2026, 7, 1),
        before_snapshot={"success_score": 61},
    )
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(m.interventions.insert().values(**(values | {"record_id": ids["r2"]})))
    with transaction(db_engine) as connection:
        connection.execute(m.interventions.insert().values(**values))
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(m.student_records.delete().where(m.student_records.c.id == ids["r1"]))
    with transaction(db_engine) as connection:
        repository = DatasetRepository(connection, ids["i1"])
        assert repository.intervention(ids["d1"], identifier)["before_snapshot"] == {"success_score": 61}
        assert repository.intervention(ids["d2"], identifier) is None
        assert DatasetRepository(connection, ids["i2"]).intervention(ids["d1"], identifier) is None


def test_failed_unit_of_work_rolls_back_rows_and_selection(db_engine, dataset_graph):
    """Fault injection after inserts and a pointer update preserves the baseline."""
    ids = dataset_graph
    candidate = uuid4()
    with pytest.raises(RuntimeError, match="injected"), transaction(db_engine) as connection:
        connection.execute(
            m.dataset_versions.insert().values(
                id=candidate,
                institution_id=ids["i1"],
                dataset_id=ids["d1"],
                created_by_user_id=ids["faculty_a"],
                parent_version_id=ids["v1"],
                version_number=2,
                student_count=1,
                schema_version="1",
            )
        )
        connection.execute(
            m.student_records.insert().values(
                institution_id=ids["i1"],
                dataset_id=ids["d1"],
                version_id=candidate,
                student_id=ids["s1"],
                department_code="cse",
                snapshot_month=date(2026, 7, 31),
            )
        )
        connection.execute(
            m.user_dataset_selections.update()
            .where(m.user_dataset_selections.c.user_id == ids["faculty_a"])
            .values(version_id=candidate, revision=1)
        )
        raise RuntimeError("injected failure")
    with transaction(db_engine) as connection:
        repository = DatasetRepository(connection, ids["i1"])
        assert repository.version(ids["d1"], candidate) is None
        assert repository.selection(ids["faculty_a"])["version_id"] == ids["v1"]
        assert connection.scalar(sa.select(sa.func.count()).select_from(m.student_records)) == 3


def test_unknown_role_and_invalid_score_are_rejected(db_engine, dataset_graph):
    """Domain checks hold even without the future request-validation layer."""
    ids = dataset_graph
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(
            m.memberships.update().where(m.memberships.c.user_id == ids["faculty_a"]).values(role="superadmin")
        )
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(
            m.student_records.update().where(m.student_records.c.id == ids["r1"]).values(academic_risk_probability=1.1)
        )


def test_version_numbers_and_parent_scope_are_constrained(db_engine, dataset_graph):
    """Versions are unique per dataset and cannot inherit another dataset's parent."""
    ids = dataset_graph
    values = dict(
        institution_id=ids["i1"],
        dataset_id=ids["d1"],
        created_by_user_id=ids["faculty_a"],
        version_number=1,
        student_count=1,
        schema_version="1",
    )
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(m.dataset_versions.insert().values(**values))
    with pytest.raises(IntegrityError), transaction(db_engine) as connection:
        connection.execute(
            m.dataset_versions.insert().values(**(values | {"version_number": 2, "parent_version_id": ids["v2"]}))
        )


def test_clear_one_selection_preserves_other_selection_and_records(db_engine, dataset_graph):
    """Storage-level independence only; authenticated reset workflows are deferred."""
    ids = dataset_graph
    with transaction(db_engine) as connection:
        connection.execute(
            m.user_dataset_selections.update()
            .where(
                m.user_dataset_selections.c.institution_id == ids["i1"],
                m.user_dataset_selections.c.user_id == ids["faculty_a"],
            )
            .values(dataset_id=None, version_id=None, revision=1)
        )
    with transaction(db_engine) as connection:
        repository = DatasetRepository(connection, ids["i1"])
        assert repository.selection(ids["faculty_a"])["version_id"] is None
        assert repository.selection(ids["faculty_b"])["version_id"] == ids["v2"]
        assert connection.scalar(sa.select(sa.func.count()).select_from(m.dataset_versions)) == 3
