"""Portable relational schema; tables are inert until an explicit migration runs."""

import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)
JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def _id(name="id", *, primary_key=False, nullable=False):
    """Use portable UUIDs; generated IDs are identifiers, not seeded ML randomness."""
    return sa.Column(
        name, sa.Uuid(), primary_key=primary_key, nullable=nullable, default=uuid.uuid4 if name == "id" else None
    )


def _created():
    """Record insertion time without requiring a client-provided timestamp."""
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.current_timestamp()
    )


def _json(name, empty="{}"):
    """Preserve flexible precomputed payloads without losing relational identity."""
    return sa.Column(name, JSON_TYPE, nullable=False, server_default=sa.text("'" + empty + "'"))


def _fk(local, target):
    """Historical identity references must not cascade-delete records."""
    return sa.ForeignKeyConstraint(local, target, ondelete="RESTRICT")


institutions = sa.Table(
    "institutions",
    metadata,
    _id(primary_key=True),
    sa.Column("name", sa.String(200), nullable=False),
    sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
    _created(),
)
users = sa.Table(
    "users",
    metadata,
    _id(primary_key=True),
    sa.Column("username_key", sa.String(200), nullable=False, unique=True),
    sa.Column("password_hash", sa.Text(), nullable=False),
    sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
    sa.Column("token_version", sa.Integer(), nullable=False, server_default="0"),
    _created(),
    sa.CheckConstraint("length(trim(username_key)) > 0", name="username_present"),
    sa.CheckConstraint("length(password_hash) > 0", name="password_hash_present"),
    sa.CheckConstraint("token_version >= 0", name="token_version_nonnegative"),
)
memberships = sa.Table(
    "memberships",
    metadata,
    _id("institution_id", primary_key=True),
    _id("user_id", primary_key=True),
    sa.Column("role", sa.String(16), nullable=False),
    sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
    _fk(["institution_id"], ["institutions.id"]),
    _fk(["user_id"], ["users.id"]),
    sa.CheckConstraint("role IN ('admin', 'faculty', 'viewer')", name="valid_role"),
)
departments = sa.Table(
    "departments",
    metadata,
    _id("institution_id", primary_key=True),
    sa.Column("code", sa.String(64), primary_key=True),
    sa.Column("name", sa.String(200), nullable=False),
    _fk(["institution_id"], ["institutions.id"]),
    sa.CheckConstraint("length(trim(code)) > 0", name="code_present"),
)
membership_departments = sa.Table(
    "membership_departments",
    metadata,
    _id("institution_id", primary_key=True),
    _id("user_id", primary_key=True),
    sa.Column("department_code", sa.String(64), primary_key=True),
    _fk(["institution_id", "user_id"], ["memberships.institution_id", "memberships.user_id"]),
    _fk(["institution_id", "department_code"], ["departments.institution_id", "departments.code"]),
)
datasets = sa.Table(
    "datasets",
    metadata,
    _id(primary_key=True),
    _id("institution_id"),
    _id("owner_user_id"),
    sa.Column("label", sa.String(255), nullable=False),
    sa.Column("kind", sa.String(16), nullable=False),
    sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
    _created(),
    _fk(["institution_id", "owner_user_id"], ["memberships.institution_id", "memberships.user_id"]),
    sa.UniqueConstraint("institution_id", "id", name="uq_datasets_scope"),
    sa.CheckConstraint("kind IN ('upload', 'individual', 'demo')", name="valid_kind"),
    sa.CheckConstraint("revision >= 0", name="revision_nonnegative"),
    sa.CheckConstraint("length(trim(label)) > 0", name="label_present"),
)
dataset_departments = sa.Table(
    "dataset_departments",
    metadata,
    _id("institution_id", primary_key=True),
    _id("dataset_id", primary_key=True),
    sa.Column("department_code", sa.String(64), primary_key=True),
    _fk(["institution_id", "dataset_id"], ["datasets.institution_id", "datasets.id"]),
    _fk(["institution_id", "department_code"], ["departments.institution_id", "departments.code"]),
)
dataset_grants = sa.Table(
    "dataset_grants",
    metadata,
    _id("institution_id", primary_key=True),
    _id("dataset_id", primary_key=True),
    _id("user_id", primary_key=True),
    _id("granted_by_user_id"),
    _created(),
    _fk(["institution_id", "dataset_id"], ["datasets.institution_id", "datasets.id"]),
    _fk(["institution_id", "user_id"], ["memberships.institution_id", "memberships.user_id"]),
    _fk(["institution_id", "granted_by_user_id"], ["memberships.institution_id", "memberships.user_id"]),
)
dataset_versions = sa.Table(
    "dataset_versions",
    metadata,
    _id(primary_key=True),
    _id("institution_id"),
    _id("dataset_id"),
    _id("created_by_user_id"),
    _id("parent_version_id", nullable=True),
    sa.Column("version_number", sa.Integer(), nullable=False),
    sa.Column("student_count", sa.Integer(), nullable=False),
    sa.Column("schema_version", sa.String(64), nullable=False),
    sa.Column("source_sha256", sa.String(64), nullable=True),
    _json("provenance"),
    _json("quality"),
    _json("segments", "[]"),
    _created(),
    _fk(["institution_id", "dataset_id"], ["datasets.institution_id", "datasets.id"]),
    _fk(["institution_id", "created_by_user_id"], ["memberships.institution_id", "memberships.user_id"]),
    _fk(
        ["institution_id", "dataset_id", "parent_version_id"],
        ["dataset_versions.institution_id", "dataset_versions.dataset_id", "dataset_versions.id"],
    ),
    sa.UniqueConstraint("institution_id", "dataset_id", "id", name="uq_versions_scope"),
    sa.UniqueConstraint("institution_id", "dataset_id", "version_number", name="uq_versions_number"),
    sa.CheckConstraint("version_number > 0", name="positive_version"),
    sa.CheckConstraint("student_count >= 0", name="student_count_nonnegative"),
    sa.CheckConstraint("parent_version_id IS NULL OR parent_version_id <> id", name="not_own_parent"),
)
dataset_students = sa.Table(
    "dataset_students",
    metadata,
    _id(primary_key=True),
    _id("institution_id"),
    _id("dataset_id"),
    sa.Column("external_id", sa.String(128), nullable=False),
    sa.Column("external_id_key", sa.String(128), nullable=False),
    _fk(["institution_id", "dataset_id"], ["datasets.institution_id", "datasets.id"]),
    sa.UniqueConstraint("institution_id", "dataset_id", "id", name="uq_students_scope"),
    sa.UniqueConstraint("institution_id", "dataset_id", "external_id_key", name="uq_students_external_key"),
    sa.CheckConstraint(
        "length(trim(external_id)) > 0 AND length(trim(external_id_key)) > 0", name="external_id_present"
    ),
)
student_records = sa.Table(
    "student_records",
    metadata,
    _id(primary_key=True),
    _id("institution_id"),
    _id("dataset_id"),
    _id("version_id"),
    _id("student_id"),
    sa.Column("department_code", sa.String(64), nullable=False),
    sa.Column("snapshot_month", sa.Date(), nullable=False),
    sa.Column("success_score", sa.Float(), nullable=True),
    sa.Column("academic_risk_probability", sa.Float(), nullable=True),
    sa.Column("placement_risk_probability", sa.Float(), nullable=True),
    sa.Column("priority_score", sa.Float(), nullable=True),
    _json("raw"),
    _json("derived"),
    _json("drivers", "[]"),
    _json("actions", "[]"),
    _fk(
        ["institution_id", "dataset_id", "version_id"],
        ["dataset_versions.institution_id", "dataset_versions.dataset_id", "dataset_versions.id"],
    ),
    _fk(
        ["institution_id", "dataset_id", "student_id"],
        ["dataset_students.institution_id", "dataset_students.dataset_id", "dataset_students.id"],
    ),
    _fk(
        ["institution_id", "dataset_id", "department_code"],
        ["dataset_departments.institution_id", "dataset_departments.dataset_id", "dataset_departments.department_code"],
    ),
    sa.UniqueConstraint("institution_id", "dataset_id", "version_id", "student_id", name="uq_records_student_version"),
    sa.UniqueConstraint("institution_id", "dataset_id", "version_id", "student_id", "id", name="uq_records_reference"),
    sa.CheckConstraint("success_score IS NULL OR success_score BETWEEN 0 AND 100", name="score_range"),
    sa.CheckConstraint(
        "academic_risk_probability IS NULL OR academic_risk_probability BETWEEN 0 AND 1", name="academic_range"
    ),
    sa.CheckConstraint(
        "placement_risk_probability IS NULL OR placement_risk_probability BETWEEN 0 AND 1", name="placement_range"
    ),
    sa.CheckConstraint("priority_score IS NULL OR priority_score >= 0", name="priority_nonnegative"),
)
interventions = sa.Table(
    "interventions",
    metadata,
    _id(primary_key=True),
    _id("institution_id"),
    _id("dataset_id"),
    _id("version_id"),
    _id("student_id"),
    _id("record_id"),
    _id("created_by_user_id"),
    _id("assignee_user_id", nullable=True),
    sa.Column("type", sa.String(255), nullable=False),
    sa.Column("start_date", sa.Date(), nullable=False),
    sa.Column("before_snapshot", JSON_TYPE, nullable=False),
    _json("legacy_metadata"),
    _created(),
    _fk(
        ["institution_id", "dataset_id", "version_id", "student_id", "record_id"],
        [
            "student_records.institution_id",
            "student_records.dataset_id",
            "student_records.version_id",
            "student_records.student_id",
            "student_records.id",
        ],
    ),
    _fk(["institution_id", "created_by_user_id"], ["memberships.institution_id", "memberships.user_id"]),
    _fk(["institution_id", "assignee_user_id"], ["memberships.institution_id", "memberships.user_id"]),
    sa.CheckConstraint("length(trim(type)) > 0", name="type_present"),
)
user_dataset_selections = sa.Table(
    "user_dataset_selections",
    metadata,
    _id("institution_id", primary_key=True),
    _id("user_id", primary_key=True),
    _id("dataset_id", nullable=True),
    _id("version_id", nullable=True),
    sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
    _fk(["institution_id", "user_id"], ["memberships.institution_id", "memberships.user_id"]),
    _fk(
        ["institution_id", "dataset_id", "version_id"],
        ["dataset_versions.institution_id", "dataset_versions.dataset_id", "dataset_versions.id"],
    ),
    sa.CheckConstraint(
        "(dataset_id IS NULL AND version_id IS NULL) OR (dataset_id IS NOT NULL AND version_id IS NOT NULL)",
        name="complete_selection",
    ),
    sa.CheckConstraint("revision >= 0", name="revision_nonnegative"),
)
sa.Index("ix_datasets_owner", datasets.c.institution_id, datasets.c.owner_user_id)
sa.Index(
    "ix_records_worklist",
    student_records.c.institution_id,
    student_records.c.dataset_id,
    student_records.c.version_id,
    student_records.c.priority_score,
)
sa.Index(
    "ix_interventions_student", interventions.c.institution_id, interventions.c.dataset_id, interventions.c.student_id
)
