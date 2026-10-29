"""Create the fresh Gravity foundation; never import or transform legacy data.

This revision is frozen: it must not import the mutable application metadata.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_identity_and_datasets"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create empty constrained tables in an explicitly approved fresh target."""
    op.create_table(
        "institutions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_institutions")),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("username_key", sa.String(length=200), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("token_version", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("length(password_hash) > 0", name=op.f("ck_users_password_hash_present")),
        sa.CheckConstraint("length(trim(username_key)) > 0", name=op.f("ck_users_username_present")),
        sa.CheckConstraint("token_version >= 0", name=op.f("ck_users_token_version_nonnegative")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("username_key", name=op.f("uq_users_username_key")),
    )
    op.create_table(
        "departments",
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.CheckConstraint("length(trim(code)) > 0", name=op.f("ck_departments_code_present")),
        sa.ForeignKeyConstraint(
            ["institution_id"],
            ["institutions.id"],
            name=op.f("fk_departments_institution_id_institutions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("institution_id", "code", name=op.f("pk_departments")),
    )
    op.create_table(
        "memberships",
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.CheckConstraint("role IN ('admin', 'faculty', 'viewer')", name=op.f("ck_memberships_valid_role")),
        sa.ForeignKeyConstraint(
            ["institution_id"],
            ["institutions.id"],
            name=op.f("fk_memberships_institution_id_institutions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_memberships_user_id_users"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("institution_id", "user_id", name=op.f("pk_memberships")),
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("archived", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("revision", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("kind IN ('upload', 'individual', 'demo')", name=op.f("ck_datasets_valid_kind")),
        sa.CheckConstraint("length(trim(label)) > 0", name=op.f("ck_datasets_label_present")),
        sa.CheckConstraint("revision >= 0", name=op.f("ck_datasets_revision_nonnegative")),
        sa.ForeignKeyConstraint(
            ["institution_id", "owner_user_id"],
            ["memberships.institution_id", "memberships.user_id"],
            name=op.f("fk_datasets_institution_id_owner_user_id_memberships"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_datasets")),
        sa.UniqueConstraint("institution_id", "id", name="uq_datasets_scope"),
    )
    op.create_index("ix_datasets_owner", "datasets", ["institution_id", "owner_user_id"], unique=False)
    op.create_table(
        "membership_departments",
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("department_code", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["institution_id", "department_code"],
            ["departments.institution_id", "departments.code"],
            name=op.f("fk_membership_departments_institution_id_department_code_departments"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "user_id"],
            ["memberships.institution_id", "memberships.user_id"],
            name=op.f("fk_membership_departments_institution_id_user_id_memberships"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("institution_id", "user_id", "department_code", name=op.f("pk_membership_departments")),
    )
    op.create_table(
        "dataset_departments",
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("department_code", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id"],
            ["datasets.institution_id", "datasets.id"],
            name=op.f("fk_dataset_departments_institution_id_dataset_id_datasets"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "department_code"],
            ["departments.institution_id", "departments.code"],
            name=op.f("fk_dataset_departments_institution_id_department_code_departments"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("institution_id", "dataset_id", "department_code", name=op.f("pk_dataset_departments")),
    )
    op.create_table(
        "dataset_grants",
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("granted_by_user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id"],
            ["datasets.institution_id", "datasets.id"],
            name=op.f("fk_dataset_grants_institution_id_dataset_id_datasets"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "granted_by_user_id"],
            ["memberships.institution_id", "memberships.user_id"],
            name=op.f("fk_dataset_grants_institution_id_granted_by_user_id_memberships"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "user_id"],
            ["memberships.institution_id", "memberships.user_id"],
            name=op.f("fk_dataset_grants_institution_id_user_id_memberships"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("institution_id", "dataset_id", "user_id", name=op.f("pk_dataset_grants")),
    )
    op.create_table(
        "dataset_students",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=False),
        sa.Column("external_id_key", sa.String(length=128), nullable=False),
        sa.CheckConstraint(
            "length(trim(external_id)) > 0 AND length(trim(external_id_key)) > 0",
            name=op.f("ck_dataset_students_external_id_present"),
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id"],
            ["datasets.institution_id", "datasets.id"],
            name=op.f("fk_dataset_students_institution_id_dataset_id_datasets"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_students")),
        sa.UniqueConstraint("institution_id", "dataset_id", "external_id_key", name="uq_students_external_key"),
        sa.UniqueConstraint("institution_id", "dataset_id", "id", name="uq_students_scope"),
    )
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("parent_version_id", sa.Uuid(), nullable=True),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("student_count", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "provenance",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "quality",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "segments",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint(
            "parent_version_id IS NULL OR parent_version_id <> id", name=op.f("ck_dataset_versions_not_own_parent")
        ),
        sa.CheckConstraint("student_count >= 0", name=op.f("ck_dataset_versions_student_count_nonnegative")),
        sa.CheckConstraint("version_number > 0", name=op.f("ck_dataset_versions_positive_version")),
        sa.ForeignKeyConstraint(
            ["institution_id", "created_by_user_id"],
            ["memberships.institution_id", "memberships.user_id"],
            name=op.f("fk_dataset_versions_institution_id_created_by_user_id_memberships"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id", "parent_version_id"],
            ["dataset_versions.institution_id", "dataset_versions.dataset_id", "dataset_versions.id"],
            name=op.f("fk_dataset_versions_institution_id_dataset_id_parent_version_id_dataset_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id"],
            ["datasets.institution_id", "datasets.id"],
            name=op.f("fk_dataset_versions_institution_id_dataset_id_datasets"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_versions")),
        sa.UniqueConstraint("institution_id", "dataset_id", "id", name="uq_versions_scope"),
        sa.UniqueConstraint("institution_id", "dataset_id", "version_number", name="uq_versions_number"),
    )
    op.create_table(
        "student_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("student_id", sa.Uuid(), nullable=False),
        sa.Column("department_code", sa.String(length=64), nullable=False),
        sa.Column("snapshot_month", sa.Date(), nullable=False),
        sa.Column("success_score", sa.Float(), nullable=True),
        sa.Column("academic_risk_probability", sa.Float(), nullable=True),
        sa.Column("placement_risk_probability", sa.Float(), nullable=True),
        sa.Column("priority_score", sa.Float(), nullable=True),
        sa.Column(
            "raw",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "derived",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "drivers",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "actions",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "academic_risk_probability IS NULL OR academic_risk_probability BETWEEN 0 AND 1",
            name=op.f("ck_student_records_academic_range"),
        ),
        sa.CheckConstraint(
            "placement_risk_probability IS NULL OR placement_risk_probability BETWEEN 0 AND 1",
            name=op.f("ck_student_records_placement_range"),
        ),
        sa.CheckConstraint(
            "priority_score IS NULL OR priority_score >= 0", name=op.f("ck_student_records_priority_nonnegative")
        ),
        sa.CheckConstraint(
            "success_score IS NULL OR success_score BETWEEN 0 AND 100", name=op.f("ck_student_records_score_range")
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id", "department_code"],
            [
                "dataset_departments.institution_id",
                "dataset_departments.dataset_id",
                "dataset_departments.department_code",
            ],
            name=op.f("fk_student_records_institution_id_dataset_id_department_code_dataset_departments"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id", "student_id"],
            ["dataset_students.institution_id", "dataset_students.dataset_id", "dataset_students.id"],
            name=op.f("fk_student_records_institution_id_dataset_id_student_id_dataset_students"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id", "version_id"],
            ["dataset_versions.institution_id", "dataset_versions.dataset_id", "dataset_versions.id"],
            name=op.f("fk_student_records_institution_id_dataset_id_version_id_dataset_versions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_student_records")),
        sa.UniqueConstraint(
            "institution_id", "dataset_id", "version_id", "student_id", "id", name="uq_records_reference"
        ),
        sa.UniqueConstraint(
            "institution_id", "dataset_id", "version_id", "student_id", name="uq_records_student_version"
        ),
    )
    op.create_index(
        "ix_records_worklist",
        "student_records",
        ["institution_id", "dataset_id", "version_id", "priority_score"],
        unique=False,
    )
    op.create_table(
        "user_dataset_selections",
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=True),
        sa.Column("version_id", sa.Uuid(), nullable=True),
        sa.Column("revision", sa.Integer(), server_default="0", nullable=False),
        sa.CheckConstraint(
            "(dataset_id IS NULL AND version_id IS NULL) OR (dataset_id IS NOT NULL AND version_id IS NOT NULL)",
            name=op.f("ck_user_dataset_selections_complete_selection"),
        ),
        sa.CheckConstraint("revision >= 0", name=op.f("ck_user_dataset_selections_revision_nonnegative")),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id", "version_id"],
            ["dataset_versions.institution_id", "dataset_versions.dataset_id", "dataset_versions.id"],
            name=op.f("fk_user_dataset_selections_institution_id_dataset_id_version_id_dataset_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "user_id"],
            ["memberships.institution_id", "memberships.user_id"],
            name=op.f("fk_user_dataset_selections_institution_id_user_id_memberships"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("institution_id", "user_id", name=op.f("pk_user_dataset_selections")),
    )
    op.create_table(
        "interventions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("institution_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("student_id", sa.Uuid(), nullable=False),
        sa.Column("record_id", sa.Uuid(), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("assignee_user_id", sa.Uuid(), nullable=True),
        sa.Column("type", sa.String(length=255), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column(
            "before_snapshot",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "legacy_metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("length(trim(type)) > 0", name=op.f("ck_interventions_type_present")),
        sa.ForeignKeyConstraint(
            ["institution_id", "assignee_user_id"],
            ["memberships.institution_id", "memberships.user_id"],
            name=op.f("fk_interventions_institution_id_assignee_user_id_memberships"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "created_by_user_id"],
            ["memberships.institution_id", "memberships.user_id"],
            name=op.f("fk_interventions_institution_id_created_by_user_id_memberships"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["institution_id", "dataset_id", "version_id", "student_id", "record_id"],
            [
                "student_records.institution_id",
                "student_records.dataset_id",
                "student_records.version_id",
                "student_records.student_id",
                "student_records.id",
            ],
            name=op.f("fk_interventions_institution_id_dataset_id_version_id_student_id_record_id_student_records"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_interventions")),
    )
    op.create_index(
        "ix_interventions_student", "interventions", ["institution_id", "dataset_id", "student_id"], unique=False
    )


def downgrade():
    """Do not silently discard identity or student history."""
    raise RuntimeError("Destructive downgrade is disabled; use an approved restore plan.")
