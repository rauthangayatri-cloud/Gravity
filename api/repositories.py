"""Institution-scoped storage primitives, not a replacement for Phase 2 authorization."""

from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.engine import Connection, RowMapping

from api import models


class DatasetRepository:
    """Require explicit institution scope; callers must apply the approved role policy."""

    def __init__(self, connection: Connection, institution_id: UUID):
        if not isinstance(institution_id, UUID):
            raise ValueError("An explicit institution UUID is required.")
        self.connection = connection
        self.institution_id = institution_id

    def dataset(self, dataset_id: UUID) -> RowMapping | None:
        """Read dataset metadata only within the supplied institution."""
        table = models.datasets
        return (
            self.connection.execute(
                sa.select(table).where(table.c.institution_id == self.institution_id, table.c.id == dataset_id)
            )
            .mappings()
            .one_or_none()
        )

    def version(self, dataset_id: UUID, version_id: UUID) -> RowMapping | None:
        """Pin a dataset version instead of re-resolving any global active state."""
        table = models.dataset_versions
        return (
            self.connection.execute(
                sa.select(table).where(
                    table.c.institution_id == self.institution_id,
                    table.c.dataset_id == dataset_id,
                    table.c.id == version_id,
                )
            )
            .mappings()
            .one_or_none()
        )

    def selection(self, user_id: UUID) -> RowMapping | None:
        """Read a user/institution selection; possession never implies authorization."""
        table = models.user_dataset_selections
        return (
            self.connection.execute(
                sa.select(table).where(table.c.institution_id == self.institution_id, table.c.user_id == user_id)
            )
            .mappings()
            .one_or_none()
        )

    def student_identity(self, dataset_id: UUID, external_id: str) -> RowMapping | None:
        """Look up a dataset-local key without merging identities across datasets."""
        key = self._student_key(external_id)
        table = models.dataset_students
        return (
            self.connection.execute(
                sa.select(table).where(
                    table.c.institution_id == self.institution_id,
                    table.c.dataset_id == dataset_id,
                    table.c.external_id_key == key,
                )
            )
            .mappings()
            .one_or_none()
        )

    def add_student_identity(self, dataset_id: UUID, external_id: str) -> UUID:
        """Insert identity only; the caller owns authorization and transaction commit."""
        key = self._student_key(external_id)
        return self.connection.execute(
            models.dataset_students.insert()
            .values(
                institution_id=self.institution_id,
                dataset_id=dataset_id,
                external_id=external_id.strip(),
                external_id_key=key,
            )
            .returning(models.dataset_students.c.id)
        ).scalar_one()

    def record(self, dataset_id: UUID, version_id: UUID, record_id: UUID) -> RowMapping | None:
        """Read one record using the full institution/dataset/version boundary."""
        table = models.student_records
        return (
            self.connection.execute(
                sa.select(table).where(
                    table.c.institution_id == self.institution_id,
                    table.c.dataset_id == dataset_id,
                    table.c.version_id == version_id,
                    table.c.id == record_id,
                )
            )
            .mappings()
            .one_or_none()
        )

    def intervention(self, dataset_id: UUID, intervention_id: UUID) -> RowMapping | None:
        """Read history in its dataset scope, never by global intervention ID alone."""
        table = models.interventions
        return (
            self.connection.execute(
                sa.select(table).where(
                    table.c.institution_id == self.institution_id,
                    table.c.dataset_id == dataset_id,
                    table.c.id == intervention_id,
                )
            )
            .mappings()
            .one_or_none()
        )

    @staticmethod
    def _student_key(external_id: str) -> str:
        """Normalize only lookup keys; do not coerce numeric IDs or lose leading zeros."""
        if not isinstance(external_id, str) or not external_id.strip():
            raise ValueError("A nonempty textual student ID is required.")
        key = external_id.strip().casefold()
        if max(len(external_id.strip()), len(key)) > 128:
            raise ValueError("Student IDs must not exceed 128 characters.")
        return key
