"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
"""

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade():
    """Apply this reviewed schema revision explicitly."""
    ${upgrades if upgrades else "pass"}


def downgrade():
    """Do not discard data through an automatic downgrade."""
    raise RuntimeError("Destructive downgrade is disabled; use an approved restore plan.")
