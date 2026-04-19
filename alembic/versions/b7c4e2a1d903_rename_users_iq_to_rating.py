"""rename users.iq to users.rating

Revision ID: b7c4e2a1d903
Revises: f55be95078ca
Create Date: 2026-04-19

"""
from typing import Sequence, Union

from alembic import op

revision: str = "b7c4e2a1d903"
down_revision: Union[str, None] = "f55be95078ca"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.alter_column("iq", new_column_name="rating")


def downgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.alter_column("rating", new_column_name="iq")
