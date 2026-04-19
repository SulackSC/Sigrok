"""drop bets table and users.num_bets column

Revision ID: e1f4a9c83d27
Revises: b7c4e2a1d903
Create Date: 2026-04-19

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "e1f4a9c83d27"
down_revision: Union[str, None] = "b7c4e2a1d903"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("bets")
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("num_bets")


def downgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("num_bets", sa.Integer(), nullable=False, server_default="0"))
    op.create_table(
        "bets",
        sa.Column("guild_id", sa.Integer, index=True),
        sa.Column("message_id", sa.Integer, primary_key=True, index=True),
        sa.Column(
            "timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.Column("user_id_1", sa.Integer, index=True),
        sa.Column("user_id_2", sa.Integer, index=True),
    )
