"""Add encrypted AI Pass OAuth connections and short-lived PKCE transactions.

Revision ID: 0047_aipass_oauth
Revises: 0046_case_stale_slug_guard
Create Date: 2026-07-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0047_aipass_oauth"
down_revision: str | None = "0046_case_stale_slug_guard"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "aipass_connections",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("workspace_id", sa.String(length=32), nullable=False),
        sa.Column(
            "connected_by_user_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("subject_hash", sa.String(length=64), nullable=False),
        sa.Column("access_token_encrypted", sa.LargeBinary(), nullable=False),
        sa.Column("refresh_token_encrypted", sa.LargeBinary(), nullable=False),
        sa.Column("token_type", sa.String(length=32), nullable=False),
        sa.Column("scope", sa.String(length=512), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "refresh_generation",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("last_refreshed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["connected_by_user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "workspace_id",
            name="uq_aipass_connections_workspace",
        ),
    )
    op.create_index(
        "ix_aipass_connections_expires",
        "aipass_connections",
        ["expires_at"],
        unique=False,
    )

    op.create_table(
        "aipass_oauth_transactions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("workspace_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("code_verifier_encrypted", sa.LargeBinary(), nullable=False),
        sa.Column("redirect_uri", sa.String(length=500), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "state_hash",
            name="uq_aipass_oauth_transactions_state_hash",
        ),
    )
    op.create_index(
        "ix_aipass_oauth_transactions_expiry",
        "aipass_oauth_transactions",
        ["expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_aipass_oauth_transactions_expiry",
        table_name="aipass_oauth_transactions",
    )
    op.drop_table("aipass_oauth_transactions")
    op.drop_index(
        "ix_aipass_connections_expires",
        table_name="aipass_connections",
    )
    op.drop_table("aipass_connections")
