"""Server-side AI Pass OAuth connection and short-lived PKCE transaction models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from suitest_core.crypto import EncryptedBytes

from suitest_db.base import Base, TimestampMixin
from suitest_db.ids import new_id


class AiPassConnection(Base, TimestampMixin):
    """One workspace-owned AI Pass wallet connection.

    Access and refresh tokens are decrypted only inside the API/runner process
    by :class:`EncryptedBytes`; no API schema exposes these fields.
    """

    __tablename__ = "aipass_connections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    connected_by_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    subject_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    access_token_encrypted: Mapped[str] = mapped_column(EncryptedBytes, nullable=False)
    refresh_token_encrypted: Mapped[str] = mapped_column(EncryptedBytes, nullable=False)
    token_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Bearer")
    scope: Mapped[str] = mapped_column(String(512), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    refresh_generation: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    last_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("workspace_id", name="uq_aipass_connections_workspace"),
        Index("ix_aipass_connections_expires", "expires_at"),
    )


class AiPassOAuthTransaction(Base):
    """Single-use server-side PKCE verifier and hashed OAuth state."""

    __tablename__ = "aipass_oauth_transactions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    code_verifier_encrypted: Mapped[str] = mapped_column(EncryptedBytes, nullable=False)
    redirect_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("state_hash", name="uq_aipass_oauth_transactions_state_hash"),
        Index("ix_aipass_oauth_transactions_expiry", "expires_at"),
    )
