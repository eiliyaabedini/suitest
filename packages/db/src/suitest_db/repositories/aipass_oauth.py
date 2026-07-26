"""Repository operations for server-side AI Pass OAuth state and tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from suitest_db.models.aipass_oauth import AiPassConnection, AiPassOAuthTransaction

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession


class AiPassConnectionRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, workspace_id: str) -> AiPassConnection | None:
        result: AiPassConnection | None = await self._session.scalar(
            select(AiPassConnection).where(AiPassConnection.workspace_id == workspace_id)
        )
        return result

    async def get_for_update(self, workspace_id: str) -> AiPassConnection | None:
        result: AiPassConnection | None = await self._session.scalar(
            select(AiPassConnection)
            .where(AiPassConnection.workspace_id == workspace_id)
            .with_for_update()
        )
        return result

    async def create(
        self,
        *,
        workspace_id: str,
        connected_by_user_id: uuid.UUID,
        subject_hash: str,
        access_token: str,
        refresh_token: str,
        token_type: str,
        scope: str,
        expires_at: datetime,
    ) -> AiPassConnection:
        row = AiPassConnection(
            workspace_id=workspace_id,
            connected_by_user_id=connected_by_user_id,
            subject_hash=subject_hash,
            access_token_encrypted=access_token,
            refresh_token_encrypted=refresh_token,
            token_type=token_type,
            scope=scope,
            expires_at=expires_at,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def delete(self, workspace_id: str) -> None:
        await self._session.execute(
            delete(AiPassConnection).where(AiPassConnection.workspace_id == workspace_id)
        )


class AiPassOAuthTransactionRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        workspace_id: str,
        user_id: uuid.UUID,
        state_hash: str,
        code_verifier: str,
        redirect_uri: str,
        expires_at: datetime,
        created_at: datetime,
    ) -> AiPassOAuthTransaction:
        await self._session.execute(
            delete(AiPassOAuthTransaction).where(
                AiPassOAuthTransaction.workspace_id == workspace_id,
                AiPassOAuthTransaction.user_id == user_id,
            )
        )
        row = AiPassOAuthTransaction(
            workspace_id=workspace_id,
            user_id=user_id,
            state_hash=state_hash,
            code_verifier_encrypted=code_verifier,
            redirect_uri=redirect_uri,
            expires_at=expires_at,
            created_at=created_at,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def consume(
        self,
        *,
        workspace_id: str,
        user_id: uuid.UUID,
        state_hash: str,
        now: datetime,
    ) -> AiPassOAuthTransaction | None:
        row = await self._session.scalar(
            select(AiPassOAuthTransaction)
            .where(
                AiPassOAuthTransaction.workspace_id == workspace_id,
                AiPassOAuthTransaction.user_id == user_id,
                AiPassOAuthTransaction.state_hash == state_hash,
                AiPassOAuthTransaction.expires_at > now,
            )
            .with_for_update()
        )
        if row is None:
            return None
        await self._session.delete(row)
        await self._session.flush()
        return row

    async def find_for_user(
        self,
        *,
        user_id: uuid.UUID,
        state_hash: str,
        now: datetime,
    ) -> AiPassOAuthTransaction | None:
        row: AiPassOAuthTransaction | None = await self._session.scalar(
            select(AiPassOAuthTransaction).where(
                AiPassOAuthTransaction.user_id == user_id,
                AiPassOAuthTransaction.state_hash == state_hash,
                AiPassOAuthTransaction.expires_at > now,
            )
        )
        return row
