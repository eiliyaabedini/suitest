"""Optional server-side AI Pass account connection (OAuth2 + PKCE S256)."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession
from suitest_core.capabilities import TierFlag
from suitest_db.models.user import User
from suitest_shared.domain.enums import Role

from suitest_api.auth.db import get_async_session
from suitest_api.auth.manager import current_active_user
from suitest_api.deps.role import require_role
from suitest_api.deps.scope import TenantContext, require_workspace_membership
from suitest_api.deps.tier import require_tier
from suitest_api.services.aipass_oauth_service import (
    AiPassOAuthError,
    AiPassOAuthService,
    resolve_callback_context,
)
from suitest_api.settings import Settings

router = APIRouter(prefix="/api/v1", tags=["aipass"])

_ADMIN_ROLES = {Role.ADMIN, Role.OWNER}
_OAUTH_REDIRECT_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "Referrer-Policy": "no-referrer",
}


class AiPassConnectionPublic(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    configured: bool
    connected: bool
    active: bool
    active_model: str | None = Field(default=None, alias="activeModel")


class AiPassModelsPublic(BaseModel):
    models: list[dict[str, str]]


class AiPassActivateBody(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class AiPassDisconnectPublic(BaseModel):
    revoked: bool


def _runtime(request: Request) -> tuple[Settings, httpx.AsyncClient | None]:
    settings = request.app.state.settings
    if not isinstance(settings, Settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "AIPASS_NOT_CONFIGURED", "message": "AI Pass is unavailable."},
        )
    candidate = getattr(request.app.state, "aipass_http_client", None)
    client = candidate if isinstance(candidate, httpx.AsyncClient) else None
    return settings, client


def _service(
    request: Request,
    session: AsyncSession,
    ctx: TenantContext,
) -> AiPassOAuthService:
    settings, client = _runtime(request)
    return AiPassOAuthService(session, ctx, settings, http_client=client)


def _http_error(exc: AiPassOAuthError) -> HTTPException:
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    if exc.code == "AIPASS_NOT_CONFIGURED":
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif exc.code == "AIPASS_ALREADY_CONNECTED":
        status_code = status.HTTP_409_CONFLICT
    elif exc.code == "AIPASS_NOT_CONNECTED":
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code in {
        "AIPASS_TIMEOUT",
        "AIPASS_UNAVAILABLE",
        "AIPASS_UPSTREAM_ERROR",
        "AIPASS_INVALID_RESPONSE",
    }:
        status_code = status.HTTP_502_BAD_GATEWAY
    elif exc.code == "AIPASS_STATE_INVALID":
        status_code = status.HTTP_400_BAD_REQUEST
    return HTTPException(
        status_code=status_code,
        detail={"code": exc.code, "message": exc.message},
    )


def _settings_redirect(settings: Settings, **query: str) -> str:
    return f"{settings.web_url.rstrip('/')}/settings?{urlencode({'tab': 'llm', **query})}"


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(
        url=url,
        status_code=status.HTTP_303_SEE_OTHER,
        headers=_OAUTH_REDIRECT_HEADERS,
    )


@router.get(
    "/workspaces/{workspaceId}/aipass/connection",
    response_model=AiPassConnectionPublic,
)
@require_tier(TierFlag.ANY)
async def get_aipass_connection(
    request: Request,
    ctx: TenantContext = Depends(require_workspace_membership),
    session: AsyncSession = Depends(get_async_session),
) -> AiPassConnectionPublic:
    configured, connected, active_model = await _service(request, session, ctx).connection()
    return AiPassConnectionPublic(
        configured=configured,
        connected=connected,
        active=active_model is not None,
        active_model=active_model,
    )


@router.post("/workspaces/{workspaceId}/aipass/authorize")
@require_tier(TierFlag.ANY)
async def authorize_aipass(
    request: Request,
    ctx: TenantContext = Depends(require_role(_ADMIN_ROLES)),
    session: AsyncSession = Depends(get_async_session),
) -> RedirectResponse:
    """Start a same-site form POST, then redirect to AI Pass authorization."""
    try:
        url = await _service(request, session, ctx).start_authorization()
    except AiPassOAuthError as exc:
        raise _http_error(exc) from exc
    return _redirect(url)


@router.get("/aipass/callback")
@require_tier(TierFlag.ANY)
async def callback_aipass(
    request: Request,
    state_value: Annotated[
        str,
        Query(alias="state", min_length=32, max_length=256),
    ],
    code: Annotated[str | None, Query(min_length=1, max_length=4096)] = None,
    oauth_error: Annotated[
        str | None,
        Query(alias="error", min_length=1, max_length=128),
    ] = None,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> RedirectResponse:
    """Consume the single-use state and exchange the code entirely server-side."""
    settings, _ = _runtime(request)
    try:
        ctx = await resolve_callback_context(
            session,
            user_id=user.id,
            state=state_value,
        )
    except AiPassOAuthError as exc:
        return _redirect(_settings_redirect(settings, aipass_error=exc.code))
    service = _service(request, session, ctx)
    if oauth_error is not None:
        try:
            await service.consume_callback_state(state_value)
        except AiPassOAuthError as exc:
            return _redirect(_settings_redirect(settings, aipass_error=exc.code))
        return _redirect(_settings_redirect(settings, aipass_error="AIPASS_ACCESS_DENIED"))
    if code is None:
        try:
            await service.consume_callback_state(state_value)
        except AiPassOAuthError as exc:
            return _redirect(_settings_redirect(settings, aipass_error=exc.code))
        return _redirect(_settings_redirect(settings, aipass_error="AIPASS_CODE_MISSING"))
    try:
        await service.complete_callback(state=state_value, code=code)
    except AiPassOAuthError as exc:
        return _redirect(_settings_redirect(settings, aipass_error=exc.code))
    return _redirect(_settings_redirect(settings, aipass="connected"))


@router.get(
    "/workspaces/{workspaceId}/aipass/models",
    response_model=AiPassModelsPublic,
)
@require_tier(TierFlag.ANY)
async def list_aipass_models(
    request: Request,
    ctx: TenantContext = Depends(require_workspace_membership),
    session: AsyncSession = Depends(get_async_session),
) -> AiPassModelsPublic:
    try:
        models = await _service(request, session, ctx).models()
    except AiPassOAuthError as exc:
        raise _http_error(exc) from exc
    return AiPassModelsPublic(models=[model.model_dump() for model in models])


@router.put(
    "/workspaces/{workspaceId}/aipass/connection",
    response_model=AiPassConnectionPublic,
)
@require_tier(TierFlag.ANY)
async def activate_aipass(
    body: AiPassActivateBody,
    request: Request,
    ctx: TenantContext = Depends(require_role(_ADMIN_ROLES)),
    session: AsyncSession = Depends(get_async_session),
) -> AiPassConnectionPublic:
    service = _service(request, session, ctx)
    try:
        row = await service.activate(body.model)
    except AiPassOAuthError as exc:
        raise _http_error(exc) from exc
    return AiPassConnectionPublic(
        configured=True,
        connected=True,
        active=True,
        active_model=row.model,
    )


@router.delete(
    "/workspaces/{workspaceId}/aipass/connection",
    response_model=AiPassDisconnectPublic,
)
@require_tier(TierFlag.ANY)
async def disconnect_aipass(
    request: Request,
    ctx: TenantContext = Depends(require_role(_ADMIN_ROLES)),
    session: AsyncSession = Depends(get_async_session),
) -> AiPassDisconnectPublic:
    try:
        revoked = await _service(request, session, ctx).disconnect()
    except AiPassOAuthError as exc:
        raise _http_error(exc) from exc
    return AiPassDisconnectPublic(revoked=revoked)
