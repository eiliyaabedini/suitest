"""Server-side AI Pass OAuth2 + PKCE lifecycle.

Only this service handles authorization codes, access tokens, refresh tokens,
and the public client identifier.  Browser-facing schemas expose connection
status and live model metadata only.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode, urlsplit

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from suitest_agent.providers.base import LLMProvider, ProviderError
from suitest_agent.providers.litellm_router import get_provider
from suitest_db.audit import write_audit
from suitest_db.models.llm_config import LLMConfig
from suitest_db.models.tenancy import Membership
from suitest_db.repositories.aipass_oauth import (
    AiPassConnectionRepo,
    AiPassOAuthTransactionRepo,
)
from suitest_db.repositories.llm_configs import LLMConfigRepo
from suitest_shared.domain.enums import Role

from suitest_api.deps.scope import TenantContext
from suitest_api.services.llm_config_service import LLMConfigService
from suitest_api.settings import Settings, get_settings

AIPASS_ISSUER = "https://aipass.one"
AIPASS_DISCOVERY_URL = f"{AIPASS_ISSUER}/.well-known/oauth-authorization-server"
AIPASS_MODELS_URL = f"{AIPASS_ISSUER}/oauth2/v1/models?detailed=true"
AIPASS_PROVIDER = "aipass"

_SCOPES = ("api:access", "profile:read")
_PKCE_TTL = timedelta(minutes=10)
_TOKEN_EXPIRY_SKEW = timedelta(seconds=60)
_MAX_OAUTH_RESPONSE_BYTES = 512 * 1024
_MAX_MODELS_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_MODELS = 500
_HTTP_TOTAL_TIMEOUT_SECONDS = 30.0


class AiPassOAuthError(RuntimeError):
    """Safe, bounded error suitable for an API error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AiPassModel(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)


class _OAuthMetadata(BaseModel):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    revocation_endpoint: str
    scopes_supported: list[str]
    response_types_supported: list[str]
    grant_types_supported: list[str]
    code_challenge_methods_supported: list[str]
    token_endpoint_auth_methods_supported: list[str]


class _TokenResponse(BaseModel):
    access_token: SecretStr = Field(min_length=1, max_length=16_384)
    refresh_token: SecretStr | None = Field(default=None, min_length=1, max_length=16_384)
    token_type: str = Field(min_length=1, max_length=32)
    expires_in: int = Field(gt=0, le=60 * 60 * 24 * 30)
    scope: str | None = Field(default=None, max_length=512)


def code_challenge_s256(verifier: str) -> str:
    """Return the RFC 7636 S256 challenge for ``verifier``."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _state_hash(state: str) -> str:
    return hashlib.sha256(state.encode()).hexdigest()


def _object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return {str(key): item for key, item in value.items()}


def parse_models_payload(payload: object) -> list[AiPassModel]:
    """Normalize OpenAI ``{object:"list",data:[...]}`` and legacy arrays."""
    root = _object_dict(payload)
    raw_items: list[object]
    if root is not None:
        data = root.get("data")
        if root.get("object") != "list" or not isinstance(data, list):
            raise AiPassOAuthError(
                "AIPASS_INVALID_MODELS",
                "AI Pass returned an invalid model catalog.",
            )
        raw_items = list(data)
    elif isinstance(payload, list):
        raw_items = list(payload)
    else:
        raise AiPassOAuthError(
            "AIPASS_INVALID_MODELS",
            "AI Pass returned an invalid model catalog.",
        )

    models: list[AiPassModel] = []
    seen: set[str] = set()
    for item in raw_items[:_MAX_MODELS]:
        if isinstance(item, str):
            model_id = item.strip()
            name = model_id
        else:
            model = _object_dict(item)
            if model is None:
                continue
            raw_id = model.get("id")
            if not isinstance(raw_id, str):
                continue
            model_id = raw_id.strip()
            raw_name = model.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else model_id
            methods = model.get("methods")
            if isinstance(methods, list):
                method_names = {method for method in methods if isinstance(method, str)}
                if method_names and "chat_completions" not in method_names:
                    continue
        if not model_id or len(model_id) > 200 or not name or len(name) > 300:
            continue
        if model_id in seen:
            continue
        seen.add(model_id)
        models.append(AiPassModel(id=model_id, name=name))
    return models


def _validate_aipass_endpoint(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname != "aipass.one":
        raise AiPassOAuthError(
            "AIPASS_DISCOVERY_INVALID",
            "AI Pass discovery returned an untrusted endpoint.",
        )
    if parsed.username or parsed.password or parsed.fragment:
        raise AiPassOAuthError(
            "AIPASS_DISCOVERY_INVALID",
            "AI Pass discovery returned an untrusted endpoint.",
        )


class AiPassHttpClient:
    """Bounded HTTP client for discovery, OAuth, userinfo, revoke, and models."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._http_client = http_client

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            yield client

    @staticmethod
    async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > limit:
                raise AiPassOAuthError(
                    "AIPASS_RESPONSE_TOO_LARGE",
                    "AI Pass returned a response larger than the configured limit.",
                )
            chunks.append(chunk)
        return b"".join(chunks)

    async def _json_request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
        max_bytes: int = _MAX_OAUTH_RESPONSE_BYTES,
    ) -> object:
        try:
            async with asyncio.timeout(_HTTP_TOTAL_TIMEOUT_SECONDS):
                async with self._client() as client:
                    async with client.stream(
                        method,
                        url,
                        headers=headers,
                        data=data,
                    ) as response:
                        if response.status_code < 200 or response.status_code >= 300:
                            raise AiPassOAuthError(
                                "AIPASS_UPSTREAM_ERROR",
                                "AI Pass rejected the request.",
                            )
                        raw = await self._read_bounded(response, max_bytes)
        except TimeoutError as exc:
            raise AiPassOAuthError(
                "AIPASS_TIMEOUT",
                "AI Pass did not respond within the configured time limit.",
            ) from exc
        except httpx.HTTPError as exc:
            raise AiPassOAuthError(
                "AIPASS_UNAVAILABLE",
                "AI Pass is temporarily unreachable.",
            ) from exc
        try:
            loaded: object = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AiPassOAuthError(
                "AIPASS_INVALID_RESPONSE",
                "AI Pass returned an invalid response.",
            ) from exc
        return loaded

    async def discovery(self) -> _OAuthMetadata:
        loaded = await self._json_request("GET", AIPASS_DISCOVERY_URL)
        try:
            metadata = _OAuthMetadata.model_validate(loaded)
        except ValidationError as exc:
            raise AiPassOAuthError(
                "AIPASS_DISCOVERY_INVALID",
                "AI Pass discovery metadata is invalid.",
            ) from exc
        if metadata.issuer != AIPASS_ISSUER:
            raise AiPassOAuthError(
                "AIPASS_DISCOVERY_INVALID",
                "AI Pass discovery metadata has an unexpected issuer.",
            )
        for endpoint in (
            metadata.authorization_endpoint,
            metadata.token_endpoint,
            metadata.userinfo_endpoint,
            metadata.revocation_endpoint,
        ):
            _validate_aipass_endpoint(endpoint)
        if (
            "code" not in metadata.response_types_supported
            or "authorization_code" not in metadata.grant_types_supported
            or "refresh_token" not in metadata.grant_types_supported
            or "S256" not in metadata.code_challenge_methods_supported
            or "none" not in metadata.token_endpoint_auth_methods_supported
        ):
            raise AiPassOAuthError(
                "AIPASS_DISCOVERY_UNSUPPORTED",
                "AI Pass discovery does not support the required public PKCE flow.",
            )
        if not set(_SCOPES).issubset(metadata.scopes_supported):
            raise AiPassOAuthError(
                "AIPASS_DISCOVERY_UNSUPPORTED",
                "AI Pass does not advertise the required scopes.",
            )
        return metadata

    async def exchange_code(
        self,
        *,
        metadata: _OAuthMetadata,
        client_id: str,
        code: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> _TokenResponse:
        loaded = await self._json_request(
            "POST",
            metadata.token_endpoint,
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": redirect_uri,
            },
        )
        return self._parse_token_response(loaded)

    async def refresh_token(
        self,
        *,
        metadata: _OAuthMetadata,
        client_id: str,
        refresh_token: str,
    ) -> _TokenResponse:
        loaded = await self._json_request(
            "POST",
            metadata.token_endpoint,
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": refresh_token,
            },
        )
        return self._parse_token_response(loaded)

    @staticmethod
    def _parse_token_response(loaded: object) -> _TokenResponse:
        try:
            token = _TokenResponse.model_validate(loaded)
        except ValidationError as exc:
            raise AiPassOAuthError(
                "AIPASS_INVALID_TOKEN_RESPONSE",
                "AI Pass returned an invalid token response.",
            ) from exc
        if token.token_type.lower() != "bearer":
            raise AiPassOAuthError(
                "AIPASS_INVALID_TOKEN_RESPONSE",
                "AI Pass returned an unsupported token type.",
            )
        return token

    async def userinfo(self, *, metadata: _OAuthMetadata, access_token: str) -> str:
        loaded = await self._json_request(
            "GET",
            metadata.userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user = _object_dict(loaded)
        if user is None:
            raise AiPassOAuthError(
                "AIPASS_INVALID_USERINFO",
                "AI Pass returned invalid account information.",
            )
        subject = user.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            fallback = user.get("id")
            subject = fallback if isinstance(fallback, str) else ""
        if not subject or len(subject) > 500:
            raise AiPassOAuthError(
                "AIPASS_INVALID_USERINFO",
                "AI Pass returned invalid account information.",
            )
        return subject

    async def revoke(
        self,
        *,
        metadata: _OAuthMetadata,
        client_id: str,
        token: str,
        token_type_hint: str,
    ) -> bool:
        try:
            async with asyncio.timeout(_HTTP_TOTAL_TIMEOUT_SECONDS):
                async with self._client() as client:
                    async with client.stream(
                        "POST",
                        metadata.revocation_endpoint,
                        data={
                            "client_id": client_id,
                            "token": token,
                            "token_type_hint": token_type_hint,
                        },
                    ) as response:
                        return 200 <= response.status_code < 300
        except (TimeoutError, httpx.HTTPError):
            return False

    async def models(self, *, access_token: str) -> list[AiPassModel]:
        loaded = await self._json_request(
            "GET",
            AIPASS_MODELS_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            data=None,
            max_bytes=_MAX_MODELS_RESPONSE_BYTES,
        )
        return parse_models_payload(loaded)


def _client_id(settings: Settings) -> str:
    value = settings.aipass_client_id.get_secret_value().strip()
    if not value:
        raise AiPassOAuthError(
            "AIPASS_NOT_CONFIGURED",
            "AI Pass account connection is not configured for this deployment.",
        )
    return value


def _session_factory(session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    bind = session.bind
    if not isinstance(bind, AsyncEngine):
        raise ProviderError(
            "AIPASS_STORAGE_UNAVAILABLE",
            "AI Pass secure token storage is unavailable.",
        )
    return async_sessionmaker(bind, expire_on_commit=False, class_=AsyncSession)


async def resolve_callback_context(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    state: str,
) -> TenantContext:
    """Resolve a fixed callback to its workspace without exposing it in the URI."""
    transaction = await AiPassOAuthTransactionRepo(session).find_for_user(
        user_id=user_id,
        state_hash=_state_hash(state),
        now=datetime.now(UTC),
    )
    if transaction is None:
        raise AiPassOAuthError(
            "AIPASS_STATE_INVALID",
            "The AI Pass authorization state is invalid or expired.",
        )
    membership = await session.scalar(
        select(Membership).where(
            Membership.workspace_id == transaction.workspace_id,
            Membership.user_id == user_id,
        )
    )
    if membership is None or membership.role not in {Role.ADMIN, Role.OWNER}:
        raise AiPassOAuthError(
            "AIPASS_STATE_INVALID",
            "The AI Pass authorization state is invalid or expired.",
        )
    return TenantContext(
        workspace_id=transaction.workspace_id,
        user_id=str(user_id),
        role=membership.role,
    )


class AiPassTokenManager:
    """Returns valid access tokens and persists refresh rotation atomically."""

    def __init__(
        self,
        *,
        workspace_id: str,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        http: AiPassHttpClient | None = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._session_factory = session_factory
        self._settings = settings
        self._http = http or AiPassHttpClient()
        self._observed_generation: int | None = None

    async def access_token(self, force_refresh: bool) -> str:
        client_id = _client_id(self._settings)
        now = datetime.now(UTC)
        async with self._session_factory() as session, session.begin():
            repo = AiPassConnectionRepo(session)
            connection = await repo.get_for_update(self._workspace_id)
            if connection is None:
                raise ProviderError(
                    "AIPASS_NOT_CONNECTED",
                    "No AI Pass account is connected to this workspace.",
                )
            if (
                force_refresh
                and self._observed_generation is not None
                and connection.refresh_generation != self._observed_generation
            ):
                self._observed_generation = connection.refresh_generation
                return connection.access_token_encrypted
            if not force_refresh and connection.expires_at > now + _TOKEN_EXPIRY_SKEW:
                self._observed_generation = connection.refresh_generation
                return connection.access_token_encrypted

            try:
                metadata = await self._http.discovery()
                rotated = await self._http.refresh_token(
                    metadata=metadata,
                    client_id=client_id,
                    refresh_token=connection.refresh_token_encrypted,
                )
            except AiPassOAuthError as exc:
                raise ProviderError(exc.code, exc.message) from exc
            new_refresh = rotated.refresh_token
            if new_refresh is not None:
                connection.refresh_token_encrypted = new_refresh.get_secret_value()
            connection.access_token_encrypted = rotated.access_token.get_secret_value()
            connection.token_type = rotated.token_type
            connection.scope = rotated.scope or connection.scope
            connection.expires_at = now + timedelta(seconds=rotated.expires_in)
            connection.refresh_generation += 1
            connection.last_refreshed_at = now
            await write_audit(
                session,
                workspace_id=self._workspace_id,
                user_id=str(connection.connected_by_user_id),
                action="aipass.token.refresh",
                resource_type="aipass_connection",
                resource_id=connection.id,
                metadata={"generation": connection.refresh_generation},
            )
            self._observed_generation = connection.refresh_generation
            return connection.access_token_encrypted


def build_workspace_llm_provider(
    session: AsyncSession,
    *,
    workspace_id: str,
    provider: str,
    api_key: str | None,
    base_url: str | None,
    settings: Settings | None = None,
    aipass_http_client: httpx.AsyncClient | None = None,
) -> LLMProvider:
    """Build any existing provider, adding OAuth token management only for AI Pass."""
    if provider.strip().lower() != AIPASS_PROVIDER:
        return get_provider(provider, api_key=api_key, base_url=base_url)
    manager = AiPassTokenManager(
        workspace_id=workspace_id,
        session_factory=_session_factory(session),
        settings=settings or get_settings(),
        http=AiPassHttpClient(aipass_http_client),
    )
    return get_provider(
        provider,
        oauth_access_token_provider=manager.access_token,
    )


class AiPassOAuthService:
    def __init__(
        self,
        session: AsyncSession,
        ctx: TenantContext,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._session = session
        self._ctx = ctx
        self._settings = settings
        self._http = AiPassHttpClient(http_client)
        self._connections = AiPassConnectionRepo(session)
        self._transactions = AiPassOAuthTransactionRepo(session)

    @property
    def configured(self) -> bool:
        return bool(self._settings.aipass_client_id.get_secret_value().strip())

    def callback_uri(self) -> str:
        callback = f"{self._settings.api_url.rstrip('/')}/api/v1/aipass/callback"
        parsed = urlsplit(callback)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise AiPassOAuthError(
                "AIPASS_INSECURE_CALLBACK",
                "AI Pass requires an HTTPS callback outside localhost.",
            )
        return callback

    async def connection(self) -> tuple[bool, bool, str | None]:
        row = await self._connections.get(self._ctx.workspace_id)
        active = await LLMConfigRepo(self._session).get_active(self._ctx.workspace_id)
        active_model = (
            active.model
            if active is not None and active.provider.strip().lower() == AIPASS_PROVIDER
            else None
        )
        return (self.configured, row is not None, active_model)

    async def start_authorization(self) -> str:
        client_id = _client_id(self._settings)
        if await self._connections.get(self._ctx.workspace_id) is not None:
            raise AiPassOAuthError(
                "AIPASS_ALREADY_CONNECTED",
                "Disconnect the current AI Pass account before connecting another.",
            )
        metadata = await self._http.discovery()
        callback = self.callback_uri()
        state = secrets.token_urlsafe(48)
        verifier = secrets.token_urlsafe(64)
        now = datetime.now(UTC)
        transaction = await self._transactions.create(
            workspace_id=self._ctx.workspace_id,
            user_id=uuid.UUID(self._ctx.user_id),
            state_hash=_state_hash(state),
            code_verifier=verifier,
            redirect_uri=callback,
            expires_at=now + _PKCE_TTL,
            created_at=now,
        )
        await write_audit(
            self._session,
            workspace_id=self._ctx.workspace_id,
            user_id=self._ctx.user_id,
            action="aipass.oauth.start",
            resource_type="aipass_connection",
            resource_id=transaction.id,
            metadata={},
        )
        await self._session.commit()
        query = urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": callback,
                "scope": " ".join(_SCOPES),
                "state": state,
                "code_challenge": code_challenge_s256(verifier),
                "code_challenge_method": "S256",
            }
        )
        return f"{metadata.authorization_endpoint}?{query}"

    async def consume_callback_state(self, state: str) -> str:
        transaction = await self._transactions.consume(
            workspace_id=self._ctx.workspace_id,
            user_id=uuid.UUID(self._ctx.user_id),
            state_hash=_state_hash(state),
            now=datetime.now(UTC),
        )
        if transaction is None:
            raise AiPassOAuthError(
                "AIPASS_STATE_INVALID",
                "The AI Pass authorization state is invalid or expired.",
            )
        verifier = transaction.code_verifier_encrypted
        await write_audit(
            self._session,
            workspace_id=self._ctx.workspace_id,
            user_id=self._ctx.user_id,
            action="aipass.oauth.consume",
            resource_type="aipass_connection",
            resource_id=transaction.id,
            metadata={},
        )
        await self._session.commit()
        return verifier

    async def complete_callback(self, *, state: str, code: str) -> None:
        client_id = _client_id(self._settings)
        verifier = await self.consume_callback_state(state)
        callback = self.callback_uri()
        metadata = await self._http.discovery()
        token = await self._http.exchange_code(
            metadata=metadata,
            client_id=client_id,
            code=code,
            code_verifier=verifier,
            redirect_uri=callback,
        )
        connected = False
        try:
            if token.refresh_token is None:
                raise AiPassOAuthError(
                    "AIPASS_REFRESH_REQUIRED",
                    "AI Pass did not return a refresh token.",
                )
            scope = token.scope or " ".join(_SCOPES)
            granted = set(scope.split())
            if "api:access" not in granted:
                raise AiPassOAuthError(
                    "AIPASS_SCOPE_MISSING",
                    "AI Pass did not grant API access.",
                )
            subject = await self._http.userinfo(
                metadata=metadata,
                access_token=token.access_token.get_secret_value(),
            )
            subject_hash = hashlib.sha256(f"{AIPASS_ISSUER}\0{subject}".encode()).hexdigest()
            now = datetime.now(UTC)
            row = await self._connections.create(
                workspace_id=self._ctx.workspace_id,
                connected_by_user_id=uuid.UUID(self._ctx.user_id),
                subject_hash=subject_hash,
                access_token=token.access_token.get_secret_value(),
                refresh_token=token.refresh_token.get_secret_value(),
                token_type=token.token_type,
                scope=scope,
                expires_at=now + timedelta(seconds=token.expires_in),
            )
            await write_audit(
                self._session,
                workspace_id=self._ctx.workspace_id,
                user_id=self._ctx.user_id,
                action="aipass.connect",
                resource_type="aipass_connection",
                resource_id=row.id,
                metadata={},
            )
            await self._session.commit()
            connected = True
        finally:
            if not connected:
                if token.refresh_token is not None:
                    with suppress(Exception):
                        await self._http.revoke(
                            metadata=metadata,
                            client_id=client_id,
                            token=token.refresh_token.get_secret_value(),
                            token_type_hint="refresh_token",
                        )
                with suppress(Exception):
                    await self._http.revoke(
                        metadata=metadata,
                        client_id=client_id,
                        token=token.access_token.get_secret_value(),
                        token_type_hint="access_token",
                    )

    async def models(self) -> list[AiPassModel]:
        manager = AiPassTokenManager(
            workspace_id=self._ctx.workspace_id,
            session_factory=_session_factory(self._session),
            settings=self._settings,
            http=self._http,
        )
        try:
            access_token = await manager.access_token(False)
        except ProviderError as exc:
            raise AiPassOAuthError(exc.code, exc.message) from exc
        return await self._http.models(access_token=access_token)

    async def activate(self, model: str) -> LLMConfig:
        models = await self.models()
        if model not in {item.id for item in models}:
            raise AiPassOAuthError(
                "AIPASS_MODEL_UNAVAILABLE",
                "Choose a model from the live AI Pass catalog.",
            )
        return await LLMConfigService(self._session, self._ctx).activate_aipass(model=model)

    async def disconnect(self) -> bool:
        connection = await self._connections.get_for_update(self._ctx.workspace_id)
        if connection is None:
            raise AiPassOAuthError(
                "AIPASS_NOT_CONNECTED",
                "No AI Pass account is connected to this workspace.",
            )
        refresh_revoked = False
        access_revoked = False
        try:
            client_id = _client_id(self._settings)
            metadata = await self._http.discovery()
            refresh_revoked = await self._http.revoke(
                metadata=metadata,
                client_id=client_id,
                token=connection.refresh_token_encrypted,
                token_type_hint="refresh_token",
            )
            access_revoked = await self._http.revoke(
                metadata=metadata,
                client_id=client_id,
                token=connection.access_token_encrypted,
                token_type_hint="access_token",
            )
        except AiPassOAuthError:
            # Disconnect is fail-closed locally: credentials are always erased
            # even if discovery/revocation is temporarily unavailable.
            pass
        resource_id = connection.id
        await self._connections.delete(self._ctx.workspace_id)
        active = await LLMConfigRepo(self._session).get_active(self._ctx.workspace_id)
        if active is not None and active.provider.strip().lower() == AIPASS_PROVIDER:
            await LLMConfigService(self._session, self._ctx).clear_config(commit=False)
        await write_audit(
            self._session,
            workspace_id=self._ctx.workspace_id,
            user_id=self._ctx.user_id,
            action="aipass.disconnect",
            resource_type="aipass_connection",
            resource_id=resource_id,
            metadata={"revoked": refresh_revoked and access_revoked},
        )
        await self._session.commit()
        return refresh_revoked and access_revoked
