"""Focused RED tests for the optional AI Pass OAuth connection."""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr
from sqlalchemy import select, text
from suitest_api.services.aipass_oauth_service import (
    AIPASS_DISCOVERY_URL,
    AIPASS_MODELS_URL,
    code_challenge_s256,
    parse_models_payload,
)
from suitest_api.settings import Settings
from suitest_db.models.aipass_oauth import AiPassConnection
from suitest_db.models.llm_config import LLMConfig
from suitest_shared.domain.enums import Role

if TYPE_CHECKING:
    from api_harness import ApiDb


def test_pkce_challenge_is_s256_without_padding() -> None:
    verifier = "strong-verifier-with-enough-entropy-0123456789"
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    assert code_challenge_s256(verifier) == expected
    assert "=" not in code_challenge_s256(verifier)


def test_models_accept_openai_list_and_keep_chat_capable_entries() -> None:
    models = parse_models_payload(
        {
            "object": "list",
            "data": [
                {
                    "id": "live-chat-model",
                    "name": "Live Chat Model",
                    "methods": ["chat_completions"],
                },
                {
                    "id": "live-image-model",
                    "name": "Live Image Model",
                    "methods": ["images_generations"],
                },
            ],
        }
    )
    assert [model.id for model in models] == ["live-chat-model"]


def test_models_accept_legacy_string_array_defensively() -> None:
    models = parse_models_payload(["model-one", "model-two", "", 42])
    assert [(model.id, model.name) for model in models] == [
        ("model-one", "model-one"),
        ("model-two", "model-two"),
    ]


@pytest.mark.asyncio
async def test_authorize_fails_closed_without_protected_client_id(api_db: ApiDb) -> None:
    user = await api_db.seed_user(email="aipass-owner@example.com")
    ws = await api_db.seed_workspace(slug="aipass-owner", name="AI Pass Owner")
    await api_db.seed_membership(workspace_id=ws.id, user_id=user.id, role=Role.OWNER)

    async with api_db.client(user) as client:
        response = await client.post(f"/api/v1/workspaces/{ws.id}/aipass/authorize")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "AIPASS_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_aipass_cannot_be_saved_through_byok_config(api_db: ApiDb) -> None:
    user = await api_db.seed_user(email="aipass-byok@example.com")
    ws = await api_db.seed_workspace(slug="aipass-byok", name="AI Pass BYOK")
    await api_db.seed_membership(workspace_id=ws.id, user_id=user.id, role=Role.OWNER)

    async with api_db.client(user) as client:
        response = await client.put(
            f"/api/v1/workspaces/{ws.id}/llm-config",
            json={
                "provider": "aipass",
                "model": "invented-model",
                "apiKey": "must-not-be-accepted",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "AIPASS_OAUTH_REQUIRED"


@pytest.mark.asyncio
async def test_oauth_lifecycle_keeps_rotated_tokens_server_side(api_db: ApiDb) -> None:
    user = await api_db.seed_user(email="aipass-flow@example.com")
    ws = await api_db.seed_workspace(slug="aipass-flow", name="AI Pass Flow")
    await api_db.seed_membership(workspace_id=ws.id, user_id=user.id, role=Role.OWNER)

    initial_access = "test-access-token-initial"
    initial_refresh = "test-refresh-token-initial"
    rotated_access = "test-access-token-rotated"
    rotated_refresh = "test-refresh-token-rotated"
    token_grants: list[str] = []
    revoked_tokens: list[str] = []
    model_request_urls: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AIPASS_DISCOVERY_URL:
            return httpx.Response(
                200,
                json={
                    "issuer": "https://aipass.one",
                    "authorization_endpoint": "https://aipass.one/oauth2/authorize",
                    "token_endpoint": "https://aipass.one/oauth2/token",
                    "userinfo_endpoint": "https://aipass.one/oauth2/userinfo",
                    "revocation_endpoint": "https://aipass.one/oauth2/revoke",
                    "scopes_supported": ["profile:read", "api:access"],
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": ["none"],
                },
            )
        if str(request.url) == "https://aipass.one/oauth2/token":
            form = parse_qs(request.content.decode())
            grant = form["grant_type"][0]
            token_grants.append(grant)
            assert "client_secret" not in form
            if grant == "authorization_code":
                assert form["code_verifier"][0]
                return httpx.Response(
                    200,
                    json={
                        "access_token": initial_access,
                        "refresh_token": initial_refresh,
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "profile:read api:access",
                    },
                )
            assert form["refresh_token"] == [initial_refresh]
            return httpx.Response(
                200,
                json={
                    "access_token": rotated_access,
                    "refresh_token": rotated_refresh,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "profile:read api:access",
                },
            )
        if str(request.url) == "https://aipass.one/oauth2/userinfo":
            assert request.headers["Authorization"] == f"Bearer {initial_access}"
            return httpx.Response(200, json={"sub": "test-account-subject"})
        if str(request.url) == AIPASS_MODELS_URL:
            model_request_urls.append(str(request.url))
            assert request.headers["Authorization"] == f"Bearer {rotated_access}"
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "catalog-chat-model",
                            "name": "Catalog Chat Model",
                            "methods": ["chat_completions"],
                        },
                        {
                            "id": "catalog-image-model",
                            "name": "Catalog Image Model",
                            "methods": ["images_generations"],
                        },
                    ],
                },
            )
        if str(request.url) == "https://aipass.one/oauth2/revoke":
            form = parse_qs(request.content.decode())
            assert "client_secret" not in form
            revoked_tokens.append(form["token"][0])
            return httpx.Response(200)
        raise AssertionError(f"unexpected upstream request: {request.method} {request.url}")

    settings = Settings(
        api_url="https://suitest.example",
        web_url="https://suitest.example",
        aipass_client_id=SecretStr("test-public-client"),
    )
    app = api_db.app_for(user)
    app.state.settings = settings
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app.state.aipass_http_client = upstream_client
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://suitest.example",
                follow_redirects=False,
            ) as client:
                authorize = await client.post(f"/api/v1/workspaces/{ws.id}/aipass/authorize")
                assert authorize.status_code == 303
                assert authorize.headers["cache-control"] == "no-store"
                assert authorize.headers["referrer-policy"] == "no-referrer"
                authorization_url = authorize.headers["location"]
                authorization_query = parse_qs(urlsplit(authorization_url).query)
                state = authorization_query["state"][0]
                assert len(state) >= 64
                assert authorization_query["response_type"] == ["code"]
                assert authorization_query["code_challenge_method"] == ["S256"]
                assert authorization_query["client_id"] == ["test-public-client"]
                assert authorization_query["redirect_uri"] == [
                    "https://suitest.example/api/v1/aipass/callback"
                ]
                assert "client_secret" not in authorization_query

                async with api_db.maker() as session:
                    raw_transaction = (
                        await session.execute(
                            text(
                                "SELECT state_hash, code_verifier_encrypted "
                                "FROM aipass_oauth_transactions"
                            )
                        )
                    ).one()
                assert raw_transaction.state_hash == hashlib.sha256(state.encode()).hexdigest()
                assert isinstance(raw_transaction.code_verifier_encrypted, bytes)
                assert state.encode() not in raw_transaction.code_verifier_encrypted

                callback = await client.get(
                    "/api/v1/aipass/callback",
                    params={"state": state, "code": "test-authorization-code"},
                )
                assert callback.status_code == 303
                assert callback.headers["cache-control"] == "no-store"
                assert callback.headers["referrer-policy"] == "no-referrer"
                assert "aipass=connected" in callback.headers["location"]
                assert initial_access not in callback.text
                assert initial_refresh not in callback.text

                replay = await client.get(
                    "/api/v1/aipass/callback",
                    params={"state": state, "code": "test-authorization-code"},
                )
                assert replay.status_code == 303
                assert "AIPASS_STATE_INVALID" in replay.headers["location"]
                assert token_grants == ["authorization_code"]

                connection = await client.get(f"/api/v1/workspaces/{ws.id}/aipass/connection")
                assert connection.json() == {
                    "configured": True,
                    "connected": True,
                    "active": False,
                    "activeModel": None,
                }
                assert initial_access not in connection.text
                assert initial_refresh not in connection.text

                async with api_db.maker() as session:
                    raw_connection = (
                        await session.execute(
                            text(
                                "SELECT access_token_encrypted, refresh_token_encrypted, "
                                "subject_hash FROM aipass_connections"
                            )
                        )
                    ).one()
                    stored_connection = await session.scalar(
                        select(AiPassConnection).where(AiPassConnection.workspace_id == ws.id)
                    )
                    assert stored_connection is not None
                    stored_connection.expires_at = datetime.now(UTC) - timedelta(minutes=1)
                    await session.commit()
                assert isinstance(raw_connection.access_token_encrypted, bytes)
                assert isinstance(raw_connection.refresh_token_encrypted, bytes)
                assert initial_access.encode() not in raw_connection.access_token_encrypted
                assert initial_refresh.encode() not in raw_connection.refresh_token_encrypted
                assert raw_connection.subject_hash != "test-account-subject"

                models = await client.get(f"/api/v1/workspaces/{ws.id}/aipass/models")
                assert models.status_code == 200, models.text
                assert models.json() == {
                    "models": [{"id": "catalog-chat-model", "name": "Catalog Chat Model"}]
                }
                assert model_request_urls == [AIPASS_MODELS_URL]
                assert token_grants == ["authorization_code", "refresh_token"]

                activate = await client.put(
                    f"/api/v1/workspaces/{ws.id}/aipass/connection",
                    json={"model": "catalog-chat-model"},
                )
                assert activate.status_code == 200, activate.text
                assert activate.json()["activeModel"] == "catalog-chat-model"

                async with api_db.maker() as session:
                    rotated = await session.scalar(
                        select(AiPassConnection).where(AiPassConnection.workspace_id == ws.id)
                    )
                    active = await session.scalar(
                        select(LLMConfig).where(
                            LLMConfig.workspace_id == ws.id,
                            LLMConfig.is_active.is_(True),
                        )
                    )
                assert rotated is not None
                assert rotated.access_token_encrypted == rotated_access
                assert rotated.refresh_token_encrypted == rotated_refresh
                assert rotated.refresh_generation == 1
                assert active is not None
                assert active.provider == "aipass"
                assert active.api_key_encrypted is None

                disconnected = await client.delete(f"/api/v1/workspaces/{ws.id}/aipass/connection")
                assert disconnected.status_code == 200
                assert disconnected.json() == {"revoked": True}

    assert set(revoked_tokens) == {rotated_access, rotated_refresh}
    async with api_db.maker() as session:
        assert (
            await session.scalar(
                select(AiPassConnection).where(AiPassConnection.workspace_id == ws.id)
            )
            is None
        )
        active_after_disconnect = await session.scalar(
            select(LLMConfig).where(
                LLMConfig.workspace_id == ws.id,
                LLMConfig.is_active.is_(True),
            )
        )
    assert active_after_disconnect is None


@pytest.mark.asyncio
async def test_disconnect_erases_tokens_when_revocation_is_unreachable(api_db: ApiDb) -> None:
    user = await api_db.seed_user(email="aipass-offline@example.com")
    ws = await api_db.seed_workspace(slug="aipass-offline", name="AI Pass Offline")
    await api_db.seed_membership(workspace_id=ws.id, user_id=user.id, role=Role.OWNER)
    await api_db.add_all(
        [
            AiPassConnection(
                workspace_id=ws.id,
                connected_by_user_id=user.id,
                subject_hash="0" * 64,
                access_token_encrypted="offline-access-token",
                refresh_token_encrypted="offline-refresh-token",
                token_type="Bearer",
                scope="api:access profile:read",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        ]
    )

    def unavailable(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == AIPASS_DISCOVERY_URL
        return httpx.Response(503)

    settings = Settings(
        api_url="https://suitest.example",
        web_url="https://suitest.example",
        aipass_client_id=SecretStr("test-public-client"),
    )
    app = api_db.app_for(user)
    app.state.settings = settings
    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as upstream_client:
        app.state.aipass_http_client = upstream_client
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://suitest.example",
            ) as client:
                response = await client.delete(f"/api/v1/workspaces/{ws.id}/aipass/connection")

    assert response.status_code == 200
    assert response.json() == {"revoked": False}
    async with api_db.maker() as session:
        remaining = await session.scalar(
            select(AiPassConnection).where(AiPassConnection.workspace_id == ws.id)
        )
    assert remaining is None
