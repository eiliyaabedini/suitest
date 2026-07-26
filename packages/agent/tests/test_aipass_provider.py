"""Focused contract tests for the AI Pass chat provider."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast

import httpx
import pytest
from suitest_agent.providers.aipass import AiPassProvider
from suitest_agent.providers.base import ChatMessage, ModelCall, ProviderError, StreamChunk


class _ClosingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        while True:
            await __import__("asyncio").sleep(0.01)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_complete_uses_oauth_chat_endpoint_without_leaking_token() -> None:
    seen: dict[str, object] = {}

    async def token_provider(force_refresh: bool) -> str:
        seen["forced"] = force_refresh
        return "test-access-token"

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "model": "live-model",
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AiPassProvider(token_provider=token_provider, http_client=client)
        result = await provider.complete(
            ModelCall(
                model="live-model",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )

    assert seen["url"] == "https://aipass.one/oauth2/v1/chat/completions"
    assert seen["authorization"] == "Bearer test-access-token"
    assert '"model":"live-model"' in str(seen["body"]).replace(" ", "")
    assert result.content == "answer"
    assert result.tokens_in == 3
    assert result.tokens_out == 1


@pytest.mark.asyncio
async def test_401_forces_one_atomic_refresh_and_retries() -> None:
    token_calls: list[bool] = []
    request_count = 0

    async def token_provider(force_refresh: bool) -> str:
        token_calls.append(force_refresh)
        return "rotated-token" if force_refresh else "expired-token"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request.headers["Authorization"] == "Bearer expired-token":
            return httpx.Response(401)
        return httpx.Response(
            200,
            json={
                "model": "live-model",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AiPassProvider(token_provider=token_provider, http_client=client)
        result = await provider.complete(
            ModelCall(
                model="live-model",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )

    assert result.content == "ok"
    assert token_calls == [False, True]
    assert request_count == 2


@pytest.mark.asyncio
async def test_provider_errors_are_bounded_and_never_echo_upstream_body() -> None:
    async def token_provider(force_refresh: bool) -> str:
        return "private-token"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"private-token " + b"x" * 100_000)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AiPassProvider(token_provider=token_provider, http_client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(
                ModelCall(
                    model="live-model",
                    messages=[ChatMessage(role="user", content="hello")],
                )
            )

    assert caught.value.code == "AIPASS_UPSTREAM_ERROR"
    assert "private-token" not in caught.value.message
    assert len(caught.value.message) < 200


@pytest.mark.asyncio
async def test_cancelling_stream_closes_upstream_response() -> None:
    import asyncio

    stream = _ClosingStream()

    async def token_provider(force_refresh: bool) -> str:
        return "test-token"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AiPassProvider(token_provider=token_provider, http_client=client)
        iterator = cast(
            "AsyncGenerator[StreamChunk, None]",
            provider.stream_complete(
                ModelCall(
                    model="live-model",
                    messages=[ChatMessage(role="user", content="hello")],
                )
            ),
        )
        first = await anext(iterator)
        assert first.delta == "hello"
        await iterator.aclose()
        await asyncio.sleep(0)

    assert stream.closed is True


@pytest.mark.asyncio
async def test_oversized_stream_tail_after_complete_line_is_rejected() -> None:
    async def token_provider(force_refresh: bool) -> str:
        return "test-token"

    oversized_delta = b"x" * (1024 * 1024 + 1)
    payload = b': keepalive\ndata: {"choices":[{"delta":{"content":"' + oversized_delta + b'"}}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AiPassProvider(token_provider=token_provider, http_client=client)
        with pytest.raises(ProviderError) as caught:
            _ = [
                chunk
                async for chunk in provider.stream_complete(
                    ModelCall(
                        model="live-model",
                        messages=[ChatMessage(role="user", content="hello")],
                    )
                )
            ]

    assert caught.value.code == "AIPASS_RESPONSE_TOO_LARGE"
