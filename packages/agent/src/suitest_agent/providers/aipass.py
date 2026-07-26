"""AI Pass OAuth-backed OpenAI-compatible chat provider.

The provider never owns OAuth persistence.  Its caller supplies an async token
provider that returns a valid access token and atomically rotates refresh tokens
when ``force_refresh`` is true.  This keeps bearer credentials out of browser
state and preserves the provider package's database independence.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx

from suitest_agent.providers.base import (
    CompletionResult,
    ModelCall,
    ProviderError,
    StreamChunk,
)

AIPASS_CHAT_COMPLETIONS_URL = "https://aipass.one/oauth2/v1/chat/completions"

AccessTokenProvider = Callable[[bool], Awaitable[str]]

_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 60.0
_TOTAL_TIMEOUT_SECONDS = 120.0
_MAX_REQUEST_BYTES = 512 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_STREAM_BYTES = 8 * 1024 * 1024
_MAX_STREAM_LINE_BYTES = 1024 * 1024


def _object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return {str(key): item for key, item in value.items()}


def _object_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


class AiPassProvider:
    """Chat provider that spends from the connected AI Pass wallet."""

    name = "aipass"

    def __init__(
        self,
        *,
        token_provider: AccessTokenProvider,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token_provider = token_provider
        self._http_client = http_client

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT_SECONDS,
            read=_READ_TIMEOUT_SECONDS,
            write=_CONNECT_TIMEOUT_SECONDS,
            pool=_CONNECT_TIMEOUT_SECONDS,
        )
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            yield client

    @staticmethod
    def _body(call: ModelCall, *, stream: bool) -> dict[str, object]:
        body: dict[str, object] = {
            "model": call.model,
            "messages": [message.model_dump() for message in call.messages],
            "temperature": call.temperature,
            "max_tokens": call.max_tokens,
            "stream": stream,
        }
        if call.tools:
            body["tools"] = call.tools
        if call.seed is not None:
            body["seed"] = call.seed
        if stream:
            body["stream_options"] = {"include_usage": True}
        encoded = json.dumps(body, separators=(",", ":")).encode()
        if len(encoded) > _MAX_REQUEST_BYTES:
            raise ProviderError(
                "AIPASS_REQUEST_TOO_LARGE",
                "AI Pass chat request exceeds the configured size limit.",
            )
        return body

    @staticmethod
    async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > limit:
                raise ProviderError(
                    "AIPASS_RESPONSE_TOO_LARGE",
                    "AI Pass response exceeds the configured size limit.",
                )
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _raise_status(status_code: int) -> None:
        if status_code == 401:
            raise ProviderError(
                "AIPASS_AUTH_FAILED",
                "The AI Pass connection is no longer authorized.",
            )
        if status_code == 402:
            raise ProviderError(
                "AIPASS_WALLET_INSUFFICIENT",
                "The connected AI Pass wallet cannot fund this request.",
            )
        if status_code == 429:
            raise ProviderError(
                "AIPASS_RATE_LIMITED",
                "AI Pass is temporarily rate limited.",
            )
        raise ProviderError(
            "AIPASS_UPSTREAM_ERROR",
            "AI Pass could not complete the request.",
        )

    async def complete(self, call: ModelCall) -> CompletionResult:
        body = self._body(call, stream=False)
        for attempt in range(2):
            token = await self._token_provider(attempt == 1)
            try:
                async with asyncio.timeout(_TOTAL_TIMEOUT_SECONDS):
                    async with self._client() as client:
                        async with client.stream(
                            "POST",
                            AIPASS_CHAT_COMPLETIONS_URL,
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Accept": "application/json",
                            },
                            json=body,
                        ) as response:
                            if response.status_code == 401 and attempt == 0:
                                continue
                            if response.status_code != 200:
                                self._raise_status(response.status_code)
                            raw = await self._read_bounded(response, _MAX_RESPONSE_BYTES)
            except TimeoutError as exc:
                raise ProviderError(
                    "AIPASS_TIMEOUT",
                    "AI Pass did not respond within the configured time limit.",
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderError(
                    "AIPASS_NETWORK_ERROR",
                    "AI Pass is temporarily unreachable.",
                ) from exc
            return self._normalize_complete(raw, call.model)
        raise ProviderError(
            "AIPASS_AUTH_FAILED",
            "The AI Pass connection is no longer authorized.",
        )

    @staticmethod
    def _normalize_complete(raw: bytes, requested_model: str) -> CompletionResult:
        try:
            loaded: object = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "AIPASS_INVALID_RESPONSE",
                "AI Pass returned an invalid completion response.",
            ) from exc
        root = _object_dict(loaded)
        choices = _object_list(root.get("choices")) if root is not None else []
        choice = _object_dict(choices[0]) if choices else None
        message = _object_dict(choice.get("message")) if choice is not None else None
        content = message.get("content") if message is not None else None
        if not isinstance(content, str):
            raise ProviderError(
                "AIPASS_INVALID_RESPONSE",
                "AI Pass returned an invalid completion response.",
            )

        usage = _object_dict(root.get("usage")) if root is not None else None
        tokens_in = _positive_int(usage.get("prompt_tokens")) if usage is not None else 0
        tokens_out = _positive_int(usage.get("completion_tokens")) if usage is not None else 0
        model_value = root.get("model") if root is not None else None
        model = model_value if isinstance(model_value, str) else requested_model
        finish_value = choice.get("finish_reason") if choice is not None else None
        finish_reason = finish_value if isinstance(finish_value, str) else "stop"

        tool_calls: list[dict[str, object]] = []
        raw_tool_calls = _object_list(message.get("tool_calls")) if message is not None else []
        for raw_tool_call in raw_tool_calls:
            tool_call = _object_dict(raw_tool_call)
            function = _object_dict(tool_call.get("function")) if tool_call is not None else None
            if tool_call is None or function is None:
                continue
            tool_calls.append(
                {
                    "id": tool_call.get("id") if isinstance(tool_call.get("id"), str) else "",
                    "name": (function.get("name") if isinstance(function.get("name"), str) else ""),
                    "arguments": (
                        function.get("arguments")
                        if isinstance(function.get("arguments"), str)
                        else ""
                    ),
                }
            )

        return CompletionResult(
            content=content,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    async def stream_complete(self, call: ModelCall) -> AsyncIterator[StreamChunk]:
        body = self._body(call, stream=True)
        for attempt in range(2):
            token = await self._token_provider(attempt == 1)
            try:
                async with asyncio.timeout(_TOTAL_TIMEOUT_SECONDS):
                    async with self._client() as client:
                        async with client.stream(
                            "POST",
                            AIPASS_CHAT_COMPLETIONS_URL,
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Accept": "text/event-stream",
                            },
                            json=body,
                        ) as response:
                            if response.status_code == 401 and attempt == 0:
                                continue
                            if response.status_code != 200:
                                self._raise_status(response.status_code)
                            async for chunk in self._stream_chunks(response):
                                yield chunk
                            return
            except TimeoutError as exc:
                raise ProviderError(
                    "AIPASS_TIMEOUT",
                    "AI Pass did not respond within the configured time limit.",
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderError(
                    "AIPASS_NETWORK_ERROR",
                    "AI Pass is temporarily unreachable.",
                ) from exc
        raise ProviderError(
            "AIPASS_AUTH_FAILED",
            "The AI Pass connection is no longer authorized.",
        )

    @staticmethod
    async def _stream_chunks(response: httpx.Response) -> AsyncIterator[StreamChunk]:
        buffer = b""
        total = 0
        tokens_out = 0
        done_sent = False
        async for part in response.aiter_bytes():
            total += len(part)
            if total > _MAX_STREAM_BYTES:
                raise ProviderError(
                    "AIPASS_RESPONSE_TOO_LARGE",
                    "AI Pass stream exceeds the configured size limit.",
                )
            buffer += part
            if len(buffer) > _MAX_STREAM_LINE_BYTES and b"\n" not in buffer:
                raise ProviderError(
                    "AIPASS_RESPONSE_TOO_LARGE",
                    "AI Pass stream frame exceeds the configured size limit.",
                )
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                line = raw_line.rstrip(b"\r")
                if len(line) > _MAX_STREAM_LINE_BYTES:
                    raise ProviderError(
                        "AIPASS_RESPONSE_TOO_LARGE",
                        "AI Pass stream frame exceeds the configured size limit.",
                    )
                parsed = AiPassProvider._parse_stream_line(line)
                if parsed is None:
                    continue
                if parsed.done:
                    done_sent = True
                    yield parsed.model_copy(update={"tokens_out": tokens_out or parsed.tokens_out})
                    continue
                if parsed.tokens_out:
                    tokens_out = parsed.tokens_out
                if parsed.delta:
                    yield parsed
            if len(buffer) > _MAX_STREAM_LINE_BYTES:
                raise ProviderError(
                    "AIPASS_RESPONSE_TOO_LARGE",
                    "AI Pass stream frame exceeds the configured size limit.",
                )
        if buffer.strip():
            parsed = AiPassProvider._parse_stream_line(buffer.rstrip(b"\r"))
            if parsed is not None:
                if parsed.done:
                    done_sent = True
                yield parsed
        if not done_sent:
            yield StreamChunk(done=True, tokens_out=tokens_out)

    @staticmethod
    def _parse_stream_line(line: bytes) -> StreamChunk | None:
        if not line.startswith(b"data:"):
            return None
        payload = line[len(b"data:") :].strip()
        if payload == b"[DONE]":
            return StreamChunk(done=True)
        if not payload:
            return None
        try:
            loaded: object = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "AIPASS_INVALID_RESPONSE",
                "AI Pass returned an invalid stream frame.",
            ) from exc
        root = _object_dict(loaded)
        if root is None:
            return None
        usage = _object_dict(root.get("usage"))
        tokens_out = _positive_int(usage.get("completion_tokens")) if usage is not None else 0
        choices = _object_list(root.get("choices"))
        if not choices:
            return StreamChunk(tokens_out=tokens_out)
        choice = _object_dict(choices[0])
        delta = _object_dict(choice.get("delta")) if choice is not None else None
        content = delta.get("content") if delta is not None else None
        return StreamChunk(
            delta=content if isinstance(content, str) else "",
            tokens_out=tokens_out,
        )

    def cost_usd(self, result: CompletionResult) -> float:
        return result.cost_usd
