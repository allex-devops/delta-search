"""A minimal OpenAI-compatible chat and embeddings client, with retries and an optional disk cache."""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import httpx

from .cache import DiskCache, make_key

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    ok: bool = True  # False when the model sent arguments that weren't valid JSON


def parse_tool_calls(message: dict) -> list[ToolCall]:
    out = []
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function", {})
        raw = fn.get("arguments", {})
        if isinstance(raw, dict):  # some providers hand back an object instead of a JSON string
            args, ok = raw, True
        else:
            try:
                args, ok = json.loads(raw or "{}"), True
            except json.JSONDecodeError:
                args, ok = {}, False
        out.append(ToolCall(id=call.get("id") or f"call_{i}", name=fn.get("name", ""), args=args, ok=ok))
    return out


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None  # missing, or the HTTP-date form; normal backoff is fine


class LLM:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: float = 120.0,
        retries: int = 3,
        backoff: float = 0.5,
        jitter: bool = True,
        cache: DiskCache | None = None,
        offline: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Any] = asyncio.sleep,
    ):
        base_url = base_url or os.environ.get("LLM_BASE_URL")
        if not base_url:
            raise LLMError("base_url is required: pass it, or set LLM_BASE_URL")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.jitter = jitter
        self.cache = cache
        self.offline = offline  # replay only: a cache miss is an error instead of a request
        self.transport = transport
        self.sleep = sleep
        self.asleep = asleep
        self._client: httpx.Client | None = None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _delay(self, attempt: int) -> float:
        delay = self.backoff * 2**attempt
        return delay * (0.5 + random.random() / 2) if self.jitter else delay

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(transport=self.transport, timeout=self.timeout)
        return self._client

    def _post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            wait = None
            try:
                resp = self._http().post(f"{self.base_url}{path}", json=payload, headers=self._headers())
            except httpx.TransportError as e:  # connection refused, resets and timeouts
                last = e
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in RETRY_STATUS:
                    raise LLMError(f"{resp.status_code} from {path}: {resp.text[:300]}")
                last = LLMError(f"{resp.status_code} from {path}")
                wait = _retry_after(resp)
            if attempt < self.retries:
                self.sleep(wait if wait is not None else self._delay(attempt))
        raise LLMError(f"gave up on {path} after {self.retries + 1} attempts: {last}") from last

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> dict:
        payload: dict[str, Any] = {"model": model or self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format

        # only temperature 0 is cached, anything else is supposed to vary
        key = make_key("chat", self.base_url, payload) if self.cache and temperature == 0 else None
        if key:
            hit = self.cache.get_json(key)
            if hit is not None:
                return hit
        if self.offline:
            raise LLMError("offline mode and no cached response for this request")

        message = self._post("/chat/completions", payload)["choices"][0]["message"]
        if key:
            self.cache.set_json(key, message)
        return message

    def ping(self) -> bool:
        """True if the server answers /models. One quick try, no retries: this backs readiness probes."""
        try:
            return self._http().get(f"{self.base_url}/models", headers=self._headers(), timeout=3).status_code < 400
        except httpx.HTTPError:
            return False

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        data = self._post("/embeddings", {"model": model or self.embed_model, "input": texts})["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    async def astream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout) as client:
            resp = await self._open_stream(client, payload)
            try:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
            finally:
                # close the connection so the server stops generating for a client that left
                await resp.aclose()

    async def _open_stream(self, client: httpx.AsyncClient, payload: dict) -> httpx.Response:
        # retries only cover getting the stream open; once tokens flow we can't replay them
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            wait = None
            try:
                req = client.build_request(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                )
                resp = await client.send(req, stream=True)
            except httpx.TransportError as e:
                last = e
            else:
                if resp.status_code < 400:
                    return resp
                body = (await resp.aread()).decode(errors="replace")[:300]
                await resp.aclose()
                if resp.status_code not in RETRY_STATUS:
                    raise LLMError(f"{resp.status_code} from /chat/completions: {body}")
                last = LLMError(f"{resp.status_code} from /chat/completions")
                wait = _retry_after(resp)
            if attempt < self.retries:
                await self.asleep(wait if wait is not None else self._delay(attempt))
        raise LLMError(f"gave up opening stream after {self.retries + 1} attempts: {last}") from last
