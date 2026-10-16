"""Small OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class LLMClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120,
        retries: int = 2,
        seed: int | None = None,
        temperature: float = 0.2,
        transport: Transport | None = None,
    ) -> None:
        if not api_url.strip() or not api_key.strip() or not model.strip():
            raise ValueError("api_url, api_key, and model are required")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.endpoint = normalize_chat_endpoint(api_url)
        self._api_url = api_url.strip().rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.seed = seed
        self.temperature = temperature
        self._transport = transport or _urllib_transport

    def complete(self, *, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._transport(
                    self.endpoint, headers, payload, self.timeout_seconds
                )
                return self.redact(_extract_text(response))
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 4))
        raise RuntimeError(
            "LLM request failed after "
            f"{self.retries + 1} attempt(s) ({type(last_error).__name__})"
        ) from None

    def redact(self, text: str) -> str:
        """Remove exact credential and endpoint values before text can be archived."""
        replacements = (
            (f"Bearer {self._api_key}", "[REDACTED_API_KEY]"),
            (self._api_key, "[REDACTED_API_KEY]"),
            (self.endpoint, "[REDACTED_API_URL]"),
            (self._api_url, "[REDACTED_API_URL]"),
        )
        safe = text
        for value, replacement in sorted(
            replacements, key=lambda item: len(item[0]), reverse=True
        ):
            if value:
                safe = safe.replace(value, replacement)
        return safe


def normalize_chat_endpoint(api_url: str) -> str:
    url = api_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _urllib_transport(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _extract_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        raise ValueError("response has no choices[0].message.content") from exc
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ).strip()
        if text:
            return text
    raise ValueError("LLM response content is empty or unsupported")
