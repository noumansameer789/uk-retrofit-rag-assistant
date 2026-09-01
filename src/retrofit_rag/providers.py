"""Small, testable HTTP adapters for OpenAI-compatible and Ollama chat APIs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

Messages = list[dict[str, str]]
MAX_PROVIDER_RESPONSE_BYTES = 1_000_000


class LLM(Protocol):
    def generate(self, messages: Messages) -> str: ...

    def ready(self) -> bool: ...


class LLMError(RuntimeError):
    """A provider failed or returned an invalid response."""


class ConfigurationError(LLMError):
    """Required non-secret provider configuration is missing."""


def _decode_response(response: Any) -> dict[str, Any]:
    raw = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
    if len(raw) > MAX_PROVIDER_RESPONSE_BYTES:
        raise LLMError("LLM provider response was too large")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMError("LLM provider returned non-JSON data") from exc
    if not isinstance(data, dict):
        raise LLMError("LLM provider returned an unexpected payload")
    return data


def _request_json(
    url: str,
    headers: dict[str, str],
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json", **headers},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return _decode_response(response)
    except HTTPError as exc:
        raise LLMError(f"LLM provider returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise LLMError("LLM provider could not be reached") from exc


@dataclass(frozen=True)
class OpenAICompatibleLLM:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> OpenAICompatibleLLM:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required")
        if not model:
            raise ConfigurationError("OPENAI_MODEL is required")
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.getenv("OPENAI_BASE_URL", cls.base_url).rstrip("/"),
        )

    def generate(self, messages: Messages) -> str:
        data = _request_json(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
            payload={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI-compatible response did not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("OpenAI-compatible response contained empty message content")
        return content

    def ready(self) -> bool:
        data = _request_json(
            f"{self.base_url}/models",
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
        )
        return isinstance(data.get("data"), list)


@dataclass(frozen=True)
class OllamaLLM:
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> OllamaLLM:
        model = os.getenv("OLLAMA_MODEL", "").strip()
        if not model:
            raise ConfigurationError("OLLAMA_MODEL is required")
        return cls(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
        )

    def generate(self, messages: Messages) -> str:
        data = _request_json(
            f"{self.base_url}/api/chat",
            {},
            self.timeout,
            payload={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
        )
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Ollama response did not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama response contained empty message content")
        return content

    def ready(self) -> bool:
        data = _request_json(f"{self.base_url}/api/tags", {}, self.timeout)
        models = data.get("models")
        if not isinstance(models, list):
            return False
        expected = {self.model, f"{self.model}:latest"}
        return any(
            isinstance(model, dict)
            and isinstance(model.get("name"), str)
            and model["name"] in expected
            for model in models
        )


def build_llm(provider: str) -> LLM:
    normalized = provider.strip().lower()
    if normalized in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLM.from_env()
    if normalized == "ollama":
        return OllamaLLM.from_env()
    raise ConfigurationError("LLM_PROVIDER must be 'openai' or 'ollama'")
