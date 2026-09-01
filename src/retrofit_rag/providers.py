"""Small, testable HTTP adapters for OpenAI-compatible and Ollama chat APIs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


Messages = list[dict[str, str]]


class LLM(Protocol):
    def generate(self, messages: Messages) -> str: ...


class LLMError(RuntimeError):
    """A provider failed or returned an invalid response."""


class ConfigurationError(LLMError):
    """Required non-secret provider configuration is missing."""


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise LLMError(f"LLM provider returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise LLMError("LLM provider could not be reached") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM provider returned non-JSON data") from exc
    if not isinstance(data, dict):
        raise LLMError("LLM provider returned an unexpected payload")
    return data


@dataclass(frozen=True)
class OpenAICompatibleLLM:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM":
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
        data = _post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI-compatible response did not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("OpenAI-compatible response contained empty message content")
        return content


@dataclass(frozen=True)
class OllamaLLM:
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "OllamaLLM":
        model = os.getenv("OLLAMA_MODEL", "").strip()
        if not model:
            raise ConfigurationError("OLLAMA_MODEL is required")
        return cls(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
        )

    def generate(self, messages: Messages) -> str:
        data = _post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            {},
            self.timeout,
        )
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Ollama response did not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama response contained empty message content")
        return content


def build_llm(provider: str) -> LLM:
    normalized = provider.strip().lower()
    if normalized in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLM.from_env()
    if normalized == "ollama":
        return OllamaLLM.from_env()
    raise ConfigurationError("LLM_PROVIDER must be 'openai' or 'ollama'")
