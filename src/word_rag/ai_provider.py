from __future__ import annotations

from typing import Protocol

import httpx

from .config import Settings
from .embeddings import OllamaClient, OllamaError


class AIProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def answer(self, prompt: str) -> str: ...


class OpenAICompatibleError(RuntimeError):
    """Raised when request to OpenAI-compatible endpoint failed."""


class OllamaProvider:
    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    def embed(self, text: str) -> list[float]:
        return self._client.embed(text)

    def answer(self, prompt: str) -> str:
        return self._client.answer(prompt)


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        embedding_model: str,
        llm_model: str,
        embed_timeout_sec: float = 60.0,
        llm_timeout_sec: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.embed_timeout_sec = embed_timeout_sec
        self.llm_timeout_sec = llm_timeout_sec

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, text: str) -> list[float]:
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.embedding_model, "input": text},
                headers=self._headers(),
                timeout=self.embed_timeout_sec,
            )
            response.raise_for_status()
            body = response.json()
            return body["data"][0]["embedding"]
        except httpx.TimeoutException as exc:
            raise OpenAICompatibleError(
                f"OpenAI-compatible embedding timeout after {self.embed_timeout_sec}s. "
                f"Check model '{self.embedding_model}' and OPENAI_COMPAT_BASE_URL={self.base_url}."
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatibleError(f"OpenAI-compatible embedding request failed: {exc}") from exc

    def answer(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                headers=self._headers(),
                timeout=self.llm_timeout_sec,
            )
            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"]["content"].strip()
        except httpx.TimeoutException as exc:
            raise OpenAICompatibleError(
                f"OpenAI-compatible generation timeout after {self.llm_timeout_sec}s. "
                "Try increasing LLM_TIMEOUT_SEC or using a smaller model."
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatibleError(f"OpenAI-compatible generation request failed: {exc}") from exc


def build_ai_provider(settings: Settings) -> AIProvider:
    provider = settings.ai_provider.lower()

    if provider == "openai_compatible":
        if not settings.openai_compat_api_key:
            raise OpenAICompatibleError("OPENAI_COMPAT_API_KEY must be set for openai_compatible provider")
        return OpenAICompatibleProvider(
            base_url=settings.openai_compat_base_url,
            api_key=settings.openai_compat_api_key,
            embedding_model=settings.openai_compat_embedding_model,
            llm_model=settings.openai_compat_llm_model,
            embed_timeout_sec=settings.embed_timeout_sec,
            llm_timeout_sec=settings.llm_timeout_sec,
        )

    if provider == "ollama":
        return OllamaProvider(
            OllamaClient(
                base_url=settings.ollama_base_url,
                embedding_model=settings.embedding_model,
                llm_model=settings.llm_model,
                embed_timeout_sec=settings.embed_timeout_sec,
                llm_timeout_sec=settings.llm_timeout_sec,
            )
        )

    raise OllamaError(f"Unknown AI provider: {settings.ai_provider}")
