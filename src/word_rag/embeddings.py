from __future__ import annotations

import httpx


class OllamaError(RuntimeError):
    """Raised when request to local Ollama failed."""


class OllamaClient:
    def __init__(self, base_url: str, embedding_model: str, llm_model: str, embed_timeout_sec: float = 60.0, llm_timeout_sec: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.embed_timeout_sec = embed_timeout_sec
        self.llm_timeout_sec = llm_timeout_sec

    def embed(self, text: str) -> list[float]:
        try:
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=self.embed_timeout_sec,
            )
            response.raise_for_status()
            body = response.json()
            return body["embedding"]
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Ollama embedding timeout after {self.embed_timeout_sec}s. "
                f"Check model '{self.embedding_model}' and OLLAMA_BASE_URL={self.base_url}."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama embedding request failed: {exc}") from exc

    def answer(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=self.llm_timeout_sec,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Ollama generation timeout after {self.llm_timeout_sec}s. "
                f"Try increasing LLM_TIMEOUT_SEC or using a smaller model."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama generation request failed: {exc}") from exc
