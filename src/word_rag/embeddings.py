from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(self, base_url: str, embedding_model: str, llm_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def embed(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
            timeout=60.0,
        )
        response.raise_for_status()
        body = response.json()
        return body["embedding"]

    def answer(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.llm_model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
