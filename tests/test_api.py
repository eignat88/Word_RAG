from fastapi.testclient import TestClient

from word_rag.api import app
from word_rag.embeddings import OllamaError


def test_ask_returns_504_on_ollama_timeout(monkeypatch):
    client = TestClient(app)

    def _raise_timeout(**kwargs):
        raise OllamaError("Ollama generation timeout after 300.0s.")

    monkeypatch.setattr("word_rag.api.service.answer", _raise_timeout)
    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 504
    detail = response.json()["detail"]
    assert detail["code"] == "OLLAMA_TIMEOUT"
    assert "увеличьте llm_timeout_sec" in detail["hint"].lower()
    assert "timeout" in detail["error"].lower()


def test_ask_passes_llm_timeout_from_payload(monkeypatch):
    client = TestClient(app)
    captured = {}

    def _answer(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr("word_rag.api.service.answer", _answer)
    response = client.post("/ask", json={"question": "test", "llm_timeout_sec": 777})

    assert response.status_code == 200
    assert captured["llm_timeout_sec"] == 777
