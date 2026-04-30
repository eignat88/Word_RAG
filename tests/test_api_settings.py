from fastapi.testclient import TestClient

from word_rag import api
from word_rag.config import Settings


EXPECTED_FIELDS = {
    "storage_backend",
    "database_url",
    "pg_schema",
    "pg_documents_table",
    "pg_chunks_table",
    "sqlite_path",
    "ollama_base_url",
    "embedding_model",
    "llm_model",
    "top_k",
    "chunk_min_chars",
    "chunk_max_chars",
    "index_min_chars",
    "embed_timeout_sec",
    "llm_timeout_sec",
}


def test_settings_endpoint_contains_expected_fields(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mydb")
    api.settings = Settings()

    client = TestClient(api.app)
    response = client.get("/settings")

    assert response.status_code == 200
    payload = response.json()
    assert EXPECTED_FIELDS.issubset(payload.keys())


def test_settings_endpoint_masks_database_password(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mydb")
    api.settings = Settings()

    client = TestClient(api.app)
    response = client.get("/settings")

    assert response.status_code == 200
    assert response.json()["database_url"] == "postgresql://user:***@localhost:5432/mydb"


def test_update_settings_changes_values(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mydb")
    api.settings = Settings()
    api.service = api.RagService(api.settings)

    client = TestClient(api.app)
    response = client.put(
        "/settings",
        json={"top_k": 42, "database_url": "postgresql://newuser:newpass@db:5432/app"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["top_k"] == 42
    assert payload["database_url"] == "postgresql://newuser:***@db:5432/app"
