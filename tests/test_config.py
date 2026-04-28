import os

from word_rag.config import Settings


def test_settings_support_embed_model_alias(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("LLM_MODEL", "llama3")
    settings = Settings()
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.llm_model == "llama3"


def test_settings_fallback_to_embedding_model(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "legacy-model")
    settings = Settings()
    assert settings.embedding_model == "legacy-model"
