from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    storage_backend: str = os.getenv("STORAGE_BACKEND", "sqlite")
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/word_rag")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./word_rag.db")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_min_chars: int = int(os.getenv("CHUNK_MIN_CHARS", "300"))
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "1000"))
