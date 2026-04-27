from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    storage_backend: str = field(default_factory=lambda: os.getenv("STORAGE_BACKEND", "sqlite"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/word_rag"))
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", "./word_rag.db"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "nomic-embed-text")))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3"))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    chunk_min_chars: int = field(default_factory=lambda: int(os.getenv("CHUNK_MIN_CHARS", "300")))
    chunk_max_chars: int = field(default_factory=lambda: int(os.getenv("CHUNK_MAX_CHARS", "1000")))
