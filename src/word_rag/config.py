from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    storage_backend: str = field(default_factory=lambda: os.getenv("STORAGE_BACKEND", "sqlite"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"))
    pg_schema: str = field(default_factory=lambda: os.getenv("PG_SCHEMA", "ai"))
    pg_documents_table: str = field(default_factory=lambda: os.getenv("PG_DOCUMENTS_TABLE", "fd_documents"))
    pg_chunks_table: str = field(default_factory=lambda: os.getenv("PG_CHUNKS_TABLE", "fd_chunks"))
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", "./word_rag.db"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "nomic-embed-text")))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3"))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    chunk_min_chars: int = field(default_factory=lambda: int(os.getenv("CHUNK_MIN_CHARS", "300")))
    chunk_max_chars: int = field(default_factory=lambda: int(os.getenv("CHUNK_MAX_CHARS", "1000")))
    index_min_chars: int = field(default_factory=lambda: int(os.getenv("INDEX_MIN_CHARS", "100")))
    embed_timeout_sec: float = field(default_factory=lambda: float(os.getenv("EMBED_TIMEOUT_SEC", "60")))
    llm_timeout_sec: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SEC", "300")))
    ai_provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "ollama"))
    openai_compat_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1"))
    openai_compat_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_COMPAT_API_KEY", ""))
    openai_compat_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_COMPAT_EMBEDDING_MODEL", os.getenv("EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))))
    openai_compat_llm_model: str = field(default_factory=lambda: os.getenv("OPENAI_COMPAT_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")))
