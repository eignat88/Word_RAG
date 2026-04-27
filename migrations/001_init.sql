CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_base (
    id BIGSERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    fd_number TEXT,
    section TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS knowledge_base_fd_number_idx ON knowledge_base(fd_number);
CREATE INDEX IF NOT EXISTS knowledge_base_section_idx ON knowledge_base(section);

-- For larger datasets consider ivfflat/hnsw index after enough rows are loaded.
