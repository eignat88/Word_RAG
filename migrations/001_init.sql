CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS ai;

CREATE TABLE IF NOT EXISTS ai.ai_fd_documents (
    id BIGSERIAL PRIMARY KEY,
    document_name TEXT NOT NULL UNIQUE,
    fd_number TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ai.ai_fd_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES ai.ai_fd_documents(id) ON DELETE CASCADE,
    section TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ai_fd_documents_fd_number_idx ON ai.ai_fd_documents(fd_number);
CREATE INDEX IF NOT EXISTS ai_fd_chunks_section_idx ON ai.ai_fd_chunks(section);
CREATE INDEX IF NOT EXISTS ai_fd_chunks_document_id_idx ON ai.ai_fd_chunks(document_id);

-- For larger datasets consider ivfflat/hnsw index after enough rows are loaded.
