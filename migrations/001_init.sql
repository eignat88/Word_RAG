CREATE SCHEMA IF NOT EXISTS ai;

CREATE TABLE IF NOT EXISTS ai.fd_documents (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    source_file TEXT NOT NULL UNIQUE,
    dax_code TEXT,
    document_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ai.fd_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES ai.fd_documents(id) ON DELETE CASCADE,
    section_title TEXT,
    section_number INTEGER,
    chunk_type TEXT,
    chunk_text TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS fd_documents_source_file_idx ON ai.fd_documents(source_file);
CREATE INDEX IF NOT EXISTS fd_documents_dax_code_idx ON ai.fd_documents(dax_code);
CREATE INDEX IF NOT EXISTS fd_chunks_document_id_idx ON ai.fd_chunks(document_id);
CREATE INDEX IF NOT EXISTS fd_chunks_section_title_idx ON ai.fd_chunks(section_title);
