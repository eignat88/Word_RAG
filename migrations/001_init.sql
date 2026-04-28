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

CREATE TABLE IF NOT EXISTS ai.fd_entities (
    id BIGSERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_value TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(entity_type, entity_value)
);

CREATE TABLE IF NOT EXISTS ai.fd_chunk_entities (
    chunk_id BIGINT NOT NULL,
    entity_id BIGINT NOT NULL,
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id) REFERENCES ai.fd_chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES ai.fd_entities(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ai.fd_document_links (
    id BIGSERIAL PRIMARY KEY,
    source_document_id BIGINT NOT NULL REFERENCES ai.fd_documents(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,
    target_document_id BIGINT NOT NULL REFERENCES ai.fd_documents(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_fd_document_links
    ON ai.fd_document_links(source_document_id, link_type, target_document_id);
