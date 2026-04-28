# Word_RAG

Локальная RAG-система по Word (`.docx`) документам:
- ingestion документов;
- чанкинг по разделам;
- embeddings через Ollama;
- хранение в PostgreSQL + pgvector;
- семантический поиск и генерация ответа с источниками.

## Структура проекта
- `src/word_rag/` — код сервиса (парсинг, чанкинг, поиск, API).
- `migrations/001_init.sql` — инициализация схемы БД.
- `docs/functional_design_ru.md` — функциональный дизайн.
- `docs/jira_breakdown.md` — Epic/Task breakdown.
- `tests/` — базовые unit-тесты.

## Быстрый старт (без Docker)

### 1) Установить зависимости
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2) Запустить Ollama и скачать модели
```bash
ollama pull nomic-embed-text
ollama pull llama3
```

### 3) Запустить API (по умолчанию SQLite backend)
```bash
python -m uvicorn src.word_rag.api:app --reload --host 127.0.0.1 --port 8010
```

После запуска:
- UI чат: `http://127.0.0.1:8010/`
- Swagger: `http://127.0.0.1:8010/docs`

### 4) Индексация документов
```bash
python -m word_rag.main ingest ./docs_fd
```

Во время индексации автоматически отфильтровываются мусорные чанки (например `Нет.`, `-`, слишком короткие фрагменты). Минимальный порог задается `INDEX_MIN_CHARS` (по умолчанию `100`).
CLI показывает прогресс по файлам в консоли: старт, обработка каждого документа, количество вставленных/пропущенных чанков и общее время.

## CLI примеры

Поиск:
```bash
python -m word_rag.main search "где используется WMS_IsOversizedItemIM?" --top-k 5
```

Ответ:
```bash
python -m word_rag.main ask "как работает алгоритм закрытия ячеек?"
```

Если видите `ReadTimeout` при `ask`:
- проверьте, что Ollama запущен (`ollama list`);
- проверьте корректность модели в `LLM_MODEL`;
- увеличьте `LLM_TIMEOUT_SEC` (например, `600`) в `.env`.

## API эндпоинты
- `GET /health`
- `POST /ingest`
- `POST /search`
- `POST /ask`

## Ограничения текущего этапа
- Начальная реализация (MVP foundation).
- Требуется локально запущенный Ollama с доступными моделями embedding/LLM.
- Если у тебя уже используются переменные `EMBED_MODEL` и `LLM_MODEL`, проект их поддерживает напрямую.

## Опционально: PostgreSQL + pgvector (если нужен production-like режим)
Если PostgreSQL уже установлен локально (без Docker), можно переключиться на него:

```bash
export STORAGE_BACKEND=postgres
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
export PG_SCHEMA=ai
export PG_DOCUMENTS_TABLE=fd_documents
export PG_CHUNKS_TABLE=fd_chunks
psql "$DATABASE_URL" -f migrations/001_init.sql
```

После миграции используются:
- БД: `postgres`
- Схема: `ai`
- Таблицы: `ai.fd_documents`, `ai.fd_chunks`

Адаптация под текущую схему:
- `fd_documents.title/source_file/dax_code` используются вместо `document_name/fd_number`.
- `fd_chunks.section_title` используется вместо `section`.
- Если `embedding` имеет тип `double precision[]`, similarity считается в Python (cosine), без `pgvector`.
- Если `embedding` имеет тип `vector`, используется SQL distance (`<->`).
- После ingest дополнительно заполняются:
  - `ai.fd_entities`
  - `ai.fd_chunk_entities`
  - `ai.fd_document_links`

`docker-compose.yml` оставлен как необязательный вариант для тех, у кого Docker есть.
