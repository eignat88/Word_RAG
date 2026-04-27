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
uvicorn word_rag.api:app --reload
```

### 4) Индексация документов
```bash
python -m word_rag.main ingest ./docs_fd
```

## CLI примеры

Поиск:
```bash
python -m word_rag.main search "где используется WMS_IsOversizedItemIM?" --top-k 5
```

Ответ:
```bash
python -m word_rag.main ask "как работает алгоритм закрытия ячеек?"
```

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
psql "$DATABASE_URL" -f migrations/001_init.sql
```

После миграции используются:
- БД: `postgres`
- Схема: `ai`
- Таблицы: `ai.ai_fd_documents`, `ai.ai_fd_chunks`

`docker-compose.yml` оставлен как необязательный вариант для тех, у кого Docker есть.

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
ollama pull llama3.1:8b
```

### 3) Запустить API (по умолчанию SQLite backend)
```bash
uvicorn word_rag.api:app --reload
```

### 4) Индексация документов
```bash
python -m word_rag.main ingest ./docs_fd
```

## CLI примеры

Поиск:
```bash
python -m word_rag.main search "где используется WMS_IsOversizedItemIM?" --top-k 5
```

Ответ:
```bash
python -m word_rag.main ask "как работает алгоритм закрытия ячеек?"
```

## API эндпоинты
- `GET /health`
- `POST /ingest`
- `POST /search`
- `POST /ask`

## Ограничения текущего этапа
- Начальная реализация (MVP foundation).
- Требуется локально запущенный Ollama с доступными моделями embedding/LLM.

## Опционально: PostgreSQL + pgvector (если нужен production-like режим)
Если PostgreSQL уже установлен локально (без Docker), можно переключиться на него:

```bash
export STORAGE_BACKEND=postgres
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/word_rag
psql "$DATABASE_URL" -f migrations/001_init.sql
```

`docker-compose.yml` оставлен как необязательный вариант для тех, у кого Docker есть.
