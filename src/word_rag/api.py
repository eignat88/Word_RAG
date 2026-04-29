from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .rag_service import RagService


app = FastAPI(title="Word RAG API", version="0.1.0")
settings = Settings()
service = RagService(settings)


class IngestRequest(BaseModel):
    directory: str = Field(..., description="Path to directory with .docx files")
    replace: bool = True


class SearchRequest(BaseModel):
    question: str
    top_k: int | None = None
    fd_number: str | None = None
    section: str | None = None


class AskRequest(SearchRequest):
    pass


class SettingsUpdateRequest(BaseModel):
    top_k: int = Field(..., ge=1, le=1000)
    index_min_chars: int = Field(..., ge=1, le=100000)
    chunk_min_chars: int = Field(..., ge=1, le=100000)
    chunk_max_chars: int = Field(..., ge=1, le=100000)
    ollama_base_url: str
    embedding_model: str
    llm_model: str

    @field_validator("ollama_base_url", "embedding_model", "llm_model")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Поле не может быть пустым")
        return cleaned

    @field_validator("chunk_max_chars")
    @classmethod
    def validate_chunk_bounds(cls, value: int, info):
        chunk_min_chars = info.data.get("chunk_min_chars")
        if chunk_min_chars is not None and value < chunk_min_chars:
            raise ValueError("CHUNK_MAX_CHARS должен быть больше или равен CHUNK_MIN_CHARS")
        return value


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Word RAG</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            background: #f6f7f9;
            color: #222;
        }
        .card {
            background: white;
            padding: 24px;
            border-radius: 14px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        .tab-btn {
            margin-top: 0;
            background: #e9ecef;
            border: none;
            border-radius: 8px;
        }
        .tab-btn.active {
            background: #0d6efd;
            color: #fff;
        }
        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
        }
        label {
            display: block;
            margin-top: 12px;
            margin-bottom: 6px;
            font-weight: bold;
        }
        input {
            width: 100%;
            padding: 10px;
            font-size: 15px;
            box-sizing: border-box;
        }
        .error {
            margin-top: 12px;
            color: #c92a2a;
            white-space: pre-wrap;
        }
        .success {
            margin-top: 12px;
            color: #2b8a3e;
        }
        textarea {
            width: 100%;
            height: 110px;
            font-size: 16px;
            padding: 12px;
            box-sizing: border-box;
        }
        button {
            margin-top: 12px;
            padding: 12px 22px;
            font-size: 16px;
            cursor: pointer;
        }
        #answer {
            margin-top: 24px;
            white-space: pre-wrap;
            background: #f1f3f5;
            padding: 16px;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>База знаний по ФД</h1>

        <div class="tabs">
            <button class="tab-btn active" data-tab="chat" onclick="switchTab('chat', this)">Чат</button>
            <button class="tab-btn" data-tab="upload" onclick="switchTab('upload', this)">Загрузка документов</button>
            <button class="tab-btn" data-tab="settings" onclick="switchTab('settings', this)">Настройки</button>
        </div>

        <div id="tab-chat" class="tab-panel active">
            <textarea id="question" placeholder="Напиши вопрос по документам..."></textarea>
            <br>
            <button onclick="ask()">Спросить</button>
            <div id="answer"></div>
        </div>

        <div id="tab-upload" class="tab-panel">
            <p>Для загрузки документов используйте API-эндпоинт <code>POST /ingest</code>.</p>
        </div>

        <div id="tab-settings" class="tab-panel">
            <label for="top_k">TOP_K</label>
            <input id="top_k" type="number" min="1" value="10">

            <label for="index_min_chars">INDEX_MIN_CHARS</label>
            <input id="index_min_chars" type="number" min="1" value="100">

            <label for="chunk_min_chars">CHUNK_MIN_CHARS</label>
            <input id="chunk_min_chars" type="number" min="1" value="300">

            <label for="chunk_max_chars">CHUNK_MAX_CHARS</label>
            <input id="chunk_max_chars" type="number" min="1" value="1000">

            <label for="ollama_base_url">OLLAMA_BASE_URL</label>
            <input id="ollama_base_url" type="text" value="http://localhost:11434">

            <label for="embedding_model">EMBEDDING_MODEL</label>
            <input id="embedding_model" type="text" value="nomic-embed-text">

            <label for="llm_model">LLM_MODEL</label>
            <input id="llm_model" type="text" value="llama3">

            <button onclick="saveSettings()">Сохранить настройки</button>
            <div id="settings-error" class="error"></div>
            <div id="settings-success" class="success"></div>
        </div>
    </div>

    <script>
        function switchTab(tabName, btn) {
            document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach((tabBtn) => tabBtn.classList.remove('active'));
            document.getElementById(`tab-${tabName}`).classList.add('active');
            btn.classList.add('active');
        }

        function parsePositiveInt(inputId, label) {
            const raw = document.getElementById(inputId).value;
            const value = Number(raw);
            if (!Number.isInteger(value) || value <= 0) {
                throw new Error(`${label}: введите целое число больше 0`);
            }
            return value;
        }

        async function saveSettings() {
            const errorBlock = document.getElementById('settings-error');
            const successBlock = document.getElementById('settings-success');
            errorBlock.innerText = '';
            successBlock.innerText = '';

            try {
                const payload = {
                    top_k: parsePositiveInt('top_k', 'TOP_K'),
                    index_min_chars: parsePositiveInt('index_min_chars', 'INDEX_MIN_CHARS'),
                    chunk_min_chars: parsePositiveInt('chunk_min_chars', 'CHUNK_MIN_CHARS'),
                    chunk_max_chars: parsePositiveInt('chunk_max_chars', 'CHUNK_MAX_CHARS'),
                    ollama_base_url: document.getElementById('ollama_base_url').value.trim(),
                    embedding_model: document.getElementById('embedding_model').value.trim(),
                    llm_model: document.getElementById('llm_model').value.trim(),
                };

                if (payload.chunk_max_chars < payload.chunk_min_chars) {
                    throw new Error('CHUNK_MAX_CHARS должен быть больше или равен CHUNK_MIN_CHARS');
                }

                if (!payload.ollama_base_url || !payload.embedding_model || !payload.llm_model) {
                    throw new Error('Текстовые поля не должны быть пустыми');
                }

                const response = await fetch('/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail ? JSON.stringify(data.detail) : 'Ошибка сохранения настроек');
                }

                successBlock.innerText = 'Настройки сохранены';
            } catch (err) {
                errorBlock.innerText = err.message || 'Не удалось сохранить настройки';
            }
        }

        async function ask() {
            const question = document.getElementById("question").value;
            const answerBlock = document.getElementById("answer");

            if (!question.trim()) {
                answerBlock.innerText = "Введите вопрос";
                return;
            }

            answerBlock.innerText = "Ищу ответ...";

            const response = await fetch("/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();
            answerBlock.innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict:
    return service.ingest_directory(payload.directory, replace=payload.replace)


@app.post("/search")
def search(payload: SearchRequest) -> dict:
    results = service.search(
        question=payload.question,
        top_k=payload.top_k,
        fd_number=payload.fd_number,
        section=payload.section,
    )
    return {
        "results": [
            {
                "id": r.id,
                "document_name": r.document_name,
                "fd_number": r.fd_number,
                "section": r.section,
                "chunk_text": r.chunk_text,
                "distance": r.distance,
            }
            for r in results
        ]
    }


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    return service.answer(
        question=payload.question,
        top_k=payload.top_k,
        fd_number=payload.fd_number,
        section=payload.section,
    )


@app.post("/settings")
def update_settings(payload: SettingsUpdateRequest) -> dict[str, str]:
    global settings, service

    settings = Settings(
        storage_backend=settings.storage_backend,
        database_url=settings.database_url,
        pg_schema=settings.pg_schema,
        pg_documents_table=settings.pg_documents_table,
        pg_chunks_table=settings.pg_chunks_table,
        sqlite_path=settings.sqlite_path,
        ollama_base_url=payload.ollama_base_url,
        embedding_model=payload.embedding_model,
        llm_model=payload.llm_model,
        top_k=payload.top_k,
        chunk_min_chars=payload.chunk_min_chars,
        chunk_max_chars=payload.chunk_max_chars,
        index_min_chars=payload.index_min_chars,
        embed_timeout_sec=settings.embed_timeout_sec,
        llm_timeout_sec=settings.llm_timeout_sec,
    )
    service = RagService(settings)
    return {"status": "ok"}
