from __future__ import annotations

from dataclasses import asdict
import json
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .rag_service import RagService


app = FastAPI(title="Word RAG API", version="0.1.0")
settings = Settings()
service = RagService(settings)

EDITABLE_SETTINGS_FIELDS = {
    "storage_backend",
    "database_url",
    "pg_schema",
    "pg_documents_table",
    "pg_chunks_table",
    "sqlite_path",
    "ollama_base_url",
    "embedding_model",
    "llm_model",
    "top_k",
    "chunk_min_chars",
    "chunk_max_chars",
    "index_min_chars",
    "embed_timeout_sec",
    "llm_timeout_sec",
}


def _mask_database_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    if not parsed.password:
        return database_url

    username = parsed.username or ""
    host = parsed.hostname or ""
    port_part = f":{parsed.port}" if parsed.port else ""
    credentials = f"{username}:***" if username else "***"
    netloc = f"{credentials}@{host}{port_part}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _ui_settings_payload() -> dict:
    payload = asdict(settings)
    payload["database_url"] = _mask_database_url(settings.database_url)
    return payload


def _parse_settings_value(key: str, value: object) -> object:
    current_value = getattr(settings, key)
    if isinstance(current_value, int):
        return int(value)
    if isinstance(current_value, float):
        return float(value)
    return str(value)


def _apply_settings_patch(changes: dict[str, object]) -> dict:
    global settings, service
    unknown = sorted(set(changes.keys()) - EDITABLE_SETTINGS_FIELDS)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown settings fields: {', '.join(unknown)}")

    next_settings_dict = asdict(settings)
    for key, value in changes.items():
        try:
            next_settings_dict[key] = _parse_settings_value(key, value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid value for {key}: {value}") from exc

    settings = Settings(**next_settings_dict)
    service = RagService(settings)
    return _ui_settings_payload()


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


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    settings_payload = _ui_settings_payload()
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
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        .tab-btn {
            margin-top: 0;
            border: 1px solid #c9ced6;
            background: #f1f3f5;
            border-radius: 8px;
            padding: 10px 16px;
        }
        .tab-btn.active {
            background: #0d6efd;
            color: #fff;
            border-color: #0d6efd;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        #settings-table {
            margin-top: 16px;
            background: #f1f3f5;
            border-radius: 10px;
            padding: 16px;
            overflow-x: auto;
            width: 100%;
            border-collapse: collapse;
        }
        #settings-table td, #settings-table th {
            border-bottom: 1px solid #d9dee5;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }
        #settings-table input {
            width: 100%;
            box-sizing: border-box;
            padding: 8px;
        }
        #settings-status {
            margin-top: 12px;
            color: #444;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>База знаний по ФД</h1>
        <div class="tabs">
            <button class="tab-btn active" data-tab="questions" onclick="switchTab(event)">Вопросы</button>
            <button class="tab-btn" data-tab="settings" onclick="switchTab(event)">Настройка</button>
        </div>

        <div id="tab-questions" class="tab-content active">
            <textarea id="question" placeholder="Напиши вопрос по документам..."></textarea>
            <br>
            <button onclick="ask()">Спросить</button>

            <div id="answer"></div>
        </div>

        <div id="tab-settings" class="tab-content">
            <table id="settings-table">
                <thead>
                    <tr><th>Key</th><th>Value</th></tr>
                </thead>
                <tbody id="settings-body"></tbody>
            </table>
            <button onclick="saveSettings()">Сохранить настройки</button>
            <div id="settings-status"></div>
        </div>
    </div>

    <script>
        let settingsData = __SETTINGS_JSON__;

        function switchTab(event) {
            const tab = event.currentTarget.getAttribute("data-tab");
            document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
            event.currentTarget.classList.add("active");

            document.querySelectorAll(".tab-content").forEach((content) => content.classList.remove("active"));
            document.getElementById(`tab-${tab}`).classList.add("active");
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

        function renderSettings() {
            const tbody = document.getElementById("settings-body");
            tbody.innerHTML = "";
            Object.entries(settingsData).forEach(([key, value]) => {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${key}</td>
                    <td><input data-key="${key}" value="${String(value).replaceAll('"', "&quot;")}"></td>
                `;
                tbody.appendChild(row);
            });
        }

        async function saveSettings() {
            const payload = {};
            document.querySelectorAll("#settings-body input").forEach((el) => {
                payload[el.getAttribute("data-key")] = el.value;
            });

            const status = document.getElementById("settings-status");
            status.innerText = "Сохраняю...";
            const response = await fetch("/settings", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok) {
                status.innerText = data.detail || "Ошибка сохранения";
                return;
            }
            settingsData = data;
            renderSettings();
            status.innerText = "Настройки сохранены";
        }

        renderSettings();
    </script>
</body>
</html>
""".replace("__SETTINGS_JSON__", json.dumps(settings_payload, ensure_ascii=False))


@app.get("/settings")
def get_settings() -> dict:
    return _ui_settings_payload()


@app.put("/settings")
def update_settings(payload: dict[str, object]) -> dict:
    return _apply_settings_patch(payload)


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
