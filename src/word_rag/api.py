from __future__ import annotations

from dataclasses import asdict
import json
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .rag_service import RagService


app = FastAPI(title="Word RAG API", version="0.1.0")
settings = Settings()
service = RagService(settings)


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
        #settings {
            margin-top: 16px;
            background: #f1f3f5;
            border-radius: 10px;
            padding: 16px;
            white-space: pre-wrap;
            overflow-x: auto;
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
            <pre id="settings"></pre>
        </div>
    </div>

    <script>
        const settingsData = __SETTINGS_JSON__;

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

        document.getElementById("settings").innerText = JSON.stringify(settingsData, null, 2);
    </script>
</body>
</html>
""".replace("__SETTINGS_JSON__", json.dumps(settings_payload, ensure_ascii=False))


@app.get("/settings")
def get_settings() -> dict:
    return _ui_settings_payload()


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
