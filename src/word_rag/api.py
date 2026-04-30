from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

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
            max-width: 980px;
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
            flex-wrap: wrap;
        }
        .tab-button {
            margin: 0;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid #d0d7de;
            background: #f5f7fa;
            cursor: pointer;
        }
        .tab-button.active {
            background: #1f6feb;
            color: white;
            border-color: #1f6feb;
        }
        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
        }
        label {
            display: block;
            margin: 10px 0 6px;
            font-weight: 600;
        }
        textarea, input {
            width: 100%;
            font-size: 15px;
            padding: 10px;
            box-sizing: border-box;
        }
        textarea {
            height: 110px;
        }
        button {
            margin-top: 12px;
            padding: 12px 22px;
            font-size: 16px;
            cursor: pointer;
        }
        .status {
            margin-top: 12px;
            color: #444;
            font-weight: 600;
        }
        .result {
            margin-top: 16px;
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
            <button class="tab-button active" data-tab="ask-tab" onclick="switchTab(event)">Ответ</button>
            <button class="tab-button" data-tab="search-tab" onclick="switchTab(event)">Поиск</button>
            <button class="tab-button" data-tab="ingest-tab" onclick="switchTab(event)">Загрузка документов</button>
            <button class="tab-button" data-tab="settings-tab" onclick="switchTab(event)">Настройки</button>
        </div>

        <div id="ask-tab" class="tab-panel active">
            <label for="question">Вопрос</label>
            <textarea id="question" placeholder="Введите вопрос по документам..."></textarea>
            <button onclick="ask()">Спросить</button>
            <div id="ask-status" class="status"></div>
            <div id="ask-result" class="result"></div>
        </div>

        <div id="search-tab" class="tab-panel">
            <label for="search-question">Запрос для поиска</label>
            <textarea id="search-question" placeholder="Введите запрос..."></textarea>
            <button onclick="searchDocs()">Найти</button>
            <div id="search-status" class="status"></div>
            <div id="search-result" class="result"></div>
        </div>

        <div id="ingest-tab" class="tab-panel">
            <label for="directory">Путь к папке с документами</label>
            <input id="directory" placeholder="Например: /data/docs" />
            <button onclick="ingestDocs()">Загрузка документов</button>
            <div id="ingest-status" class="status"></div>
            <div id="ingest-result" class="result"></div>
        </div>

        <div id="settings-tab" class="tab-panel">
            <label for="fd-number">ФД номер (фильтр)</label>
            <input id="fd-number" placeholder="Например: ФД-123" />
            <label for="section">Раздел (фильтр)</label>
            <input id="section" placeholder="Например: Общие положения" />
            <label for="top-k">Количество результатов</label>
            <input id="top-k" type="number" min="1" value="5" />
            <button onclick="saveSettings()">Сохранить</button>
            <div id="settings-status" class="status"></div>
        </div>
    </div>

    <script>
        const appSettings = {
            fd_number: "",
            section: "",
            top_k: 5,
        };

        function switchTab(event) {
            const tabId = event.target.dataset.tab;
            document.querySelectorAll(".tab-button").forEach((btn) => btn.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
            event.target.classList.add("active");
            document.getElementById(tabId).classList.add("active");
        }

        function showStatus(elementId, text) {
            document.getElementById(elementId).innerText = text;
        }

        function setJsonResult(elementId, data) {
            document.getElementById(elementId).innerText = JSON.stringify(data, null, 2);
        }

        async function ask() {
            const question = document.getElementById("question").value;
            if (!question.trim()) {
                showStatus("ask-status", "Ошибка: введите вопрос.");
                return;
            }

            showStatus("ask-status", "Идет обработка...");
            try {
                const response = await fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ question, ...appSettings })
                });

                if (!response.ok) {
                    showStatus("ask-status", "Ошибка: не удалось получить ответ.");
                    return;
                }

                const data = await response.json();
                const sourcesCount = Array.isArray(data.sources) ? data.sources.length : 0;
                document.getElementById("ask-result").innerText = data.answer || "Ответ отсутствует.";
                showStatus("ask-status", `Готово. Найдено источников: ${sourcesCount}.`);
            } catch {
                showStatus("ask-status", "Ошибка: проблема соединения с сервером.");
            }
        }

        async function searchDocs() {
            const question = document.getElementById("search-question").value;
            if (!question.trim()) {
                showStatus("search-status", "Ошибка: введите запрос для поиска.");
                return;
            }
            showStatus("search-status", "Идет обработка...");
            try {
                const response = await fetch("/search", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ question, ...appSettings })
                });
                if (!response.ok) {
                    showStatus("search-status", "Ошибка: не удалось выполнить поиск.");
                    return;
                }
                const data = await response.json();
                setJsonResult("search-result", data);
                showStatus("search-status", "Готово.");
            } catch {
                showStatus("search-status", "Ошибка: проблема соединения с сервером.");
            }
        }

        async function ingestDocs() {
            const directory = document.getElementById("directory").value;
            if (!directory.trim()) {
                showStatus("ingest-status", "Ошибка: укажите путь к папке.");
                return;
            }
            showStatus("ingest-status", "Идет обработка...");
            try {
                const response = await fetch("/ingest", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ directory, replace: true })
                });
                if (!response.ok) {
                    showStatus("ingest-status", "Ошибка: не удалось загрузить документы.");
                    return;
                }
                const data = await response.json();
                setJsonResult("ingest-result", data);
                showStatus("ingest-status", "Готово.");
            } catch {
                showStatus("ingest-status", "Ошибка: проблема соединения с сервером.");
            }
        }

        function saveSettings() {
            appSettings.fd_number = document.getElementById("fd-number").value.trim();
            appSettings.section = document.getElementById("section").value.trim();
            appSettings.top_k = Number(document.getElementById("top-k").value) || 5;
            showStatus("settings-status", "Настройки сохранены. Готово.");
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
