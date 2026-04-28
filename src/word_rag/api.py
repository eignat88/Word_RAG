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
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Word RAG Chat</title>
  <style>
    :root { --bg:#0f172a; --card:#111827; --muted:#94a3b8; --text:#e5e7eb; --accent:#22c55e; --border:#1f2937; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, Segoe UI, system-ui, sans-serif; background:linear-gradient(180deg,#0b1220,#0f172a); color:var(--text); }
    .wrap { max-width:1100px; margin:24px auto; padding:0 16px; }
    .title { font-size:28px; font-weight:700; margin-bottom:4px; }
    .subtitle { color:var(--muted); margin-bottom:16px; }
    .grid { display:grid; grid-template-columns: 320px 1fr; gap:16px; }
    .card { background:rgba(17,24,39,.85); border:1px solid var(--border); border-radius:14px; padding:14px; backdrop-filter: blur(4px); }
    .card h3 { margin:0 0 10px 0; font-size:16px; }
    label { display:block; margin:10px 0 6px; color:var(--muted); font-size:13px; }
    input, textarea { width:100%; background:#0b1220; color:var(--text); border:1px solid #263244; border-radius:10px; padding:10px; }
    textarea { min-height:90px; resize:vertical; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    button { width:100%; border:0; border-radius:10px; padding:10px; margin-top:10px; font-weight:600; cursor:pointer; }
    .btn-primary { background:var(--accent); color:#06250f; }
    .btn-dark { background:#1f2937; color:var(--text); }
    .status { margin-top:8px; font-size:13px; color:var(--muted); white-space:pre-wrap; }
    .answer { white-space:pre-wrap; line-height:1.5; }
    .source { border:1px solid #243145; border-radius:10px; padding:10px; margin-top:8px; background:#0b1220; }
    .src-meta { color:#9ca3af; font-size:12px; margin-bottom:4px; }
    @media (max-width:900px){ .grid{grid-template-columns:1fr;} }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">💬 Word RAG Chat</div>
    <div class="subtitle">Красивый интерфейс для ingest / search / ask без Swagger</div>

    <div class="grid">
      <div class="card">
        <h3>Индексация</h3>
        <label>Папка с .docx</label>
        <input id="directory" value="./docs_fd" />
        <label><input id="replace" type="checkbox" checked /> Перезаписывать документ</label>
        <button class="btn-primary" onclick="runIngest()">Запустить ingest</button>
        <div id="ingestStatus" class="status"></div>
      </div>

      <div class="card">
        <h3>Поиск и ответы</h3>
        <label>Вопрос</label>
        <textarea id="question" placeholder="Например: где используется WMS_IsOversizedItemIM?"></textarea>
        <div class="row">
          <div>
            <label>fd_number (опц.)</label>
            <input id="fdNumber" placeholder="DAX-7407" />
          </div>
          <div>
            <label>section (опц.)</label>
            <input id="section" placeholder="Алгоритмы" />
          </div>
        </div>
        <label>top_k</label>
        <input id="topK" type="number" value="5" min="1" max="20" />

        <div class="row">
          <button class="btn-dark" onclick="runSearch()">Только поиск</button>
          <button class="btn-primary" onclick="runAsk()">Спросить LLM</button>
        </div>

        <div id="queryStatus" class="status"></div>
        <h3 style="margin-top:14px;">Ответ</h3>
        <div id="answer" class="answer"></div>
        <h3 style="margin-top:14px;">Источники</h3>
        <div id="sources"></div>
      </div>
    </div>
  </div>

<script>
  const q = (id) => document.getElementById(id);

  function basePayload() {
    return {
      question: q('question').value,
      top_k: Number(q('topK').value) || 5,
      fd_number: q('fdNumber').value || null,
      section: q('section').value || null
    };
  }

  async function runIngest() {
    q('ingestStatus').textContent = '⏳ Индексация запущена...';
    const payload = { directory: q('directory').value, replace: q('replace').checked };
    const res = await fetch('/ingest', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();
    q('ingestStatus').textContent = res.ok
      ? `✅ Готово: documents=${data.documents}, chunks=${data.chunks}, skipped=${data.skipped_chunks}, elapsed=${data.elapsed_sec}s`
      : `❌ Ошибка: ${JSON.stringify(data)}`;
  }

  function renderSources(items) {
    const root = q('sources');
    root.innerHTML = '';
    (items || []).forEach(s => {
      const div = document.createElement('div');
      div.className = 'source';
      div.innerHTML = `<div class="src-meta">${s.document_name} | ${s.section} | score=${Number(s.distance).toFixed(4)}</div><div>${(s.chunk_text || '').replace(/</g,'&lt;')}</div>`;
      root.appendChild(div);
    });
  }

  async function runSearch() {
    q('queryStatus').textContent = '⏳ Поиск...';
    q('answer').textContent = '';
    const res = await fetch('/search', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(basePayload()) });
    const data = await res.json();
    if (!res.ok) {
      q('queryStatus').textContent = `❌ Ошибка: ${JSON.stringify(data)}`;
      return;
    }
    q('queryStatus').textContent = `✅ Найдено: ${(data.results || []).length}`;
    renderSources(data.results || []);
  }

  async function runAsk() {
    q('queryStatus').textContent = '⏳ Генерация ответа...';
    q('answer').textContent = '';
    const res = await fetch('/ask', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(basePayload()) });
    const data = await res.json();
    if (!res.ok) {
      q('queryStatus').textContent = `❌ Ошибка: ${JSON.stringify(data)}`;
      return;
    }
    q('queryStatus').textContent = '✅ Ответ получен';
    q('answer').textContent = data.answer || '';
    renderSources(data.sources || []);
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
