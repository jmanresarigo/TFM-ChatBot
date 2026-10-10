"""Web server: a chat page plus the API behind it.

Endpoints:
    GET  /           the chat page (app/static/index.html)
    POST /api/chat   {"question": "..."} -> the answer as a stream of
                     Server-Sent Events (SSE), see answer_stream()
    GET  /health     is the server able to work? (database + Ollama)

Run from the project root:
    uvicorn main:app --app-dir app --reload
then open http://127.0.0.1:8000
"""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import llm
import retrieval

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("uvicorn.error")  # shows up in the server's console


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load the embedding model now, so the
    first user doesn't wait several seconds for it."""
    log.info("Loading embedding model %s ...", retrieval.EMBEDDING_MODEL)
    retrieval.get_model()
    log.info("Ready. LLM: %s at %s", llm.LLM_MODEL, llm.LLM_BASE_URL)
    yield


app = FastAPI(title="TFM Chatbot", lifespan=lifespan)


class ChatRequest(BaseModel):
    # Validation happens before our code runs: an empty or huge question is
    # rejected with a 422 error automatically.
    question: str = Field(min_length=1, max_length=2000)


def sse(event: str, data) -> str:
    """One Server-Sent Event: a named message followed by a blank line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def answer_stream(question: str) -> Iterator[str]:
    """The RAG round trip as a sequence of events:

    excerpts  the 5 retrieved chunks (sent first, before the model starts)
    token     a piece of the answer, as soon as the model writes it
    done      citations, refusal flag and timings, once the answer is complete
    error     something failed; the message says what (the server keeps running)
    """
    start = time.perf_counter()
    try:
        chunks = retrieval.search(question, top_k=llm.TOP_K)
    except (retrieval.RetrievalError, psycopg.OperationalError) as e:
        log.error("Retrieval failed: %s", e)
        yield sse("error", {"message": f"Search failed: {e}"})
        return

    yield sse("excerpts", [
        {"n": i, "source": c["source"], "pages": c["pages"], "preview": c["content"][:400]}
        for i, c in enumerate(chunks, start=1)
    ])

    messages = [
        {"role": "system", "content": llm.load_system_prompt()},
        {"role": "user", "content": llm.build_user_message(question, chunks)},
    ]
    stats, pieces = {}, []
    first_token = None
    try:
        for piece in llm.stream_chat(messages, stats):
            if first_token is None:
                first_token = time.perf_counter() - start
            pieces.append(piece)
            yield sse("token", {"text": piece})
    except llm.LLMError as e:
        log.error("Generation failed: %s", e)
        yield sse("error", {"message": str(e)})
        return

    answer = "".join(pieces).strip()
    cited, invalid = llm.check_citations(answer, chunks)
    total = time.perf_counter() - start
    log.info("Answered in %.1f s (%s output tokens): %s", total, stats.get("output_tokens", "?"), question[:80])
    yield sse("done", {
        "cited": [c["n"] for c in cited],
        "invalid_citations": invalid,
        "refused": llm.REFUSAL.lower() in answer.lower(),
        "seconds": round(total, 2),
        "first_token_seconds": round(first_token or 0, 2),
        "output_tokens": stats.get("output_tokens", 0),
    })


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        answer_stream(request.question.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells proxies (e.g. nginx) not to buffer: without it, a proxy may
            # hold the whole answer back and deliver it at once at the end.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health() -> JSONResponse:
    """Checks the two dependencies. 200 if both work, 503 otherwise, so that
    monitoring tools (and later Railway) can tell when the app is unusable."""
    checks = {}
    try:
        with retrieval.connect() as conn:
            count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        checks["database"] = {"ok": count > 0, "chunks": count}
    except Exception as e:  # noqa: BLE001 - report any failure, never crash
        checks["database"] = {"ok": False, "error": str(e)}
    try:
        tags = httpx.get(f"{llm.LLM_BASE_URL}/api/tags", timeout=3).json()
        models = [m["name"] for m in tags.get("models", [])]
        checks["llm"] = {"ok": llm.LLM_MODEL in models, "model": llm.LLM_MODEL}
        if llm.LLM_MODEL not in models:
            checks["llm"]["error"] = f"model not pulled; available: {models}"
    except Exception as e:  # noqa: BLE001
        checks["llm"] = {"ok": False, "error": f"Ollama unreachable at {llm.LLM_BASE_URL}: {e}"}
    healthy = all(c["ok"] for c in checks.values())
    return JSONResponse({"status": "ok" if healthy else "degraded", **checks}, status_code=200 if healthy else 503)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")