"""Web server: a chat page plus the API behind it.

Endpoints:
    GET  /           the chat page (app/static/index.html)
    POST /api/chat   {"question": "...", "history": [...]} -> the answer as a
                     stream of Server-Sent Events (SSE), see answer_stream()
    GET  /health     is the server able to work? (database + Ollama)

The server keeps no conversation state: the browser holds the history and
sends it with every question. Resetting the conversation is just the browser
forgetting it. This keeps the server simple, and any number of server copies
can answer any request (useful once it is deployed).

Two ways of running the app:
Development (app runs on my PC's venv):
    docker compose up -d db
    uvicorn main:app --app-dir app --reload --port 8080
    then open http://127.0.0.1:8000

Packaged version (app runs on the container):
    docker compose up -d --build
    then open http://127.0.0.1:8000
"""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

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
    log.info(
        "Ready. LLM: %s at %s, context window %d tokens, prompt %s",
        llm.LLM_MODEL, llm.LLM_BASE_URL, llm.NUM_CTX, llm.SYSTEM_PROMPT_FILE.name,
    )
    yield


app = FastAPI(title="TFM Chatbot", lifespan=lifespan)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    # Validation happens before our code runs: an empty or huge question, or a
    # malformed history, is rejected with a 422 error automatically.
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=200)


def sse(event: str, data) -> str:
    """One Server-Sent Event: a named message followed by a blank line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def answer_stream(question: str, history: list[dict]) -> Iterator[str]:
    """The RAG round trip as a sequence of events:

    route     {"search": bool, "query": str}: whether the documents are searched
              and with what query (sent only when there is history to route on)
    excerpts  the 5 retrieved chunks (sent before the model starts writing;
              an empty list when no search was needed)
    token     a piece of the answer, as soon as the model writes it
    done      citations, refusal flag, context window usage and timings
    error     something failed; the message says what (the server keeps running)
    """
    start = time.perf_counter()
    try:
        route = llm.route_question(question, history)
    except llm.LLMError as e:
        log.error("Routing failed: %s", e)
        yield sse("error", {"message": str(e)})
        return
    if history:
        yield sse("route", route)

    chunks = []
    if route["search"]:
        try:
            chunks = retrieval.search(route["query"], top_k=llm.TOP_K)
        except (retrieval.RetrievalError, psycopg.OperationalError) as e:
            log.error("Retrieval failed: %s", e)
            yield sse("error", {"message": f"Search failed: {e}"})
            return

    yield sse("excerpts", [
        {"n": i, "source": c["source"], "pages": c["pages"], "preview": c["content"][:400]}
        for i, c in enumerate(chunks, start=1)
    ])

    messages, dropped = llm.build_messages(question, chunks, history)
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
    context = llm.context_usage(stats)
    total = time.perf_counter() - start
    log.info(
        "Answered in %.1f s, %s, context %d/%d tokens (%.0f%%), %d history messages: %s",
        total, "searched" if route["search"] else "no search", context["used"],
        context["limit"], context["percent"], len(history), question[:80],
    )
    yield sse("done", {
        "cited": [c["n"] for c in cited],
        "invalid_citations": invalid,
        "refused": llm.REFUSAL.lower() in answer.lower(),
        "searched": route["search"],
        "context": context,
        "history_dropped": dropped,
        "seconds": round(total, 2),
        "first_token_seconds": round(first_token or 0, 2),
        "output_tokens": stats.get("output_tokens", 0),
    })


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    history = [m.model_dump() for m in request.history]
    return StreamingResponse(
        answer_stream(request.question.strip(), history),
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