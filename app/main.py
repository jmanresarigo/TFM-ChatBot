"""Web server: a chat page plus the API behind it.

Endpoints:
    GET  /           the chat page (app/static/index.html)
    GET  /static/... the page's scripts, styles and fonts (Markdown and math rendering)
    POST /api/chat   {"question": "...", "history": [...]} -> the answer as a
                     stream of Server-Sent Events (SSE), see answer_stream()
    GET  /health     is the server able to work? (database, embeddings API, LLM API)
    GET  /health/live is the server process running? (always 200 once started)

With APP_PASSWORD set, every page except the two health checks asks for that
password (the browser shows its own login box; any user name works).

The server keeps no conversation state: the browser holds the history and
sends it with every question. Resetting the conversation is just the browser
forgetting it. This keeps the server simple, and any number of server copies
can answer any request (useful once it is deployed).

Run from the project root:
    uvicorn main:app --app-dir app --reload
then open http://127.0.0.1:8000
"""

import base64
import binascii
import json
import logging
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import psycopg
from fastapi import FastAPI
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import llm
import retrieval

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("uvicorn.error")  # shows up in the server's console

# The IEEE reference of each document, by PDF file name, so the page can cite
# "[1] K. Fukami and K. Taira, ..." instead of a file name.
REFERENCES = {
    source: entry
    for source, entry in json.loads((Path(__file__).parent / "references.json").read_text(encoding="utf-8")).items()
    if not source.startswith("_")
}


def reference(source: str) -> dict:
    """The reference shown for a document; just its file name if it has none."""
    return REFERENCES.get(source, {"text": source})

# Empty = no password (local development). Set it on any public deployment.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
# Health checks stay open: hosting platforms and monitors call them without a password.
PUBLIC_PATHS = {"/health", "/health/live"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: log the configuration. (Nothing heavy
    to load: embeddings and the LLM are both called over HTTP.)"""
    log.info(
        "Embeddings: %s at %s (vectors of %s)",
        retrieval.EMBEDDING_API_MODEL, retrieval.EMBEDDING_BASE_URL or "(not set)", retrieval.EMBEDDING_MODEL,
    )
    log.info(
        "Ready. LLM: %s at %s (router: %s), token budget %d, prompt %s",
        llm.LLM_MODEL, llm.LLM_BASE_URL, llm.ROUTER_MODEL, llm.NUM_CTX, llm.SYSTEM_PROMPT_FILE.name,
    )
    yield


def password_ok(authorization: str) -> bool:
    """Check an "Authorization: Basic <base64 of user:password>" header."""
    if not authorization.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    _, _, password = decoded.partition(":")
    # compare_digest takes the same time whether the guess is close or not,
    # so response times give away nothing about the password.
    return secrets.compare_digest(password.encode(), APP_PASSWORD.encode())


class PasswordMiddleware:
    """Asks for APP_PASSWORD before any request reaches the app (HTTP Basic auth).

    Written as plain ASGI middleware (a wrapper around the whole app) so that
    streamed answers pass through untouched. Without a valid password the
    reply is 401 with a WWW-Authenticate header, which makes the browser show
    its login box; after logging in, the browser resends the password with
    every request, including the chat page's own fetch() calls."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not APP_PASSWORD or scope["path"] in PUBLIC_PATHS:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if password_ok(headers.get(b"authorization", b"").decode("latin-1")):
            return await self.app(scope, receive, send)
        response = PlainTextResponse(
            "Password required.",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="TFM Chatbot", charset="UTF-8"'},
        )
        await response(scope, receive, send)


app = FastAPI(title="TFM Chatbot", lifespan=lifespan)
app.add_middleware(PasswordMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    excerpts  the 5 retrieved chunks: number, document, pages and its IEEE
              reference (sent before the model starts writing; an empty list
              when no search was needed)
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
        {"n": i, "source": c["source"], "pages": c["pages"], "reference": reference(c["source"])}
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


# The API checks call external providers, so their results are reused for a
# minute: monitoring that calls /health every few seconds won't hammer them.
API_CHECK_SECONDS = 60
_llm_check: dict = {"at": float("-inf"), "result": None}
_embedding_check: dict = {"at": float("-inf"), "result": None}


def check_embeddings() -> dict:
    if time.monotonic() - _embedding_check["at"] < API_CHECK_SECONDS:
        return _embedding_check["result"]
    try:
        dims = len(retrieval.embed_query("health check"))
        result = {"ok": True, "model": retrieval.EMBEDDING_API_MODEL, "dimensions": dims}
    except retrieval.RetrievalError as e:
        result = {"ok": False, "model": retrieval.EMBEDDING_API_MODEL, "error": str(e)}
    _embedding_check.update(at=time.monotonic(), result=result)
    return result


def check_llm() -> dict:
    if time.monotonic() - _llm_check["at"] < API_CHECK_SECONDS:
        return _llm_check["result"]
    try:
        models = llm.list_models()
        needed = {llm.LLM_MODEL, llm.ROUTER_MODEL}
        missing = sorted(needed - set(models))
        result = {"ok": not missing, "model": llm.LLM_MODEL, "provider": llm.LLM_BASE_URL}
        if missing:
            result["error"] = f"model(s) not available with this key: {missing}"
    except llm.LLMError as e:
        result = {"ok": False, "provider": llm.LLM_BASE_URL, "error": str(e)}
    _llm_check.update(at=time.monotonic(), result=result)
    return result


@app.get("/health/live")
def live() -> dict:
    """Liveness: answers as soon as the server runs, without checking anything
    else. Railway calls this before switching traffic to a new deployment, so a
    temporarily unavailable API doesn't block a deploy."""
    return {"status": "alive"}


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
    checks["embeddings"] = check_embeddings()
    checks["llm"] = check_llm()
    healthy = all(c["ok"] for c in checks.values())
    return JSONResponse({"status": "ok" if healthy else "degraded", **checks}, status_code=200 if healthy else 503)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
