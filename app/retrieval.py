"""Find the chunks that best match a question: the "R" in RAG.

Two searches run on the same table and see the question differently:

    Vector search   compares meanings. Finds "how does the neuron forget its
                    input?" -> a passage about the membrane decay factor,
                    even with no shared words. Weak on exact terms.
    Keyword search  compares words (stemmed: "gradients" = "gradient").
                    Finds "STDP", "NACA 0012", author names exactly.
                    Blind to synonyms and paraphrases.

Hybrid search runs both and merges the two rankings with Reciprocal Rank
Fusion (RRF): a chunk gets 1 / (RRF_K + rank) points from each list it
appears in, and chunks are sorted by their total. Ranks are used instead of
raw scores because the two searches score on completely different scales.

The question is turned into a vector by an embeddings API (OpenAI-compatible:
Cloudflare Workers AI by default, or a local Ollama), running the SAME model
that embedded the chunks during ingestion. The app therefore needs no PyTorch:
it stays small enough for a free hosting tier.

This module is used by the chatbot. It can also be run directly to inspect
results:
    python app/retrieval.py "What is the membrane decay factor?"
    python app/retrieval.py "surrogate gradient" --mode keyword
"""

import argparse
import os
import sys
import time

import httpx
import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

# The model whose vectors are stored in the database (written by ingestion).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# Where questions are embedded, and the provider's name for that same model.
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", "@cf/qwen/qwen3-embedding-0.6b")
# Qwen3-Embedding expects questions (not documents) to start with a task
# instruction. This is the exact prompt sentence-transformers' encode_query()
# adds for this model, so API vectors match the local ones.
QUERY_INSTRUCTION = os.getenv(
    "EMBEDDING_QUERY_INSTRUCTION",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
)

CANDIDATES = 20  # how many results each search contributes before fusion
RRF_K = 60  # standard RRF constant: dampens the gap between rank 1 and 2

_client = httpx.Client(timeout=httpx.Timeout(10, read=30))


class RetrievalError(Exception):
    """A problem the user can fix (missing config, empty database, API down...).

    Raised instead of SystemExit so a web server can turn it into an error
    response instead of shutting down. The command line converts it below."""


def embed_query(question: str) -> np.ndarray:
    """The question as a unit-length vector, from the embeddings API.

    POST {EMBEDDING_BASE_URL}/embeddings {"model": ..., "input": [text]}
    -> {"data": [{"embedding": [...]}]}, the OpenAI format. One retry on 429.
    """
    if not EMBEDDING_BASE_URL:
        raise RetrievalError("EMBEDDING_BASE_URL is not set. Add it to your .env file.")
    headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}"} if EMBEDDING_API_KEY else {}
    payload = {"model": EMBEDDING_API_MODEL, "input": [QUERY_INSTRUCTION + question]}
    url = f"{EMBEDDING_BASE_URL}/embeddings"
    for attempt in range(2):
        try:
            response = _client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            raise RetrievalError(f"Cannot reach the embeddings API at {EMBEDDING_BASE_URL}: {e}") from None
        if response.status_code == 429 and attempt == 0:
            time.sleep(min(float(response.headers.get("retry-after", "1") or 1), 5))
            continue
        break
    if response.status_code == 401 or response.status_code == 403:
        raise RetrievalError("The embeddings API rejected the key. Check EMBEDDING_API_KEY in .env.")
    if response.status_code != 200:
        raise RetrievalError(f"Embeddings API returned {response.status_code}: {response.text[:300]}")
    vector = np.asarray(response.json()["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise RetrievalError("The embeddings API returned an all-zero vector.")
    return vector / norm


def connect() -> psycopg.Connection:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RetrievalError("DATABASE_URL is not set. Add it to your .env file.")
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


def check_model(conn: psycopg.Connection) -> None:
    """Refuse to search if the table was filled with a different model."""
    row = conn.execute(
        "SELECT value FROM ingest_info WHERE key = 'embedding_model'"
    ).fetchone()
    if row is None:
        raise RetrievalError("The database is empty. Run the ingestion pipeline first.")
    if row[0] != EMBEDDING_MODEL:
        raise RetrievalError(
            f"Chunks were embedded with {row[0]} but the app uses {EMBEDDING_MODEL}."
        )


def vector_search(conn: psycopg.Connection, question: str, limit: int) -> list[dict]:
    query_vector = embed_query(question)
    rows = conn.execute(
        """
        SELECT id, source, pages, content
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query_vector, limit),
    ).fetchall()
    return [dict(zip(("id", "source", "pages", "content"), r, strict=True)) for r in rows]


def keyword_search(conn: psycopg.Connection, question: str, limit: int) -> list[dict]:
    # plainto_tsquery turns the question into stemmed terms joined with AND
    # ("membrane & decay & factor"). A full question rarely has ALL its words
    # in one chunk, so AND is swapped for OR (|): a chunk matching more terms,
    # or rarer ones, still ranks higher through ts_rank_cd.
    rows = conn.execute(
        """
        WITH q AS (
            SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS query
        )
        SELECT id, source, pages, content
        FROM chunks, q
        WHERE content_tsv @@ q.query
        ORDER BY ts_rank_cd(content_tsv, q.query) DESC
        LIMIT %s
        """,
        (question, limit),
    ).fetchall()
    return [dict(zip(("id", "source", "pages", "content"), r, strict=True)) for r in rows]


def hybrid_search(
    conn: psycopg.Connection, question: str, top_k: int = 5
) -> list[dict]:
    """Merge vector and keyword results with Reciprocal Rank Fusion."""
    vector_hits = vector_search(conn, question, CANDIDATES)
    keyword_hits = keyword_search(conn, question, CANDIDATES)

    fused: dict[str, dict] = {}
    for method, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, chunk in enumerate(hits, start=1):
            entry = fused.setdefault(chunk["id"], {**chunk, "score": 0.0})
            entry["score"] += 1 / (RRF_K + rank)
            entry[f"{method}_rank"] = rank  # kept to show where it came from

    ranked = sorted(fused.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]


def search(question: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
    """Entry point for the rest of the app."""
    with connect() as conn:
        check_model(conn)
        if mode == "vector":
            return vector_search(conn, question, top_k)
        if mode == "keyword":
            return keyword_search(conn, question, top_k)
        return hybrid_search(conn, question, top_k)


def main() -> None:
    try:
        cli()
    except RetrievalError as e:
        raise SystemExit(str(e)) from None


def cli() -> None:
    parser = argparse.ArgumentParser(description="Search the chunk database.")
    parser.add_argument("question")
    parser.add_argument(
        "--mode", choices=["hybrid", "vector", "keyword"], default="hybrid"
    )
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    # Windows may print with an old character set (cp1252) that has no "−" or
    # "→", which crashes print(). Force UTF-8, replacing anything unprintable.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Time each stage separately, so a slow run shows where the time goes
    timings = {}
    if args.mode != "keyword":
        t = time.perf_counter()
        embed_query(args.question)
        timings["embed question (API call)"] = time.perf_counter() - t
    t = time.perf_counter()
    with connect():
        pass
    timings["connect to database"] = time.perf_counter() - t
    t = time.perf_counter()
    results = search(args.question, args.top, args.mode)
    timings["search (embed + SQL)"] = time.perf_counter() - t
    print(f'\nQuestion: "{args.question}"  (mode: {args.mode})')
    if not results:
        print("No results.")
    for i, c in enumerate(results, start=1):
        origin = ""
        if args.mode == "hybrid":
            v, k = c.get("vector_rank", "-"), c.get("keyword_rank", "-")
            origin = f" | vector #{v}, keyword #{k}"
        print(f"\n{i}. {c['source']} | pages {', '.join(c['pages'])}{origin}")
        print("   " + c["content"][:400].replace("\n\n", " ") + " ...")

    print("\nTimings:")
    for stage, seconds in timings.items():
        print(f"  {stage:<30} {seconds:6.2f} s")


if __name__ == "__main__":
    main()
