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

This module is used by the chatbot. It can also be run directly to inspect
results:
    python app/retrieval.py "What is the membrane decay factor?"
    python app/retrieval.py "surrogate gradient" --mode keyword
"""

import time

_start = time.perf_counter()  # to measure how long the imports below take

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# Read .env BEFORE importing sentence_transformers: the Hugging Face library
# reads settings such as HF_HUB_OFFLINE once, when it is imported.
load_dotenv()

import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - _start

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

CANDIDATES = 20  # how many results each search contributes before fusion
RRF_K = 60  # standard RRF constant: dampens the gap between rank 1 and 2

_model = None  # loaded once, on first use, then reused for every question


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def connect() -> psycopg.Connection:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Add it to your .env file.")
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


def check_model(conn: psycopg.Connection) -> None:
    """Refuse to search if the table was filled with a different model."""
    row = conn.execute(
        "SELECT value FROM ingest_info WHERE key = 'embedding_model'"
    ).fetchone()
    if row is None:
        raise SystemExit("The database is empty. Run the ingestion pipeline first.")
    if row[0] != EMBEDDING_MODEL:
        raise SystemExit(
            f"Chunks were embedded with {row[0]} but the app uses {EMBEDDING_MODEL}."
        )


def vector_search(conn: psycopg.Connection, question: str, limit: int) -> list[dict]:
    query_vector = get_model().encode_query(question, normalize_embeddings=True)
    rows = conn.execute(
        """
        SELECT id, source, pages, content
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query_vector, limit),
    ).fetchall()
    return [dict(zip(("id", "source", "pages", "content"), r)) for r in rows]


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
    return [dict(zip(("id", "source", "pages", "content"), r)) for r in rows]


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
    timings = {"imports (PyTorch etc.)": IMPORT_SECONDS}
    t = time.perf_counter()
    if args.mode != "keyword":
        model = get_model()
        timings[f"load model ({model.device})"] = time.perf_counter() - t
        t = time.perf_counter()
        model.encode_query("warm-up", normalize_embeddings=True)
        timings["first encode (GPU warm-up)"] = time.perf_counter() - t
    t = time.perf_counter()
    with connect():
        pass
    timings["connect to database"] = time.perf_counter() - t
    t = time.perf_counter()
    results = search(args.question, args.top, args.mode)
    timings["search itself"] = time.perf_counter() - t
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