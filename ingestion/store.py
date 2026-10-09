"""Load chunks and their embeddings into Postgres (pgvector).

Reads, for every document:
    data/chunks/<name>.jsonl      the chunk texts and metadata
    data/embeddings/<name>.npy    one vector per chunk, same order
and writes one row per chunk into the `chunks` table.

The database is a copy of these files, never the original: running this script
again replaces each document's rows, so the database can always be rebuilt
from the PDFs (parse -> chunk -> embed -> store).

Usage:
    python ingestion/store.py              # load / reload everything
    python ingestion/store.py --reset      # drop the table first (needed if
                                           # the embedding model changes size)
    python ingestion/store.py --ask "..."  # vector search with SQL, to check
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")
INFO_FILE = EMBEDDINGS_DIR / "info.json"


def connect() -> psycopg.Connection:
    """Open a connection using DATABASE_URL from the .env file."""
    load_dotenv()  # copies the values in .env into the environment
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Add it to your .env file.")
    conn = psycopg.connect(url)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # lets Python numpy arrays be sent as pgvector values
    return conn


def create_schema(conn: psycopg.Connection, dimension: int) -> None:
    """Create the tables and indexes if they don't exist yet."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id        text PRIMARY KEY,           -- "thesis-0023"
            source    text NOT NULL,              -- "thesis.pdf"
            pages     text[] NOT NULL,            -- {{"10","11"}}, printed labels
            words     integer NOT NULL,
            content   text NOT NULL,
            embedding vector({dimension}) NOT NULL,
            -- Keyword search support, computed by Postgres from the text:
            -- words are reduced to their stems ("neurons" -> "neuron").
            content_tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        )
    """)
    # Index for vector search: HNSW is a graph that finds near neighbours
    # without comparing against every row. Cosine matches how we normalized.
    conn.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
    """)
    # Index for keyword search (used in the next step)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
        ON chunks USING gin (content_tsv)
    """)
    # Which embedding model filled the table. Searching with another model
    # would return nonsense, so retrieval code can check this first.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_info (
            key   text PRIMARY KEY,
            value text NOT NULL
        )
    """)


def load_chunks(jsonl_file: Path) -> list[dict]:
    with open(jsonl_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def store_all(reset: bool) -> None:
    if not INFO_FILE.exists():
        raise SystemExit("No embeddings found. Run: python ingestion/embed.py")
    info = json.loads(INFO_FILE.read_text(encoding="utf-8"))

    with connect() as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks, ingest_info")
            print("Dropped existing tables.")
        create_schema(conn, info["dimension"])

        vector_files = sorted(EMBEDDINGS_DIR.glob("*.npy"))
        if not vector_files:
            # Without this guard, the cleanup below would empty the table
            raise SystemExit("No .npy files found. Run: python ingestion/embed.py")

        stored_sources = []
        for vector_file in vector_files:
            chunks = load_chunks(CHUNKS_DIR / f"{vector_file.stem}.jsonl")
            vectors = np.load(vector_file)
            if len(chunks) != len(vectors):
                raise SystemExit(
                    f"{vector_file.stem}: {len(chunks)} chunks but {len(vectors)} "
                    "vectors. Re-run: python ingestion/embed.py"
                )
            source = chunks[0]["source"]

            # Replace this document's rows. Nothing is permanent yet: all
            # documents are committed together when the `with connect()`
            # block ends. If any document fails, the whole run is undone and
            # the database keeps its previous, complete state.
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chunks (id, source, pages, words, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (c["id"], c["source"], c["pages"], c["words"], c["text"], v)
                            for c, v in zip(chunks, vectors)
                        ],
                    )
            print(f"{source}: {len(chunks)} chunks loaded")
            stored_sources.append(source)

        # Remove documents that are in the database but no longer have files
        # (e.g. a PDF you deleted from data/). "<> ALL(list)" means "differs
        # from every value in the list". RETURNING gives back what was deleted.
        removed = conn.execute(
            "DELETE FROM chunks WHERE source <> ALL(%s) RETURNING source",
            (stored_sources,),
        ).fetchall()
        for removed_source in sorted({row[0] for row in removed}):
            count = sum(1 for row in removed if row[0] == removed_source)
            print(f"{removed_source}: {count} chunks removed (no longer in data/)")

        conn.execute(
            """
            INSERT INTO ingest_info (key, value) VALUES ('embedding_model', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (info["model"],),
        )

        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    # Leaving the `with` block above committed everything
    print(f"\nCommitted. Table 'chunks' holds {total} rows (model: {info['model']})")


def ask(question: str, top_k: int) -> None:
    """Vector search done by Postgres instead of numpy."""
    # Imported here so storing doesn't need to load the embedding model
    from embed import EMBEDDING_MODEL, embed_query, load_model

    with connect() as conn:
        row = conn.execute(
            "SELECT value FROM ingest_info WHERE key = 'embedding_model'"
        ).fetchone()
        if row is None:
            raise SystemExit("The database is empty. Run: python ingestion/store.py")
        if row[0] != EMBEDDING_MODEL:
            raise SystemExit(
                f"The table was filled with {row[0]}, but the current model is "
                f"{EMBEDDING_MODEL}. Re-run embed.py and store.py --reset."
            )

        model = load_model()
        query_vector = embed_query(model, question)
        # <=> is pgvector's cosine distance (0 = same direction), so
        # 1 - distance gives the same similarity score as embed.py --ask.
        rows = conn.execute(
            """
            SELECT source, pages, content, 1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vector, query_vector, top_k),
        ).fetchall()

    print(f'\nQuestion: "{question}"')
    for source, pages, content, score in rows:
        print(f"\n--- score {score:.3f} | {source} | pages {', '.join(pages)}")
        print(content[:500] + " ...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="drop tables first")
    parser.add_argument("--ask", metavar="QUESTION", help="search instead of storing")
    parser.add_argument(
        "--top", type=int, default=3, help="results to show (default 3)"
    )
    args = parser.parse_args()

    if args.ask:
        ask(args.ask, args.top)
    else:
        store_all(args.reset)


if __name__ == "__main__":
    main()