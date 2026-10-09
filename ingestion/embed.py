"""Turn chunks into embeddings (vectors), and search them in memory.

An embedding model reads a piece of text and outputs a fixed-length list of
numbers (here 1024). Texts with similar meaning get vectors pointing in
similar directions, so "find relevant passages" becomes "find the vectors
closest to the question's vector".

Two modes:
    python ingestion/embed.py
        Embed every chunk in data/chunks/*.jsonl and save the vectors to
        data/embeddings/<name>.npy (row i = chunk i of the .jsonl file).

    python ingestion/embed.py --ask "What is the membrane decay factor?"
        Embed the question and print the closest chunks. This is retrieval
        without a database; Step 1.6 moves the same idea into Postgres.

Rules this script enforces:
    - Questions and chunks must be embedded with the SAME model. The model
      name is saved next to the vectors and checked before searching.
    - Changing the model means re-embedding everything.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# The model can be changed without editing code: set EMBEDDING_MODEL in the
# environment. Remember to re-run this script afterwards.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
BATCH_SIZE = 16  # chunks embedded at once; lower it if you run out of memory

CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")
INFO_FILE = EMBEDDINGS_DIR / "info.json"


def load_model() -> SentenceTransformer:
    """Download the model the first time (cached afterwards) and load it.

    sentence-transformers picks the GPU automatically if PyTorch can see it.
    """
    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Running on: {model.device}")
    print(f"  Vector size: {model.get_embedding_dimension()}")
    return model


def embed_documents(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed chunk texts. Vectors are normalized to length 1, so the dot
    product of two vectors is their cosine similarity (1 = same direction).

    encode_document() applies the model's "document" prompt if it defines
    one (Qwen3 doesn't: documents are embedded as they are)."""
    return model.encode_document(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(model: SentenceTransformer, question: str) -> np.ndarray:
    """Embed a question.

    Some models are trained to treat questions and documents differently and
    expect a prompt (an instruction) in front of the question. encode_query()
    adds the model's "query" prompt automatically when the model defines one,
    which Qwen3-Embedding does.
    """
    return model.encode_query(question, normalize_embeddings=True)


def load_chunks(jsonl_file: Path) -> list[dict]:
    with open(jsonl_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build() -> None:
    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if not chunk_files:
        print("No chunks found. Run: python ingestion/chunk.py")
        return

    model = load_model()
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    total, start = 0, time.perf_counter()

    for chunk_file in chunk_files:
        chunks = load_chunks(chunk_file)
        print(f"\n{chunk_file.name}: {len(chunks)} chunks")
        vectors = embed_documents(model, [c["text"] for c in chunks])
        np.save(EMBEDDINGS_DIR / f"{chunk_file.stem}.npy", vectors)
        total += len(chunks)

    elapsed = time.perf_counter() - start
    # Record which model produced these vectors, so a later search with a
    # different model is refused instead of silently returning nonsense.
    info = {
        "model": EMBEDDING_MODEL,
        "dimension": model.get_embedding_dimension(),
        "chunks": total,
    }
    INFO_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nEmbedded {total} chunks in {elapsed:.1f} s on {model.device}")
    print(f"Saved to {EMBEDDINGS_DIR}/")


def ask(question: str, top_k: int) -> None:
    if not INFO_FILE.exists():
        print("No embeddings found. Run: python ingestion/embed.py")
        return
    info = json.loads(INFO_FILE.read_text(encoding="utf-8"))
    if info["model"] != EMBEDDING_MODEL:
        print(
            f"Embeddings were made with {info['model']}, but the current "
            f"model is {EMBEDDING_MODEL}. Re-run: python ingestion/embed.py"
        )
        return

    model = load_model()
    query_vector = embed_query(model, question)

    # Score every chunk of every document against the question
    results = []
    for vector_file in sorted(EMBEDDINGS_DIR.glob("*.npy")):
        chunks = load_chunks(CHUNKS_DIR / f"{vector_file.stem}.jsonl")
        vectors = np.load(vector_file)
        if len(vectors) != len(chunks):
            # chunk.py was re-run after embedding: rows no longer match chunks
            print(
                f"{vector_file.stem}: chunks changed. Re-run: python ingestion/embed.py"
            )
            return
        scores = vectors @ query_vector  # one dot product per chunk
        results.extend(zip(scores.tolist(), chunks))

    results.sort(key=lambda r: r[0], reverse=True)
    print(f'\nQuestion: "{question}"')
    for score, chunk in results[:top_k]:
        pages = ", ".join(chunk["pages"])
        print(f"\n--- score {score:.3f} | {chunk['source']} | pages {pages}")
        print(chunk["text"][:500] + " ...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ask", metavar="QUESTION", help="search instead of building")
    parser.add_argument(
        "--top", type=int, default=3, help="results to show (default 3)"
    )
    args = parser.parse_args()

    if args.ask:
        ask(args.ask, args.top)
    else:
        build()


if __name__ == "__main__":
    main()
