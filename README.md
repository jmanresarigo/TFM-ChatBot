# TFM Chatbot

[![CI](https://github.com/jmanresarigo/TFM-ChatBot/actions/workflows/ci.yml/badge.svg)](https://github.com/jmanresarigo/TFM-ChatBot/actions/workflows/ci.yml)

A chat assistant that answers questions about my Master's thesis, *Spiking Neural Networks for Lift Coefficient Reconstruction in Extreme Vortex-Gust Airfoil Interactions*, and the papers it cites. Answers come only from the documents, with citations to document and page (retrieval-augmented generation, RAG).

**Status:** deployed on Railway (password-protected), with CI/CD from `main`. Next: observability (Phase 8).

Built as a learning project for the full lifecycle of an AI application: ingestion, retrieval, evaluation, an LLM API, a web app, Docker, CI, and cloud deployment.

---

## Architecture

```
  Browser (anyone with the link + password)
     │ HTTPS
     ▼
 RAILWAY (EU West)                                        EXTERNAL APIs (free tiers)
 ┌───────────────────────────────────────────────┐
 │ app   FastAPI + chat page (Docker image)      │ ─────►  Cloudflare Workers AI
 │        │ private network                      │         embeds the QUESTION
 │        ▼ db.railway.internal:5432             │         (Qwen3-Embedding-0.6B)
 │ db    Postgres 17 + pgvector, on a volume     │
 │        2285 chunks: text + 1024-dim vectors   │ ─────►  Groq
 └───────────────────────────────────────────────┘         writes the ANSWER
                                                            (Qwen 3.8 27B)
 MY PC (never deployed)
   PDFs → parse → chunk → embed (GPU) → store ──► local DB and, when updating, Railway DB
   evaluations: generation via Groq, judge on the local GPU (Ollama)
```

**How a question is answered**

1. **Route** (follow-up questions only): one short LLM call decides whether the documents are needed at all ("what did I ask first?" doesn't) and rewrites follow-ups ("and its time step?") into a standalone search query.
2. **Embed** the query through Cloudflare, with the same model that embedded the chunks.
3. **Hybrid search** in Postgres: vector search (meaning) + full-text search (exact terms), merged with Reciprocal Rank Fusion. Top 5 chunks.
4. **Generate** with the LLM: system prompt + conversation history (trimmed to a token budget) + numbered excerpts + question. The answer streams to the browser (Server-Sent Events).
5. **Check citations**: each `[n]` is mapped back to document and page; citations to non-existent excerpts are flagged.

The server keeps no conversation state: the browser holds the history and sends it with each question.

## Tech stack

| Area | Choice |
|---|---|
| Language, packaging | Python 3.12, [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`) |
| PDF parsing | PyMuPDF, with custom cleaning (headers/footers, ligatures, page labels) |
| Embeddings | Qwen3-Embedding-0.6B: locally (sentence-transformers, GPU) for ingestion; Cloudflare Workers AI for questions |
| Vector database | PostgreSQL 17 + pgvector (HNSW index), full-text search (GIN) |
| LLM | Groq, OpenAI-compatible API, `qwen/qwen3.8-27b` (any OpenAI-compatible provider works, including a local Ollama) |
| Web | FastAPI, uvicorn, vanilla HTML/CSS/JS, Server-Sent Events; answers rendered as Markdown + math (marked, KaTeX, sanitized by DOMPurify, vendored in `app/static/vendor/`) |
| Containers | Docker (slim image, no ML libraries, ~140 MB compressed), Docker Compose locally |
| CI | GitHub Actions: ruff, pytest, Docker build + start-up test |
| Hosting | Railway: app + database, deployed from `main` after CI passes |

## Repository structure

```
app/
  main.py            web server: chat API (SSE), health checks, password protection
  llm.py             prompts, routing, history trimming, OpenAI-compatible LLM client
  retrieval.py       question embedding (API), vector + keyword search, RRF fusion
  references.json    IEEE reference of each document, shown under answers
  static/index.html  chat page
  static/vendor/     Markdown, math and sanitizer libraries (MIT / Apache-2.0)
ingestion/
  parse.py           PDF -> clean page text            (data/parsed/)
  chunk.py           pages -> overlapping chunks       (data/chunks/)
  embed.py           chunks -> vectors on the GPU      (data/embeddings/)
  store.py           vectors + chunks -> Postgres
evals/
  questions.jsonl    40 questions with answer keys (document + pages)
  run_retrieval_eval.py   hit@k, MRR
  run_answer_eval.py      deterministic checks + LLM judge
  results/           score summaries (no document text)
prompts/             system prompt versions, router prompt, judge prompt
tests/               unit tests (made-up text, fakes; no real services)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Data and privacy

- **Never committed:** the PDFs (the thesis is unpublished; the papers are copyrighted), everything under `data/`, `.env`, model weights, database dumps.
- **What leaves my PC, and where it goes:**

  | Data | Goes to | Notes |
  |---|---|---|
  | The question | Cloudflare Workers AI (embedding) | Cloudflare states it doesn't train on inputs |
  | The question + 5 retrieved excerpts + recent history | Groq (answer generation) | Zero Data Retention enabled |
  | All chunks (text + vectors) | Railway Postgres, EU West | private network only; the public address is opened only while loading data |
  | The PDFs, parsed text, eval runs with excerpts, judge prompts | nowhere: stay on my PC | the judge runs locally |

- `evals/questions.jsonl` contains answer key facts taken from the unpublished thesis, which is one reason this repository is **private**. Before making it public, move that file out of the repository or replace it with a sanitized set.

## Run locally

**Requirements:** Python 3.12 with uv, Docker Desktop, the PDFs in `data/pdfs/`, a Groq API key and a Cloudflare account (free).

```bash
uv sync                          # all dependencies, including ingestion (PyTorch) and dev tools
cp .env.example .env             # then fill in the values (see Configuration)
docker compose up -d db          # Postgres + pgvector on 127.0.0.1:5432
```

**Build the database** (once, and again when documents change):

```bash
python ingestion/parse.py
python ingestion/chunk.py
python ingestion/embed.py        # uses the GPU if available
python ingestion/store.py
```

**Run the app:**

```bash
uvicorn main:app --app-dir app --reload --port 8080      # development, auto-reload
# or the packaged version, exactly as deployed:
docker compose up -d --build
```

Open http://127.0.0.1:8080. Other useful commands:

```bash
python app/retrieval.py "What is the membrane decay factor?"   # inspect search results
python app/llm.py --chat                                        # chat in the terminal
```

## Configuration

All settings are environment variables (`.env` locally, Railway variables in production). See `.env.example`.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string |
| `EMBEDDING_MODEL` | model whose vectors are stored (`Qwen/Qwen3-Embedding-0.6B`); must match ingestion |
| `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_API_MODEL` | embeddings API for questions (Cloudflare, or a local Ollama) |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_REASONING_EFFORT` | LLM API (Groq, or any OpenAI-compatible provider) |
| `ROUTER_MODEL` (optional) | separate model for routing, to use its own quota |
| `APP_PASSWORD` | password for the chat page; empty = no password (local only) |
| `JUDGE_MODEL`, `JUDGE_BASE_URL` | local evaluation judge (Ollama), a different model family than `LLM_MODEL` |

## Deployment (Railway)

Two services in one project, region EU West:

| Service | Source | Key settings |
|---|---|---|
| `db` | Docker image `pgvector/pgvector:pg17` | volume at `/var/lib/postgresql/data`, `PGDATA=/var/lib/postgresql/data/pgdata`, SSL on (Debian's built-in certificate), private only |
| `app` | this repository (`Dockerfile`), branch `main` | `DATABASE_URL` via reference variables (`${{db.POSTGRES_PASSWORD}}`, `${{db.RAILWAY_PRIVATE_DOMAIN}}`), API keys, `APP_PASSWORD`; health check `/health/live`; public HTTPS domain |

Both services use **app sleeping** (serverless): they stop after a few idle minutes and wake on the next request, which keeps usage within the free allowance. The first request after a pause takes a few seconds longer.

**Updating the production database** (e.g. after adding a PDF): run the ingestion pipeline locally, then
1. `db` → Settings → Networking → enable a **TCP Proxy** to port 5432;
2. load from the PC: set `DATABASE_URL=postgresql://tfm:<password>@<proxy-host>:<proxy-port>/tfm?sslmode=require` in the terminal and run `python ingestion/store.py`;
3. **delete the TCP proxy** again.

`store.py` reloads everything from the local `data/` files in one transaction, so the live app switches from the old data to the new data at once.

## CI/CD

```
pull request ──► CI: ruff · pytest · Docker build + start-up test
merge to main ──► CI again on main ──► Railway waits for CI ("Wait for CI")
              ──► builds the image ──► checks /health/live ──► switches traffic (no downtime)
```

Run the same checks locally before pushing:

```bash
uv run ruff check .
uv run pytest
```

CI never needs documents, a database or API keys: the tests use made-up text and fakes.

## Endpoints

| Path | Description |
|---|---|
| `GET /` | chat page (password-protected when `APP_PASSWORD` is set) |
| `POST /api/chat` | `{"question", "history"}` → Server-Sent Events: `route`, `excerpts`, `token`, `done` / `error` |
| `GET /health` | readiness: database, embeddings API and LLM API; 200 or 503 (public) |
| `GET /health/live` | liveness: 200 as soon as the server runs (public; used by Railway) |

## Evaluation

Run by hand (they need the documents and, for judging, the local GPU):

```bash
python evals/run_retrieval_eval.py --mode all
python evals/run_answer_eval.py generate      # via the LLM API; paced by its rate limit
python evals/run_answer_eval.py judge         # local judge model (Ollama)
```

Baselines (35 answerable + 5 unanswerable questions):

| Retrieval (local GPU embeddings) | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| hybrid | 57% | 91% | 97% | 0.72 |
| vector | 63% | 83% | 91% | 0.74 |
| keyword | 54% | 77% | 91% | 0.67 |

| Answers (local qwen3.5:27b, prompt v1) | Result |
|---|---|
| Unanswerable questions correctly refused | 5 / 5 |
| Judge: correct / partial / wrong | 32 / 2 / 1 (of 35) |
| Faithful to the excerpts | 34 / 35 |

Current runs (Cloudflare question embeddings, Groq `qwen/qwen3.8-27b`, prompt v3) are summarized in `evals/results/`.

## Cost

Designed to run at **$0/month** for light use: Railway's free allowance with both services sleeping when idle, Groq's free tier (8,000 tokens/min, ~50 questions/day), and Cloudflare Workers AI's free daily allowance. Free tiers change; check each provider's pricing page.

## Project phases

| Phase | Topic | Status |
|---|---|---|
| 0 | Repository, Git workflow | ✅ |
| 1 | Ingestion and retrieval (parse, chunk, embed, pgvector, hybrid search) | ✅ |
| 2 | Evaluation set (40 questions, retrieval metrics) | ✅ |
| 3 | LLM: grounded answers with citations, answer evaluation with an LLM judge | ✅ |
| 4 | API and UI: FastAPI, streaming, conversation memory, routing | ✅ |
| 5 | Docker: image + Compose | ✅ |
| 6 | CI: lint, unit tests, image build in GitHub Actions | ✅ |
| 7 | Deployment: Railway, hosted LLM and embedding APIs, CD | ✅ |
| 8 | Observability: logs, metrics, tracing of each answer | next |

## Known limitations

- Equations and tables are often garbled by PDF text extraction; the model is told not to reconstruct them.
- Groq's free tier allows roughly 2 questions per minute; beyond that, users see a "rate limit reached" message.
- The first request after the app has been idle takes a few seconds longer (app sleeping).
- One shared password, no individual accounts.