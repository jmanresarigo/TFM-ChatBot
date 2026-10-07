# TFM Chatbot

A personal assistant for my Master's thesis on Spiking Neural Networks. It answers questions about my thesis and the papers I cited, using Retrieval-Augmented Generation (RAG) so that answers are grounded in the actual documents and come with source and page citations.

The language model runs locally on my own GPU, and the document chunks are stored in PostgreSQL with pgvector. The main goal of the project is to learn how AI applications are built and deployed in practice: packaging, CI/CD, rollout and observability.

> **Status:** early stage. Only the repository skeleton exists so far (Phase 0).


## How it will work

1. **Ingestion:** PDFs are parsed, split into chunks, embedded, and stored in Postgres (pgvector) together with their source document and page number.
2. **Retrieval:** a question is embedded and matched against the stored chunks, using vector search combined with keyword search.
3. **Generation:** the retrieved chunks and the question are sent to a local LLM, which answers using only that context and cites its sources.
4. **Chat UI:** a minimal web interface streams the answer and shows the citations.

## Repository structure

```
app/          FastAPI service (API and chat backend)
ingestion/    Scripts to parse, chunk, embed and store the PDFs
prompts/      Versioned prompt files
evals/        Evaluation questions and scoring scripts
tests/        Unit tests
data/         My PDFs (git-ignored, never committed)
```

More files (Dockerfile, docker-compose.yml, CI workflow, dependency list) will be added in later phases.

## Data and secrets

- The thesis and papers are **not** in this repository. The thesis is unpublished and the papers are copyrighted. Put your own PDFs in a local `data/` folder, which is git-ignored.
- Real configuration lives in a local `.env` file, which is also git-ignored. Copy `.env.example` to `.env` and fill in your values.
- Model weights and database files are never committed.

## Getting started

Setup instructions will be added as the project grows. The planned order of work is:

| Phase | Goal |
|---|---|
| 0 | Repository, `.gitignore`, hygiene |
| 1 | Ingestion and retrieval (no LLM yet) |
| 2 | Evaluation set |
| 3 | Add the local LLM |
| 4 | API and chat UI |
| 5 | Containerize with Docker Compose |
| 6 | CI with GitHub Actions |
| 7 | Deploy to Railway |
| 8 | Observability and failure drills (optional) |

## Tech stack (planned)

Python, FastAPI, PostgreSQL + pgvector, a local LLM served through Ollama (vLLM later, maybe), Docker, GitHub Actions, Railway.


## Development workflow

All changes go through short-lived branches and pull requests into `main`.