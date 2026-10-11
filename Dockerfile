# The web app (FastAPI + retrieval) as a container image.
# No machine-learning libraries inside: questions are embedded by an
# embeddings API and answers come from an LLM API, so the image stays small
# (fits free hosting tiers) and starts in seconds.
#
# Build:  docker build -t tfm-chatbot .
# Run:    docker compose up   (see docker-compose.yml)

FROM python:3.12-slim

# uv, copied as a single binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# A normal user to run the app: if someone broke into the app, they would not
# be root inside the container.
RUN useradd --system app

# 1. Dependencies. Copied before the code so this step is cached: it only
#    re-runs when pyproject.toml or uv.lock change, not on every code edit.
#    --no-default-groups: only the app's own dependencies. The "dev" (ruff,
#    pytest) and "ingest" (PyTorch, sentence-transformers, PyMuPDF) groups
#    are for your PC, not for the server.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# 2. The code: the part that changes most often, so it goes last.
COPY app/ app/
COPY prompts/ prompts/

USER app
EXPOSE 8000
# Same command as in development, minus --reload, listening on 0.0.0.0 so
# connections from outside the container are accepted. Hosting platforms
# (Railway) tell the app which port to use through $PORT; 8000 otherwise.
# "exec" makes uvicorn replace the shell, so stop signals reach it directly.
CMD ["sh", "-c", "exec uvicorn main:app --app-dir app --host 0.0.0.0 --port ${PORT:-8000}"]