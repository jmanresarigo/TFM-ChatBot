# The web app (FastAPI + retrieval) as a container image.
# Ollama is NOT in the image: it keeps running on the host, where the GPU is.
#
# Build:  docker build -t tfm-chatbot .
# Run:    docker compose up   (see docker-compose.yml)

FROM python:3.12-slim

# uv, copied as a single binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/huggingface

WORKDIR /app

# A normal user to run the app: if someone broke into the app, they would not
# be root inside the container.
RUN useradd --system app && mkdir -p "$HF_HOME" && chown app "$HF_HOME"

# 1. Dependencies. Copied before the code so this slow step is cached: it only
#    re-runs when pyproject.toml or uv.lock change, not on every code edit.
#    On Linux, pyproject.toml selects the CPU build of PyTorch (much smaller).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# 2. The embedding model, downloaded now so the container starts without
#    internet access and without a 1 GB download on every start.
ARG EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
USER app
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${EMBEDDING_MODEL}')"
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HUB_OFFLINE=1

# 3. The code: the part that changes most often, so it goes last.
COPY app/ app/
COPY prompts/ prompts/

EXPOSE 8000
# Same command as in development, minus --reload, and listening on 0.0.0.0 so
# connections from outside the container are accepted.
CMD ["uvicorn", "main:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "8000"]