# ---- builder: resolves and installs dependencies into a venv ----
FROM python:3.11-slim AS builder

# git is required at build time only: edf2parquet is installed from a pinned
# GitHub commit (not PyPI) -- see pyproject.toml.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ src/
RUN uv sync --frozen --no-dev

# ---- runtime: slim image, no build tools, no git, no uv ----
FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "bscarlos"]
