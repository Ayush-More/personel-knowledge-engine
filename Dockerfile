
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Configure uv for predictable container installs
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first for better Docker layer caching
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# Copy application source and required data
COPY src/ ./src/
COPY data/processed/chunk.jsonl ./data/processed/chunk.jsonl

# Make the local source package importable
ENV PYTHONPATH=/app/src

# Create the persistent Qdrant storage directory
RUN mkdir -p /app/data/qdrant

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "knowledge_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
