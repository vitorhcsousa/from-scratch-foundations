FROM python:3.13-slim AS base

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies (non-frozen so this image stays usable as the repo evolves)
RUN uv sync --no-dev

# Copy source
COPY src/ src/

# Install the project
RUN uv sync --no-dev

ENTRYPOINT ["uv", "run", "foundations"]
CMD ["--help"]
