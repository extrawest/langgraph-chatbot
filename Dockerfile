# syntax=docker/dockerfile:1

# =====================================================================
#  Stage 1: builder - exists only to produce .venv.
#  Exactly one directory from here ends up in the final image.
# =====================================================================
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

# A compiler is needed by the dependencies that ship no prebuilt wheel for
# cp314/linux: asyncpg, argon2-cffi and friends are built from C sources.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry>=2.0,<3.0"

WORKDIR /app

# Dependency files only, as their own layer, so that edits to the source do not
# invalidate the cached install.
# README.md is here because pyproject.toml references it (readme = "README.md");
# without it Poetry fails while reading the project metadata.
COPY pyproject.toml poetry.lock README.md ./

# --only main : no dev groups
# --no-root   : dependencies only, not the package itself (src/ isn't copied yet)
RUN poetry install --only main --no-root


# =====================================================================
#  Stage 2: runtime - this is the image that ships to the server.
#  Starts from a clean base: no gcc, no Poetry, no caches here.
# =====================================================================
FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Unprivileged user. Fixed uid so ownership on mounted volumes stays
# predictable across rebuilds of the base image.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

# --chown inside COPY rather than a separate `RUN chown -R`: chown rewrites file
# metadata, which would add a second layer holding a full copy of .venv.
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Migrations run from this same image, so alembic.ini and alembic/ are needed.
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic/ ./alembic/

# Source last: it changes most often, so only this layer gets rebuilt.
# src/ also carries voice/static/, which FastAPI serves at /voice-ui.
COPY --chown=app:app src/ ./src/

USER app

# Documentation for humans and tooling; this does not publish the port.
EXPOSE 8000

# Default command is the API. The UI and the voice worker run from this same
# image with a different `command` in docker-compose:
#   api    : uvicorn langgraph_chatbot.main:app --host 0.0.0.0 --port 8000
#   ui     : python -m langgraph_chatbot.ui.app
#   worker : python -m langgraph_chatbot.voice.agent start
CMD ["uvicorn", "langgraph_chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
