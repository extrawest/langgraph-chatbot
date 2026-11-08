# syntax=docker/dockerfile:1

# =====================================================================
#  Стадия 1: builder — существует только чтобы собрать .venv.
#  В финальный образ из неё попадёт ровно одна папка.
# =====================================================================
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

# Компилятор нужен тем зависимостям, для которых нет готового колеса
# под cp314/linux: asyncpg, argon2-cffi и т.п. собираются из C-исходников.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry>=2.0,<3.0"

WORKDIR /app

# Только файлы зависимостей — отдельным слоем, чтобы правки в коде
# не инвалидировали кэш установки.
# README.md здесь потому, что на него ссылается pyproject.toml
# (readme = "README.md"); без него Poetry падает на чтении метаданных.
COPY pyproject.toml poetry.lock README.md ./

# --only main    : без dev-групп
# --no-root      : ставим только зависимости, сам пакет — нет (src/ ещё не скопирован)
RUN poetry install --only main --no-root


# =====================================================================
#  Стадия 2: runtime — это и есть образ, который поедет на сервер.
#  Начинается с чистой базы: ни gcc, ни Poetry, ни кэшей здесь нет.
# =====================================================================
FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Непривилегированный пользователь. Фиксированный uid — чтобы права на
# примонтированных томах были предсказуемыми.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

# --chown прямо в COPY, а не отдельным `RUN chown -R`: chown меняет
# метаданные файлов, поэтому создал бы второй слой с полной копией venv.
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Миграции запускаются из этого же образа, значит alembic.ini и alembic/ нужны.
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic/ ./alembic/

# Код — последним: меняется чаще всего, пересобирать только его.
# Внутри src/ едет и voice/static/, который FastAPI отдаёт на /voice-ui.
COPY --chown=app:app src/ ./src/

USER app

# Справочно для человека и для инструментов; порт этим не открывается.
EXPOSE 8000

# Команда по умолчанию — API. UI и voice-воркер поднимаются из этого же
# образа с другим `command` в docker-compose:
#   api    : uvicorn langgraph_chatbot.main:app --host 0.0.0.0 --port 8000
#   ui     : python -m langgraph_chatbot.ui.app
#   worker : python -m langgraph_chatbot.voice.agent start
CMD ["uvicorn", "langgraph_chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
