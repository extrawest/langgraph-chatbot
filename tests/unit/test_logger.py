import json
import logging
import sys

import pytest

from langgraph_chatbot.logger import (
    ContextFilter,
    JsonFormatter,
    log_context,
    setup_logging,
)


@pytest.fixture
def context():
    """Set log_context for one test and restore it afterwards (it is process-global)."""
    tokens = []

    def _set(values: dict):
        tokens.append(log_context.set(values))

    yield _set
    for token in reversed(tokens):
        log_context.reset(token)


def _record(msg="hello %s", args=("world",), exc_info=None) -> logging.LogRecord:
    return logging.LogRecord("langgraph_chatbot.test", logging.INFO, __file__, 1, msg, args, exc_info)


def test_context_filter_copies_context_to_record(context):
    context({"chat_id": "c1", "user_id": "u1"})
    record = _record()

    assert ContextFilter().filter(record) is True
    assert record.chat_id == "c1"
    assert record.user_id == "u1"


def test_context_filter_without_context_keeps_record():
    record = _record()
    assert ContextFilter().filter(record) is True
    assert not hasattr(record, "chat_id")


def test_json_formatter_basic_fields():
    data = json.loads(JsonFormatter().format(_record()))

    assert data["level"] == "INFO"
    assert data["logger"] == "langgraph_chatbot.test"
    assert data["message"] == "hello world"
    assert "time" in data
    assert "exception" not in data


def test_json_formatter_includes_context_fields(context):
    context({"chat_id": "c1", "mode": "voice"})
    record = _record()
    ContextFilter().filter(record)

    data = json.loads(JsonFormatter().format(record))

    assert data["chat_id"] == "c1"
    assert data["mode"] == "voice"


def test_json_formatter_includes_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        record = _record(exc_info=sys.exc_info())

    data = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in data["exception"]


def test_json_formatter_keeps_non_ascii():
    out = JsonFormatter().format(_record("привет", ()))
    assert "привет" in out


@pytest.fixture
def restore_logging(monkeypatch):
    yield
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    setup_logging()  # back to what main.py configures


@pytest.mark.parametrize(
    ("fmt", "formatter_type"), [("json", JsonFormatter), ("text", logging.Formatter)]
)
def test_setup_logging(monkeypatch, restore_logging, fmt, formatter_type):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", fmt)

    setup_logging()

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert logging.getLogger("langgraph_chatbot").level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.WARNING
    (handler,) = root.handlers
    assert type(handler.formatter) is formatter_type
    assert any(isinstance(f, ContextFilter) for f in handler.filters)
