import json
import logging
import logging.config
import os
from contextvars import ContextVar

log_context: ContextVar[dict] = ContextVar("log_context", default={})


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in log_context.get().items():
            setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in log_context.get():
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    use_json = os.getenv("LOG_FORMAT", "text") == "json"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # do not silence libraries' loggers
            "filters": {"context": {"()": ContextFilter}},
            "formatters": {
                "json": {"()": JsonFormatter},
                "text": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "json" if use_json else "text",
                    "filters": ["context"],
                }
            },
            "root": {"level": "WARNING", "handlers": ["stdout"]},
            "loggers": {
                "langgraph_chatbot": {"level": level},  # our code at the chosen level
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
            },
        }
    )
