"""Logging for the server: one line per event, plain text or JSON.

Configured from the environment so the same image fits a laptop and a log
collector: ``LOG_LEVEL`` (default ``INFO``) and ``LOG_FORMAT`` (``text``,
the default, or ``json`` for one JSON object per line).
"""

import json
import logging
import os
import sys
import time

# Fields that logging puts on every record; everything else passed via ``extra`` is ours.
_BUILTIN = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "taskName", "color_message"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, and the extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _BUILTIN and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """Readable lines with the extra fields appended as key=value."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = " ".join(f"{k}={v}" for k, v in record.__dict__.items() if k not in _BUILTIN and not k.startswith("_"))
        return f"{base} {extras}".rstrip()


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """
    Configure the root logger from the arguments or the environment.

    Parameters
    ----------
    level : str or None, optional
        Log level name; default ``LOG_LEVEL`` or ``INFO``.
    fmt : str or None, optional
        ``"json"`` or ``"text"``; default ``LOG_FORMAT`` or ``text``.
    """
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.environ.get("LOG_FORMAT", "text")).lower()
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn's own access log duplicates the request log below: keep its errors only.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
