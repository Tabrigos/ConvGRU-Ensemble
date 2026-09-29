import json
import logging

from convgru_ensemble.logging_config import JsonFormatter, TextFormatter, configure_logging


def _record(**extra):
    record = logging.LogRecord("convgru_ensemble.test", logging.INFO, __file__, 1, "request", (), None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_emits_one_object_with_extras():
    line = JsonFormatter().format(_record(request_id="abc", status=200, elapsed_ms=12.5))
    entry = json.loads(line)
    assert entry["message"] == "request" and entry["level"] == "INFO"
    assert entry["request_id"] == "abc" and entry["status"] == 200 and entry["elapsed_ms"] == 12.5
    assert entry["time"].endswith("Z")
    assert "args" not in entry and "msecs" not in entry


def test_text_formatter_appends_extras():
    line = TextFormatter("%(levelname)s %(message)s").format(_record(request_id="abc", status=200))
    assert line == "INFO request request_id=abc status=200"


def test_configure_logging_reads_the_environment(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging()
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    configure_logging(level="INFO", fmt="text")
    assert isinstance(logging.getLogger().handlers[0].formatter, TextFormatter)
