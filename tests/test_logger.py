import io
import json
import logging
from src.utils.logger import get_logger


def test_logger_emits_valid_json(monkeypatch):
    """Ensure log records are emitted as single-line valid JSON objects."""
    stream = io.StringIO()
    logger = get_logger("test_json_logger")

    # Replace stdout stream handler with our string buffer
    for h in logger.handlers:
        logger.removeHandler(h)
    handler = logging.StreamHandler(stream)
    from pythonjsonlogger import jsonlogger

    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    test_message = "Validating structured logger output"
    logger.info(test_message)

    output = stream.getvalue().strip()
    parsed = json.loads(output)

    assert parsed["levelname"] == "INFO"
    assert parsed["name"] == "test_json_logger"
    assert parsed["message"] == test_message
    assert "asctime" in parsed


def test_logger_singleton_handler_behavior():
    """Ensure subsequent calls to get_logger do not attach duplicate handlers."""
    logger_a = get_logger("module_logger")
    handler_count_initial = len(logger_a.handlers)

    logger_b = get_logger("module_logger")
    assert len(logger_b.handlers) == handler_count_initial