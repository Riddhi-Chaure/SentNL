"""
SentNL — Unified Logging Service

Creates separate log files for system, training, and inference events
while sharing a consistent format.
"""

import logging
import sys
from pathlib import Path


_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _init_log_dir():
    """Ensure the logs directory exists."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a named logger that writes to both console and an optional file.

    Args:
        name:     Logger name (e.g. 'training', 'inference').
        log_file: Filename inside logs/ (e.g. 'training.log').
                  If None, logs only to console.
        level:    Logging level.

    Returns:
        Configured logging.Logger.
    """
    global _initialized
    if not _initialized:
        _init_log_dir()
        _initialized = True

    logger = logging.getLogger(name)

    # Prevent duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(
            _LOG_DIR / log_file, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
