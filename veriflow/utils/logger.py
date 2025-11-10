# utils/logger.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARN,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _env_level(default: str = "INFO") -> int:
    """Read LOG_LEVEL from env, fallback to default."""
    lvl = os.getenv("LOG_LEVEL", default).upper()
    return _LEVEL_MAP.get(lvl, logging.INFO)


def _colorize(level: int, msg: str) -> str:
    """Basic ANSI colorization by level (works on most terminals)."""
    if not sys.stdout.isatty():
        return msg
    if level >= logging.ERROR:
        return f"\033[91m{msg}\033[0m"   # red
    if level >= logging.WARNING:
        return f"\033[93m{msg}\033[0m"   # yellow
    if level >= logging.INFO:
        return f"\033[92m{msg}\033[0m"   # green
    return msg  # DEBUG or others keep default


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        return _colorize(record.levelno, base)


def init_logger(
    name: str = "veriflow",
    level: int | None = None,
    log_dir: str | Path | None = None,
    file_name: str = "run.log",
    file_max_mb: int = 5,
    file_backup: int = 3,
) -> logging.Logger:
    """
    Initialize a project logger:
      - colored stream handler to stdout
      - optional rotating file handler
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level if level is not None else _env_level("INFO"))

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Stream handler (colored)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logger.level)
    sh.setFormatter(_ColorFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(sh)

    # Rotating file handler (optional)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            filename=str(log_dir / file_name),
            maxBytes=file_max_mb * 1024 * 1024,
            backupCount=file_backup,
            encoding="utf-8",
        )
        fh.setLevel(logger.level)
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(fh)

    return logger


# Convenience default logger
log = init_logger()


def get_logger(child: str) -> logging.Logger:
    """Create/get a child logger under the root project logger."""
    base = logging.getLogger("veriflow")
    return base.getChild(child)