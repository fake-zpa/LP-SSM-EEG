"""Structured logging utilities."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = None, level: str = "INFO") -> logging.Logger:
    """Return a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"{name}_{ts}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
