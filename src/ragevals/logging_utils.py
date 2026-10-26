"""Logging helpers: one place to configure format and level."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging once. Level comes from the argument or RAGEVALS_LOG."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.environ.get("RAGEVALS_LOG", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
