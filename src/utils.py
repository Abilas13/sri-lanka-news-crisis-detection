"""
utils.py

Small, dependency-light helpers shared across pipeline modules.
Nothing here should import from other src/ modules, to avoid circular imports.
"""

import hashlib
import logging
import re
from datetime import datetime


def setup_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger. Each module calls this once:
        logger = setup_logger(__name__)
    Keeps log formatting consistent across the whole pipeline.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on reimport
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def content_hash(text: str) -> str:
    """
    Deterministic hash of article content, used for exact-duplicate detection.
    Normalizes whitespace first so trivial formatting differences don't
    produce different hashes for otherwise-identical text.
    """
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def iso_week_string(dt: datetime) -> str:
    """
    Converts a datetime to an ISO week string like '2022-W25'.
    Used consistently for publication_week and temporal_features.week
    so joins between tables never mismatch on formatting.
    """
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Returns None instead of raising on divide-by-zero — used in growth-rate calcs."""
    if not denominator:
        return None
    return numerator / denominator