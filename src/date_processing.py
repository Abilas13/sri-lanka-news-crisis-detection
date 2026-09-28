"""
date_processing.py

Purpose
-------
Converts the raw `date` string from scraped_news (confirmed format: "DD Mon YYYY",
e.g. "01 Apr 2026") into a proper timestamp plus the derived temporal columns
that publication_week-based aggregation depends on.
"""

from datetime import datetime
import utils

logger = utils.setup_logger(__name__)

DATE_FORMAT = "%d %b %Y"   # matches "01 Apr 2026"


def normalize_date(date_raw: str) -> dict:
    """
    Parses one raw date string. Never raises — always returns a dict,
    with date_parse_status='FAILED' and all derived fields None if parsing
    fails, so a single bad row can't stop batch processing.
    """
    result = {
        "date_raw": date_raw,
        "published_at": None,
        "publication_date": None,
        "publication_year": None,
        "publication_month": None,
        "publication_week": None,
        "publication_day": None,
        "date_parse_status": "FAILED",
    }

    if not date_raw or not str(date_raw).strip():
        return result

    try:
        dt = datetime.strptime(str(date_raw).strip(), DATE_FORMAT)
    except ValueError:
        logger.warning(f"Could not parse date: {date_raw!r}")
        return result

    result["published_at"] = dt              # midnight on that date
    result["publication_date"] = dt.date()
    result["publication_year"] = dt.year
    result["publication_month"] = dt.month
    result["publication_day"] = dt.day
    result["publication_week"] = utils.iso_week_string(dt)   # e.g. '2026-W14'
    result["date_parse_status"] = "OK"
    return result


def normalize_dates_batch(date_raw_list: list[str]) -> list[dict]:
    """Convenience wrapper for processing many rows at once (e.g. in a notebook)."""
    return [normalize_date(d) for d in date_raw_list]