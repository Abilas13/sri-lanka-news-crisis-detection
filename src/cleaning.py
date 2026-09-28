
import re
from bs4 import BeautifulSoup

import config
import utils

logger = utils.setup_logger(__name__)

try:
    import ftfy
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False
    logger.warning(
        "ftfy not installed — encoding-fix step will be skipped. "
        "Run: pip install ftfy"
    )

# Common scraper/site boilerplate seen in Sri Lankan news sites — extend this
# list as you spot more during Step 2/notebook review. Matched case-insensitively.
_BOILERPLATE_PATTERNS = [
    r"share this article.*",
    r"follow us on (facebook|twitter|instagram|whatsapp).*",
    r"click here to (read more|subscribe|download).*",
    r"also read:.*",
    r"related (articles?|news):.*",
    r"(read|see) more:.*",
    r"advertisement\s*",
    r"^\s*\(with inputs from.*\)\s*$",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

# A crude but effective sentence splitter — good enough for a word/sentence
# COUNT (not for NLP correctness). Splits on . ! ? followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Step 2: Validation
# ---------------------------------------------------------------------------

def validate_article(article: dict) -> tuple[bool, str]:
    """
    Checks an article dict (as returned by database.get_raw_article) for the
    minimum required fields and content quality.

    Returns (is_valid, reason). reason is '' if valid, otherwise a short
    machine-readable code suitable for processing_log.message.
    """
    required_fields = ["id", "title", "content", "date", "link"]
    for field in required_fields:
        if field not in article or article[field] is None:
            return False, f"MISSING_FIELD:{field}"

    title = str(article["title"]).strip()
    content = str(article["content"]).strip()

    if not title:
        return False, "EMPTY_TITLE"
    if not content:
        return False, "EMPTY_CONTENT"
    if len(content) < config.MIN_CONTENT_LENGTH_CHARS:
        return False, f"CONTENT_TOO_SHORT:{len(content)}chars"

    return True, ""


# ---------------------------------------------------------------------------
# Step 3: Text cleaning
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Removes HTML tags, keeping the visible text."""
    if "<" not in text and ">" not in text:
        return text  # skip BeautifulSoup overhead when there's clearly no HTML
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def _fix_encoding(text: str) -> str:
    """Fixes mojibake / encoding artifacts (e.g. â€™ -> '). No-op if ftfy isn't installed."""
    if not _FTFY_AVAILABLE:
        return text
    return ftfy.fix_text(text)


def _remove_boilerplate(text: str) -> str:
    """Strips known scraper/site boilerplate phrases."""
    return _BOILERPLATE_RE.sub("", text)


def _normalize_whitespace(text: str) -> str:
    """Collapses repeated whitespace/newlines into single spaces, trims ends,
    and removes spurious spaces before punctuation left by HTML tag stripping
    (e.g. 'rise .' -> 'rise.')."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def clean_text(raw_text: str) -> str:
    """
    Full cleaning pipeline for one piece of text (title or content).
    Order matters: fix encoding first (so boilerplate regexes match
    correctly), then strip HTML, then boilerplate, then whitespace last.
    """
    if not raw_text:
        return ""
    text = _fix_encoding(str(raw_text))
    text = _strip_html(text)
    text = _remove_boilerplate(text)
    text = _normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Step 6: Basic length features
# ---------------------------------------------------------------------------

def compute_basic_features(content_clean: str) -> dict:
    """Word count, char count, and a rough sentence count on CLEANED content."""
    words = content_clean.split()
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(content_clean) if s.strip()]
    return {
        "article_length_words": len(words),
        "article_length_chars": len(content_clean),
        "sentence_count": len(sentences),
    }


# ---------------------------------------------------------------------------
# Combined entry point used by pipeline.py
# ---------------------------------------------------------------------------

def clean_article(title: str, content: str) -> dict:
    """
    Runs the full Step 3 + Step 6 pipeline on one article's title/content.
    Does NOT validate — call validate_article() first and skip cleaning
    entirely for invalid articles (no point cleaning text you're going to
    mark FAILED anyway).
    """
    title_clean = clean_text(title)
    content_clean = clean_text(content)
    features = compute_basic_features(content_clean)
    return {
        "title_clean": title_clean,
        "content_clean": content_clean,
        **features,
    }