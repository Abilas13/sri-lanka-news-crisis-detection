"""
deduplication.py

Purpose
-------
Step 5: flag exact and near-duplicate articles. Never deletes anything —
only annotates duplicate_status / duplicate_of / content_hash.

Input
-----
A pandas DataFrame with at least: article_id, content_clean, link,
publication_date (from date_processing.py's output).

Output
------
The same DataFrame with three new columns:
    content_hash     - sha256 of normalized content
    duplicate_status  - 'UNIQUE' | 'EXACT_DUPLICATE' | 'NEAR_DUPLICATE'
    duplicate_of      - article_id of the original, or None

Design note on near-duplicates
-------------------------------
Full pairwise TF-IDF cosine similarity across the whole corpus (16k+
articles) is O(n^2) and mostly wasted work — near-duplicates (republished
or syndicated versions of the same story) almost always appear within the
same day or two, not months apart. So near-duplicate comparison is scoped
to same-day groups only. This keeps runtime reasonable without sacrificing
the cases that actually matter for this research.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import utils

logger = utils.setup_logger(__name__)

NEAR_DUPLICATE_THRESHOLD = 0.85   # cosine similarity above this = near-duplicate


def compute_content_hash(content_clean: str) -> str:
    """Wraps utils.content_hash for exact-duplicate detection."""
    return utils.content_hash(content_clean)


def _mark_exact_duplicates_by_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    For a given column (content_hash or link), marks the first-seen row per
    value as UNIQUE (untouched) and every subsequent row sharing that value
    as EXACT_DUPLICATE, pointing duplicate_of at the first-seen article_id.
    Only touches rows still marked UNIQUE — doesn't overwrite an
    already-flagged near-duplicate from a previous pass.
    """
    seen: dict[str, int] = {}
    for idx, row in df.iterrows():
        if df.at[idx, "duplicate_status"] != "UNIQUE":
            continue  # already flagged by an earlier check
        val = row[column]
        if not val:
            continue
        if val in seen:
            df.at[idx, "duplicate_status"] = "EXACT_DUPLICATE"
            df.at[idx, "duplicate_of"] = seen[val]
        else:
            seen[val] = row["article_id"]
    return df


def _mark_near_duplicates_same_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each publication_date group, TF-IDF vectorizes content_clean for
    rows still marked UNIQUE, computes pairwise cosine similarity, and marks
    any pair above NEAR_DUPLICATE_THRESHOLD (the later article_id points at
    the earlier one).
    """
    for date_val, group in df.groupby("publication_date"):
        candidates = group[group["duplicate_status"] == "UNIQUE"]
        if len(candidates) < 2:
            continue  # nothing to compare within this day

        texts = candidates["content_clean"].fillna("").tolist()
        ids = candidates["article_id"].tolist()
        idx_map = candidates.index.tolist()

        try:
            vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
            tfidf = vectorizer.fit_transform(texts)
        except ValueError:
            continue  # e.g. all-empty text in this group, skip safely

        sims = cosine_similarity(tfidf)

        for i in range(len(ids)):
            if df.at[idx_map[i], "duplicate_status"] != "UNIQUE":
                continue  # got marked by an earlier pair in this same loop
            for j in range(i + 1, len(ids)):
                if df.at[idx_map[j], "duplicate_status"] != "UNIQUE":
                    continue
                if sims[i, j] >= NEAR_DUPLICATE_THRESHOLD:
                    df.at[idx_map[j], "duplicate_status"] = "NEAR_DUPLICATE"
                    df.at[idx_map[j], "duplicate_of"] = ids[i]
    return df


def assign_duplicate_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Runs, in order:
      1. content_hash computation
      2. exact-duplicate marking by link
      3. exact-duplicate marking by content_hash
      4. near-duplicate marking within same-day groups (on remaining UNIQUE rows)

    Order matters: link and hash matches are cheap and certain, so they run
    first and shrink the pool that the more expensive TF-IDF step has to
    look at.
    """
    df = df.copy()
    df["content_hash"] = df["content_clean"].fillna("").apply(compute_content_hash)
    df["duplicate_status"] = "UNIQUE"
    df["duplicate_of"] = None

    df = _mark_exact_duplicates_by_column(df, "link")
    df = _mark_exact_duplicates_by_column(df, "content_hash")
    df = _mark_near_duplicates_same_day(df)

    counts = df["duplicate_status"].value_counts().to_dict()
    logger.info(f"Deduplication results: {counts}")
    return df