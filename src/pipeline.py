"""
pipeline.py

Purpose
-------
Wires cleaning.py + date_processing.py + deduplication.py + database.py
together into the two functions Step 26 requires:

    process_existing_articles()  — batch-processes the whole scraped_news
                                    backlog, skipping already-COMPLETED rows.
    process_new_article(id)      — processes one newly-scraped article.

Current scope: validation -> date normalization -> cleaning -> dedup ->
storage in processed_news. Sentiment/topics/keywords/NER are NOT called
here yet — those modules don't exist yet. Once they do, this file gets
extended (not rewritten) to call them after cleaning succeeds.

Design note on batch vs. single-article dedup
-----------------------------------------------
Deduplication is inherently a corpus-level operation (you can only find a
duplicate by comparing against other articles). process_existing_articles()
runs deduplication.assign_duplicate_status() across the whole batch being
processed, grouped by day, per its own design.

process_new_article() only has ONE new row to work with, so it instead
compares that single article against already-processed articles from the
same day (pulled from the DB) plus a corpus-wide content-hash set — see
_check_single_article_duplicate() below.
"""

import pandas as pd
from datetime import datetime

import config
import database as db
import cleaning
import date_processing as dp
import deduplication as dedup
import utils

logger = utils.setup_logger(__name__)


def _process_one_article_core(article: dict) -> dict | None:
    """
    Runs validation -> date normalization -> cleaning for one raw article.
    Returns a dict of processed_news fields (WITHOUT duplicate_status set
    yet — that's handled separately since it's corpus-level), or None if
    validation failed (in which case a FAILED row is written and logged
    directly, so callers don't need to handle that case).
    """
    article_id = article["id"]

    is_valid, reason = cleaning.validate_article(article)
    if not is_valid:
        db.log_stage(article_id, "VALIDATION", "FAILED", reason)
        db.upsert_processed_articles([{
            "article_id": article_id,
            "title_original": article.get("title"),
            "content_original": article.get("content"),
            "date_raw": article.get("date"),
            "processing_status": "FAILED",
            "pipeline_version": config.PIPELINE_VERSION,
        }])
        return None
    db.log_stage(article_id, "VALIDATION", "SUCCESS")

    date_result = dp.normalize_date(article.get("date"))
    if date_result["date_parse_status"] != "OK":
        db.log_stage(article_id, "DATE_NORMALIZATION", "FAILED", f"raw={article.get('date')!r}")
    else:
        db.log_stage(article_id, "DATE_NORMALIZATION", "SUCCESS")

    clean_result = cleaning.clean_article(article.get("title"), article.get("content"))
    db.log_stage(article_id, "CLEANING", "SUCCESS")

    return {
        "article_id": article_id,
        "title_original": article.get("title"),
        "title_clean": clean_result["title_clean"],
        "content_original": article.get("content"),
        "content_clean": clean_result["content_clean"],
        "date_raw": article.get("date"),
        "published_at": date_result["published_at"],
        "publication_date": date_result["publication_date"],
        "publication_year": date_result["publication_year"],
        "publication_month": date_result["publication_month"],
        "publication_week": date_result["publication_week"],
        "publication_day": date_result["publication_day"],
        "article_length_words": clean_result["article_length_words"],
        "article_length_chars": clean_result["article_length_chars"],
        "sentence_count": clean_result["sentence_count"],
        "link": article.get("link"),
        "pipeline_version": config.PIPELINE_VERSION,
    }


def process_existing_articles(batch_size: int = 500):
    """
    Processes every unprocessed (or previously FAILED) article in
    scraped_news, in batches. Deduplication runs per-batch, grouped by day
    within deduplication.py — batch_size just controls memory/DB round-trip
    size, it doesn't affect correctness since near-dup grouping is by date,
    not by batch position, EXCEPT for the edge case where the same day's
    articles are split across two different batches. To avoid that, articles
    are fetched in article_id order and MySQL rows for a given scrape are
    typically inserted in date order, but if you're re-running on a
    heavily out-of-order backlog, consider batch_size=None to process
    everything in one pass for full correctness on the dedup step.
    """
    ids = db.get_unprocessed_article_ids()
    total = len(ids)
    logger.info(f"process_existing_articles: {total} articles to process")

    if total == 0:
        print("Nothing to process — all articles already COMPLETED.")
        return

    processed_count = 0
    failed_count = 0

    for start in range(0, total, batch_size):
        batch_ids = ids[start:start + batch_size]
        cleaned_rows = []

        for article_id in batch_ids:
            article = db.get_raw_article(article_id)
            if article is None:
                logger.warning(f"article_id {article_id} not found in scraped_news, skipping")
                continue
            result = _process_one_article_core(article)
            if result is None:
                failed_count += 1
            else:
                cleaned_rows.append(result)

        if cleaned_rows:
            df = pd.DataFrame(cleaned_rows)
            df = dedup.assign_duplicate_status(df)

            final_rows = df.to_dict(orient="records")
            for row in final_rows:
                row["processing_status"] = "COMPLETED"

            db.upsert_processed_articles(final_rows)
            for row in final_rows:
                db.log_stage(row["article_id"], "DEDUPLICATION", "SUCCESS", row["duplicate_status"])

            processed_count += len(final_rows)

        print(f"  Progress: {min(start + batch_size, total)}/{total} "
              f"(completed={processed_count}, failed={failed_count})")

    print(f"\nDone. Completed: {processed_count}, Failed validation: {failed_count}")


def _check_single_article_duplicate(row: dict) -> dict:
    """
    For process_new_article(): compares ONE new article against the
    already-processed corpus. Returns duplicate_status/duplicate_of to
    merge into the row.
    """
    existing_hashes = db.get_existing_content_hashes()
    if row["content_hash"] in existing_hashes:
        match_id = db.get_article_id_by_content_hash(row["content_hash"])
        return {"duplicate_status": "EXACT_DUPLICATE", "duplicate_of": match_id}

    same_day_articles = db.get_processed_articles_by_date(row["publication_date"])
    if same_day_articles:
        same_day_articles.append({"article_id": row["article_id"], "content_clean": row["content_clean"]})
        temp_df = pd.DataFrame(same_day_articles)
        temp_df["publication_date"] = row["publication_date"]
        temp_df["link"] = temp_df.get("link", "")
        result_df = dedup._mark_near_duplicates_same_day(
            temp_df.assign(duplicate_status="UNIQUE", duplicate_of=None)
        )
        this_row = result_df[result_df["article_id"] == row["article_id"]].iloc[0]
        if this_row["duplicate_status"] == "NEAR_DUPLICATE":
            return {"duplicate_status": "NEAR_DUPLICATE", "duplicate_of": this_row["duplicate_of"]}

    return {"duplicate_status": "UNIQUE", "duplicate_of": None}


def process_new_article(article_id: int):
    """
    Processes a single newly-scraped article end to end: validate -> date
    normalize -> clean -> check duplicate against existing processed corpus
    -> store. This is what gets called as new rows land in scraped_news.
    """
    article = db.get_raw_article(article_id)
    if article is None:
        logger.error(f"article_id {article_id} not found in scraped_news")
        return

    result = _process_one_article_core(article)
    if result is None:
        print(f"Article {article_id} FAILED validation — see processing_log.")
        return

    result["content_hash"] = dedup.compute_content_hash(result["content_clean"])
    dup_info = _check_single_article_duplicate(result)
    result.update(dup_info)
    result["processing_status"] = "COMPLETED"

    db.upsert_processed_articles([result])
    db.log_stage(article_id, "DEDUPLICATION", "SUCCESS", result["duplicate_status"])
    print(f"Article {article_id} processed: {result['duplicate_status']}")


if __name__ == "__main__":
    process_existing_articles()