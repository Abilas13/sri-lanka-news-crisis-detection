"""
database.py

Purpose
-------
Single place for all database connectivity and schema management.
Every other module imports get_engine() / get_connection() from here rather
than opening its own MySQL connection.

Input
-----
Reads connection settings from config.py (which reads from .env).

Output
------
- A SQLAlchemy Engine for the project's MySQL database.
- init_schema(): creates the 8 processed-data tables if they don't exist.
  Never touches or alters `scraped_news`.

Notes
-----
- Uses SQLAlchemy Core (raw SQL via text()) rather than an ORM, since the
  schema is small and fixed — this keeps things simple and readable, which
  matters more than ORM abstraction for a project this size.
- Schema DDL lives in sql/create_tables.sql and sql/indexes.sql, not in this
  file — init_schema() reads and executes those files. This keeps the SQL
  reviewable/editable on its own, separate from Python logic.
- All CREATE TABLE statements use `IF NOT EXISTS`, so init_schema() is safe
  to run repeatedly. Index creation is NOT re-run-safe on every MySQL
  version (CREATE INDEX has no IF NOT EXISTS in older MySQL), so
  init_schema() only creates indexes the first time — see create_indexes().
"""

import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError, ProgrammingError

import config

# Project root is one level up from src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SQL_DIR = _PROJECT_ROOT / "sql"

_engine = None


def get_engine():
    """
    Returns a singleton SQLAlchemy engine connected to the MySQL database.
    Creating the engine does not open a connection immediately — SQLAlchemy
    connects lazily on first use, and pools connections after that.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(config.DB_URL, pool_pre_ping=True, future=True)
    return _engine


@contextmanager
def get_connection():
    """
    Context manager yielding a live DB connection, committing on success
    and rolling back on error. Use like:

        with get_connection() as conn:
            conn.execute(text("SELECT ..."))
    """
    engine = get_engine()
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except SQLAlchemyError:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """
    Quick sanity check that credentials and network access are correct.
    Returns True on success, raises a clear error otherwise.
    """
    try:
        with get_connection() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError as e:
        raise ConnectionError(
            f"Could not connect to MySQL at {config.DB_HOST}:{config.DB_PORT}/"
            f"{config.DB_NAME}. Check your .env credentials. Original error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Schema (DDL lives in sql/create_tables.sql and sql/indexes.sql)
# ---------------------------------------------------------------------------

def _read_sql_statements(filename: str) -> list[str]:
    """
    Reads a .sql file and splits it into individual executable statements.

    Comment lines (starting with --) are stripped line-by-line BEFORE
    splitting on semicolons. This matters: a leading comment block with no
    semicolon of its own would otherwise get glued onto the next statement
    when splitting naively, making the merged chunk start with "--" and
    causing the whole statement (comment AND the real SQL after it) to be
    mistaken for a comment-only fragment and silently dropped.
    """
    path = _SQL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected SQL file not found: {path}")

    raw = path.read_text(encoding="utf-8")

    # Strip comment lines individually, BEFORE splitting on ";"
    code_lines = []
    for line in raw.splitlines():
        if line.strip().startswith("--"):
            continue
        code_lines.append(line)
    code_only = "\n".join(code_lines)

    statements = []
    for chunk in code_only.split(";"):
        stmt = chunk.strip()
        if stmt:
            statements.append(stmt)
    return statements


def create_tables():
    """
    Executes every statement in sql/create_tables.sql — table creation first
    (safe to re-run via IF NOT EXISTS), then foreign key ALTER TABLE
    statements. Commits after EVERY statement individually (not just at the
    end) — MySQL 8's atomic DDL can otherwise roll back an earlier
    successful CREATE/ALTER together with a later failing one if they share
    an uncommitted transaction. A duplicate-constraint error on re-run is
    treated as a no-op rather than a failure.
    """
    engine = get_engine()
    with engine.connect() as conn:
        for statement in _read_sql_statements("create_tables.sql"):
            try:
                conn.execute(text(statement))
                conn.commit()   # commit immediately, don't batch with later statements
            except ProgrammingError as e:
                conn.rollback()
                if "Duplicate" in str(e) or "already exists" in str(e):
                    continue
                raise
    print("create_tables.sql applied.")


def create_indexes():
    """
    Executes every statement in sql/indexes.sql.
    Commits after every statement individually, same reasoning as
    create_tables(). CREATE INDEX has no IF NOT EXISTS in MySQL, so a
    duplicate-index error on re-run is caught and treated as a no-op.

    Each statement gets one retry after a short delay if it fails — this
    quietly absorbs the transient "table doesn't exist" error some Windows
    setups produce when antivirus briefly locks a just-created InnoDB file.
    """
    import time
    engine = get_engine()
    with engine.connect() as conn:
        for statement in _read_sql_statements("indexes.sql"):
            for attempt in (1, 2):
                try:
                    conn.execute(text(statement))
                    conn.commit()
                    break
                except ProgrammingError as e:
                    conn.rollback()
                    if "Duplicate key name" in str(e):
                        break  # index already exists, fine
                    if attempt == 1:
                        time.sleep(1.5)  # give the OS/AV a moment, then retry once
                        continue
                    raise
    print("indexes.sql applied.")


def init_schema():
    """
    Full schema setup: tables first (required), then indexes (best-effort).
    Safe to call every time the app starts. Does NOT touch scraped_news.

    Index creation is wrapped in a try/except: indexes are a performance
    optimization, not something correctness depends on, and on some Windows
    setups antivirus software briefly locks a freshly-written InnoDB table
    file right after creation, causing the very next statement touching it
    to fail spuriously. If that happens here, tables are still fully usable
    — just re-run create_indexes() on its own later (e.g. after a few
    seconds, or directly in MySQL Workbench) to pick up any indexes that
    didn't get created.
    """
    create_tables()
    try:
        create_indexes()
    except SQLAlchemyError as e:
        print(
            "WARNING: index creation failed (tables themselves are fine and "
            "usable). This is often a transient Windows/antivirus file-lock "
            "issue right after table creation, not a schema problem. "
            "You can safely continue — re-run create_indexes() later to retry.\n"
            f"Original error: {e}"
        )
    print("Schema check complete: all processed-data tables exist (indexes best-effort).")


def get_unprocessed_article_ids(limit: int = None) -> list[int]:
    """
    Returns article_ids from scraped_news that either:
      - have no row yet in processed_news, or
      - have processing_status = 'FAILED' (so they get retried)

    Input:  optional limit (int) to cap batch size
    Output: list of integer ids
    """
    query = f"""
        SELECT s.id
        FROM {config.RAW_TABLE} s
        LEFT JOIN processed_news p ON s.id = p.article_id
        WHERE p.article_id IS NULL OR p.processing_status = 'FAILED'
        ORDER BY s.id
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    with get_connection() as conn:
        result = conn.execute(text(query))
        return [row[0] for row in result]


def get_raw_article(article_id: int) -> dict | None:
    """
    Fetches a single raw article from scraped_news by id.
    Returns a dict with keys: id, title, content, date, link — or None if
    not found.
    """
    query = f"SELECT id, title, content, date, link FROM {config.RAW_TABLE} WHERE id = :aid"
    with get_connection() as conn:
        result = conn.execute(text(query), {"aid": article_id}).mappings().first()
        return dict(result) if result else None


def log_stage(article_id: int, stage: str, status: str, message: str = ""):
    """
    Writes one row to processing_log. Called by every pipeline stage so
    failures are traceable without stopping the whole batch.
    """
    from datetime import datetime
    query = """
        INSERT INTO processing_log (article_id, stage, status, message, timestamp)
        VALUES (:aid, :stage, :status, :message, :ts)
    """
    with get_connection() as conn:
        conn.execute(text(query), {
            "aid": article_id,
            "stage": stage,
            "status": status,
            "message": message,
            "ts": datetime.now(),
        })


def upsert_processed_articles(rows: list[dict]):
    """
    Batch insert/update into processed_news. Uses INSERT ... ON DUPLICATE KEY
    UPDATE so re-running on already-processed (or previously FAILED)
    articles works without needing a separate delete step first.

    Input: list of dicts, each matching processed_news columns. Missing
    keys default to NULL / current pipeline_version.
    """
    if not rows:
        return

    from datetime import datetime
    columns = [
        "article_id", "title_original", "title_clean", "content_original",
        "content_clean", "date_raw", "published_at", "publication_date",
        "publication_year", "publication_month", "publication_week",
        "publication_day", "article_length_words", "article_length_chars",
        "sentence_count", "duplicate_status", "duplicate_of", "content_hash",
        "processing_status", "processing_timestamp", "pipeline_version",
    ]
    placeholders = ", ".join(f":{c}" for c in columns)
    col_list = ", ".join(columns)
    update_clause = ", ".join(f"{c}=VALUES({c})" for c in columns if c != "article_id")

    query = f"""
        INSERT INTO processed_news ({col_list})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
    """

    now = datetime.now()
    prepared = []
    for row in rows:
        r = {c: row.get(c) for c in columns}
        r["processing_timestamp"] = now
        r.setdefault("pipeline_version", config.PIPELINE_VERSION)
        r["pipeline_version"] = row.get("pipeline_version", config.PIPELINE_VERSION)
        prepared.append(r)

    with get_connection() as conn:
        conn.execute(text(query), prepared)


def get_existing_content_hashes() -> set[str]:
    """Returns the set of all content_hash values already in processed_news
    (used for exact-duplicate checks when processing a single new article)."""
    query = "SELECT DISTINCT content_hash FROM processed_news WHERE content_hash IS NOT NULL"
    with get_connection() as conn:
        return {row[0] for row in conn.execute(text(query))}


def get_processed_articles_by_date(publication_date) -> list[dict]:
    """
    Returns already-processed articles published on the given date
    (used for near-duplicate comparison when processing a single new article
    against same-day articles already in the DB).
    """
    query = """
        SELECT article_id, content_clean, link
        FROM processed_news pn
        JOIN scraped_news sn ON pn.article_id = sn.id
        WHERE pn.publication_date = :pdate AND pn.duplicate_status != 'EXACT_DUPLICATE'
    """
    with get_connection() as conn:
        result = conn.execute(text(query), {"pdate": publication_date}).mappings().all()
        return [dict(r) for r in result]


def get_article_id_by_content_hash(content_hash: str) -> int | None:
    """Returns the article_id of the first processed_news row with this
    content_hash, or None. Used for single-article exact-duplicate lookup
    in pipeline.process_new_article()."""
    query = "SELECT article_id FROM processed_news WHERE content_hash = :h LIMIT 1"
    with get_connection() as conn:
        result = conn.execute(text(query), {"h": content_hash}).first()
        return result[0] if result else None


def get_articles_needing_sentiment(limit: int = None) -> list[dict]:
    """
    Returns COMPLETED articles that don't have an article_sentiment row yet.
    Used by sentiment.process_all_sentiment().
    """
    query = """
        SELECT pn.article_id, pn.content_clean
        FROM processed_news pn
        LEFT JOIN article_sentiment s ON pn.article_id = s.article_id
        WHERE pn.processing_status = 'COMPLETED' AND s.article_id IS NULL
        ORDER BY pn.article_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        result = conn.execute(text(query)).mappings().all()
        return [dict(r) for r in result]


def upsert_article_sentiment(rows: list[dict]):
    """
    Batch insert/update into article_sentiment.
    Input: list of dicts with article_id, sentiment, sentiment_score, sentiment_model.
    """
    if not rows:
        return
    query = """
        INSERT INTO article_sentiment (article_id, sentiment, sentiment_score, sentiment_model)
        VALUES (:article_id, :sentiment, :sentiment_score, :sentiment_model)
        ON DUPLICATE KEY UPDATE
            sentiment = VALUES(sentiment),
            sentiment_score = VALUES(sentiment_score),
            sentiment_model = VALUES(sentiment_model)
    """
    with get_connection() as conn:
        conn.execute(text(query), rows)


def get_articles_for_topic_modeling(limit: int = None) -> list[dict]:
    """Returns all COMPLETED articles' title_clean + content_clean for BERTopic fitting."""
    query = """
        SELECT article_id, title_clean, content_clean
        FROM processed_news
        WHERE processing_status = 'COMPLETED'
        ORDER BY article_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        result = conn.execute(text(query)).mappings().all()
        return [dict(r) for r in result]


def clear_article_topics():
    """Deletes all article_topics rows — used before a full refit, since
    topic_id numbering changes each time BERTopic is refit on the corpus."""
    with get_connection() as conn:
        conn.execute(text("DELETE FROM article_topics"))


def upsert_article_topics(rows: list[dict]):
    """
    Batch insert into article_topics.
    Input: list of dicts with article_id, topic_id, topic_name, topic_probability.
    """
    if not rows:
        return
    query = """
        INSERT INTO article_topics (article_id, topic_id, topic_name, topic_probability)
        VALUES (:article_id, :topic_id, :topic_name, :topic_probability)
        ON DUPLICATE KEY UPDATE
            topic_name = VALUES(topic_name),
            topic_probability = VALUES(topic_probability)
    """
    with get_connection() as conn:
        conn.execute(text(query), rows)


def update_topic_names(name_mapping: dict[int, str]):
    """
    Updates topic_name for every article_topics row matching each topic_id
    in the mapping. Used by topics.apply_topic_names().
    """
    if not name_mapping:
        return
    query = "UPDATE article_topics SET topic_name = :name WHERE topic_id = :tid"
    params = [{"tid": tid, "name": name} for tid, name in name_mapping.items()]
    with get_connection() as conn:
        conn.execute(text(query), params)


def get_articles_needing_keywords(limit: int = None) -> list[dict]:
    """
    Returns COMPLETED articles that don't have any article_keywords rows yet.
    Used by keywords.process_all_keywords().
    """
    query = """
        SELECT pn.article_id, pn.content_clean
        FROM processed_news pn
        LEFT JOIN article_keywords k ON pn.article_id = k.article_id
        WHERE pn.processing_status = 'COMPLETED' AND k.article_id IS NULL
        ORDER BY pn.article_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        result = conn.execute(text(query)).mappings().all()
        return [dict(r) for r in result]


def upsert_article_keywords(rows: list[dict]):
    """
    Batch insert into article_keywords.
    Input: list of dicts with article_id, keyword, score, rank.
    """
    if not rows:
        return
    query = """
        INSERT INTO article_keywords (article_id, keyword, score, `rank`)
        VALUES (:article_id, :keyword, :score, :rank)
        ON DUPLICATE KEY UPDATE
            keyword = VALUES(keyword),
            score = VALUES(score)
    """
    with get_connection() as conn:
        conn.execute(text(query), rows)


def get_articles_needing_ner(limit: int = None) -> list[dict]:
    """
    Returns COMPLETED articles that don't have any article_entities rows yet.
    Used by ner.process_all_ner().
    """
    query = """
        SELECT pn.article_id, pn.content_clean
        FROM processed_news pn
        LEFT JOIN article_entities e ON pn.article_id = e.article_id
        WHERE pn.processing_status = 'COMPLETED' AND e.article_id IS NULL
        ORDER BY pn.article_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        result = conn.execute(text(query)).mappings().all()
        return [dict(r) for r in result]


def upsert_article_entities(rows: list[dict]):
    """
    Batch insert into article_entities.
    Input: list of dicts with article_id, entity, entity_type, confidence.
    """
    if not rows:
        return
    query = """
        INSERT INTO article_entities (article_id, entity, entity_type, confidence)
        VALUES (:article_id, :entity, :entity_type, :confidence)
        ON DUPLICATE KEY UPDATE
            confidence = VALUES(confidence)
    """
    with get_connection() as conn:
        conn.execute(text(query), rows)


def get_articles_needing_entities(limit: int = None) -> list[dict]:
    """
    Returns COMPLETED articles that don't have any article_entities rows yet.
    Used by ner.process_all_entities().
    """
    query = """
        SELECT pn.article_id, pn.content_clean
        FROM processed_news pn
        LEFT JOIN article_entities e ON pn.article_id = e.article_id
        WHERE pn.processing_status = 'COMPLETED' AND e.article_id IS NULL
        ORDER BY pn.article_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        result = conn.execute(text(query)).mappings().all()
        return [dict(r) for r in result]


def upsert_article_entities(rows: list[dict]):
    """
    Batch insert into article_entities.
    Input: list of dicts with article_id, entity, entity_type, confidence.
    """
    if not rows:
        return
    query = """
        INSERT INTO article_entities (article_id, entity, entity_type, confidence)
        VALUES (:article_id, :entity, :entity_type, :confidence)
        ON DUPLICATE KEY UPDATE
            confidence = VALUES(confidence)
    """
    with get_connection() as conn:
        conn.execute(text(query), rows)


if __name__ == "__main__":
    # Running this file directly does a full setup check.
    test_connection()
    print("Connection OK.")
    init_schema()