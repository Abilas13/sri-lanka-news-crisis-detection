"""
test_pipeline.py


Uses an in-memory fake in place of database.py's DB-touching functions, so
this test runs fast and doesn't need a real MySQL connection. It verifies
the pipeline's LOGIC (validation -> date -> clean -> dedup -> the right
shape of data being written), not the actual SQL. Your real end-to-end
check is running pipeline.process_existing_articles() against the real DB
separately.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Stub sqlalchemy for this test only ---
# database.py imports sqlalchemy at module level, but this test monkeypatches
# every database.py function that actually touches the DB — so a minimal
# stub is enough to satisfy the import without needing sqlalchemy installed.
# (On the real machine, sqlalchemy IS installed via requirements.txt — this
# stub only exists to let this specific mocked test run standalone.)
import types
if "sqlalchemy" not in sys.modules:
    fake_sa = types.ModuleType("sqlalchemy")
    fake_sa.create_engine = lambda *a, **k: None
    fake_sa.text = lambda q: q
    fake_sa_exc = types.ModuleType("sqlalchemy.exc")
    class _FakeExc(Exception): pass
    fake_sa_exc.OperationalError = _FakeExc
    fake_sa_exc.SQLAlchemyError = _FakeExc
    fake_sa_exc.ProgrammingError = _FakeExc
    fake_sa.exc = fake_sa_exc
    sys.modules["sqlalchemy"] = fake_sa
    sys.modules["sqlalchemy.exc"] = fake_sa_exc

import database as db
import pipeline

# ---------------------------------------------------------------------------
# Fake in-memory "database" — replaces the real db functions for this test
# ---------------------------------------------------------------------------

_FAKE_RAW = {
    1: {"id": 1, "title": "Fuel prices rise", "content": "Fuel prices rose sharply across Colombo today amid ongoing shortages nationwide.", "date": "01 Apr 2026", "link": "https://x.com/1"},
    2: {"id": 2, "title": "Fuel prices rise", "content": "Fuel prices rose sharply across Colombo today amid ongoing shortages nationwide.", "date": "01 Apr 2026", "link": "https://x.com/1"},  # exact dup of 1
    3: {"id": 3, "title": "Cricket match today", "content": "The national cricket team won their match yesterday evening in Kandy.", "date": "01 Apr 2026", "link": "https://x.com/3"},
    4: {"id": 4, "title": "", "content": "short", "date": "01 Apr 2026", "link": "https://x.com/4"},  # invalid: empty title + too short
}
_FAKE_LOG = []
_FAKE_PROCESSED = {}


def fake_get_raw_article(article_id):
    return _FAKE_RAW.get(article_id)

def fake_log_stage(article_id, stage, status, message=""):
    _FAKE_LOG.append((article_id, stage, status, message))

def fake_upsert_processed_articles(rows):
    for row in rows:
        _FAKE_PROCESSED[row["article_id"]] = row

def fake_get_unprocessed_article_ids(limit=None):
    return list(_FAKE_RAW.keys())

def fake_get_existing_content_hashes():
    return {r["content_hash"] for r in _FAKE_PROCESSED.values() if r.get("content_hash")}

def fake_get_processed_articles_by_date(pdate):
    return [
        {"article_id": r["article_id"], "content_clean": r["content_clean"], "link": r.get("link", "")}
        for r in _FAKE_PROCESSED.values()
        if r.get("publication_date") == pdate and r["duplicate_status"] != "EXACT_DUPLICATE"
    ]

def fake_get_article_id_by_content_hash(content_hash):
    for r in _FAKE_PROCESSED.values():
        if r.get("content_hash") == content_hash:
            return r["article_id"]
    return None


def run():
    # monkeypatch
    db.get_raw_article = fake_get_raw_article
    db.log_stage = fake_log_stage
    db.upsert_processed_articles = fake_upsert_processed_articles
    db.get_unprocessed_article_ids = fake_get_unprocessed_article_ids
    db.get_existing_content_hashes = fake_get_existing_content_hashes
    db.get_processed_articles_by_date = fake_get_processed_articles_by_date
    db.get_article_id_by_content_hash = fake_get_article_id_by_content_hash

    print("1. process_existing_articles on 4 synthetic articles (1 exact dup, 1 invalid)")
    pipeline.process_existing_articles(batch_size=10)

    assert _FAKE_PROCESSED[1]["duplicate_status"] == "UNIQUE"
    assert _FAKE_PROCESSED[2]["duplicate_status"] == "EXACT_DUPLICATE"
    assert _FAKE_PROCESSED[2]["duplicate_of"] == 1
    assert _FAKE_PROCESSED[3]["duplicate_status"] == "UNIQUE"
    assert _FAKE_PROCESSED[4]["processing_status"] == "FAILED"
    print("   OK -> article 1 UNIQUE, article 2 EXACT_DUPLICATE of 1, "
          "article 3 UNIQUE, article 4 FAILED validation\n")

    print("2. process_new_article on a NEW article that duplicates an existing processed one")
    _FAKE_RAW[5] = {"id": 5, "title": "Fuel prices rise", "content": "Fuel prices rose sharply across Colombo today amid ongoing shortages nationwide.", "date": "01 Apr 2026", "link": "https://x.com/5"}
    pipeline.process_new_article(5)
    assert _FAKE_PROCESSED[5]["duplicate_status"] == "EXACT_DUPLICATE"
    print(f"   OK -> article 5 correctly detected as EXACT_DUPLICATE of {_FAKE_PROCESSED[5]['duplicate_of']}\n")

    print("3. process_new_article on a genuinely new, unique article")
    _FAKE_RAW[6] = {"id": 6, "title": "New election results", "content": "The election commission announced final results late Tuesday night in Colombo.", "date": "01 Apr 2026", "link": "https://x.com/6"}
    pipeline.process_new_article(6)
    assert _FAKE_PROCESSED[6]["duplicate_status"] == "UNIQUE"
    print("   OK -> article 6 correctly UNIQUE\n")

    print("4. processing_log captured entries for every stage")
    stages_logged = {entry[1] for entry in _FAKE_LOG}
    assert "VALIDATION" in stages_logged
    assert "DATE_NORMALIZATION" in stages_logged
    assert "CLEANING" in stages_logged
    assert "DEDUPLICATION" in stages_logged
    print(f"   OK -> stages logged: {stages_logged}\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()