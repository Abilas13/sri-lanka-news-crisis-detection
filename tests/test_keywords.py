"""
test_keywords.py

How to run:
    python tests/test_keywords.py

Mocks KeyBERT and the database layer — verifies batching, rank assignment,
empty-text handling, and DB wiring without needing keybert/sentence-transformers
installed or a real MySQL connection.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keywords


class _FakeKeyBERT:
    """Returns deterministic fake keywords based on simple content matching."""
    def extract_keywords(self, docs, keyphrase_ngram_range=(1, 2), stop_words="english",
                          use_mmr=True, diversity=0.5, top_n=10):
        results = []
        for doc in docs:
            if "fuel" in doc.lower():
                results.append([("fuel shortage", 0.72), ("fuel prices", 0.65), ("colombo", 0.31)])
            elif "election" in doc.lower():
                results.append([("presidential election", 0.80), ("results", 0.40)])
            else:
                results.append([])
        return results


def run():
    keywords._keybert_model = _FakeKeyBERT()
    keywords._get_keybert_model = lambda: keywords._keybert_model

    print("1. extract_keywords_batch: mixed real texts")
    texts = [
        "Fuel shortage causes long queues across Colombo today.",
        "Presidential election results announced late Tuesday.",
        "The weather was mild today.",
    ]
    results = keywords.extract_keywords_batch(texts, top_n=10)
    assert len(results) == 3
    assert results[0][0]["keyword"] == "fuel shortage"
    assert results[0][0]["rank"] == 1
    assert results[0][1]["rank"] == 2
    assert results[1][0]["keyword"] == "presidential election"
    assert results[2] == []
    print(f"   OK -> {results}\n")

    print("2. extract_keywords_batch: empty string doesn't call the model")
    results = keywords.extract_keywords_batch(["", "  ", "fuel shortage today"])
    assert results[0] == []
    assert results[1] == []
    assert len(results[2]) == 3
    print(f"   OK -> {results}\n")

    print("3. process_all_keywords: mocked DB end-to-end")
    fake_articles = [
        {"article_id": 1, "content_clean": "Fuel shortage causes long queues."},
        {"article_id": 2, "content_clean": "Presidential election results announced."},
        {"article_id": 3, "content_clean": "The weather was mild today."},
    ]
    written_rows = []

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
    db.get_articles_needing_keywords = lambda limit=None: fake_articles
    db.upsert_article_keywords = lambda rows: written_rows.extend(rows)

    keywords.process_all_keywords(batch_size=2)

    assert len(written_rows) == 5  # 3 + 2 + 0 keywords across the 3 articles
    article_1_rows = [r for r in written_rows if r["article_id"] == 1]
    assert len(article_1_rows) == 3
    assert article_1_rows[0]["keyword"] == "fuel shortage"
    assert article_1_rows[0]["rank"] == 1
    print(f"   OK -> wrote {len(written_rows)} keyword rows total\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()