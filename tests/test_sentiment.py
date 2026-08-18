"""
test_sentiment.py

How to run:
    python tests/test_sentiment.py

Mocks the transformer model AND the database layer, so this runs fast with
no model download and no real MySQL connection needed. Verifies the
batching/mapping/upsert LOGIC is correct. Your real check is running
sentiment.process_all_sentiment() against the actual DB separately (which
will genuinely download and run the model — expect it to take a while the
first time, since it has to fetch the model weights).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentiment


class _FakePipeline:
    """Stands in for the real HuggingFace pipeline — returns deterministic
    fake predictions based on simple keyword matching, so test assertions
    are meaningful without needing the real model."""
    def __call__(self, texts, batch_size=32, truncation=True):
        outputs = []
        for t in texts:
            if "great" in t.lower() or "success" in t.lower():
                outputs.append({"label": "positive", "score": 0.95})
            elif "crisis" in t.lower() or "shortage" in t.lower():
                outputs.append({"label": "negative", "score": 0.88})
            else:
                outputs.append({"label": "neutral", "score": 0.60})
        return outputs


def run():
    # monkeypatch the model loader so no real download happens
    sentiment._pipeline = _FakePipeline()
    sentiment._get_pipeline = lambda: sentiment._pipeline

    print("1. predict_sentiment_batch: mixed real texts")
    texts = [
        "The economy showed great success this quarter.",
        "Fuel shortage causes crisis across the country.",
        "The meeting was held on Tuesday.",
    ]
    results = sentiment.predict_sentiment_batch(texts)
    assert results[0]["sentiment"] == "positive"
    assert results[1]["sentiment"] == "negative"
    assert results[2]["sentiment"] == "neutral"
    print(f"   OK -> {results}\n")

    print("2. predict_sentiment_batch: empty string doesn't call the model")
    results = sentiment.predict_sentiment_batch(["", "  ", "great success"])
    assert results[0]["sentiment"] == "neutral" and results[0]["sentiment_score"] == 0.0
    assert results[1]["sentiment"] == "neutral" and results[1]["sentiment_score"] == 0.0
    assert results[2]["sentiment"] == "positive"
    print(f"   OK -> {results}\n")

    print("3. predict_sentiment_batch: order is preserved")
    texts = ["great success", "crisis shortage", "", "great success"]
    results = sentiment.predict_sentiment_batch(texts)
    assert [r["sentiment"] for r in results] == ["positive", "negative", "neutral", "positive"]
    print("   OK -> order preserved correctly\n")

    print("4. process_all_sentiment: mocked DB end-to-end")
    fake_articles = [
        {"article_id": 1, "content_clean": "The economy showed great success."},
        {"article_id": 2, "content_clean": "Fuel shortage causes crisis."},
        {"article_id": 3, "content_clean": "The meeting was held Tuesday."},
    ]
    written_rows = []

    # Stub sqlalchemy so database.py's module-level import succeeds without
    # it installed — this test mocks every DB-touching function anyway.
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
    db.get_articles_needing_sentiment = lambda limit=None: fake_articles
    db.upsert_article_sentiment = lambda rows: written_rows.extend(rows)

    sentiment.process_all_sentiment(batch_size=2)

    assert len(written_rows) == 3
    assert written_rows[0]["article_id"] == 1
    assert written_rows[0]["sentiment"] == "positive"
    assert written_rows[0]["sentiment_model"] == sentiment.MODEL_NAME
    assert written_rows[1]["sentiment"] == "negative"
    print(f"   OK -> wrote {len(written_rows)} rows, e.g. {written_rows[0]}\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()