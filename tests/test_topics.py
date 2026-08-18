"""
test_topics.py

How to run:
    python tests/test_topics.py

Mocks BERTopic's fit_transform and the database layer — verifies the
orchestration logic (auto-naming, row building, clear-before-refit,
manual-naming round trip) without needing bertopic/sentence-transformers
installed or a real MySQL connection. Your real check is running
topics.process_all_topics() against the actual DB separately.
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

import topics


class _FakeBERTopicModel:
    """Stands in for a real fitted BERTopic model."""
    def get_topic(self, topic_id):
        fake_topics = {
            0: [("fuel", 0.5), ("prices", 0.4), ("shortage", 0.3)],
            1: [("election", 0.6), ("president", 0.5), ("results", 0.3)],
            -1: [],
        }
        return fake_topics.get(topic_id, [])

    def get_topic_info(self):
        import pandas as pd
        return pd.DataFrame({
            "Topic": [0, 1, -1],
            "Count": [2, 1, 1],
            "Name": ["0_fuel_prices_shortage", "1_election_president_results", "-1_outlier"],
        })

    def save(self, path, **kwargs):
        pass  # no-op for test


def run():
    print("1. _auto_topic_name: normal topic")
    model = _FakeBERTopicModel()
    name = topics._auto_topic_name(model, 0)
    assert name == "fuel_prices_shortage"
    print(f"   OK -> {name}\n")

    print("2. _auto_topic_name: outlier topic (-1)")
    name = topics._auto_topic_name(model, -1)
    assert name == "outlier_uncategorized"
    print(f"   OK -> {name}\n")

    print("3. process_all_topics: full mocked flow")
    fake_articles = [
        {"article_id": 1, "title_clean": "Fuel crisis", "content_clean": "Fuel prices rose sharply."},
        {"article_id": 2, "title_clean": "Election news", "content_clean": "President election results announced."},
        {"article_id": 3, "title_clean": "Fuel shortage continues", "content_clean": "Shortage of fuel across the country."},
        {"article_id": 4, "title_clean": "Uncategorized story", "content_clean": "Something unrelated happened."},
    ]
    written_rows = []
    cleared = {"called": False}

    import database as db
    db.get_articles_for_topic_modeling = lambda limit=None: fake_articles
    db.clear_article_topics = lambda: cleared.__setitem__("called", True)
    db.upsert_article_topics = lambda rows: written_rows.extend(rows)

    # mock the expensive fit_topic_model to avoid needing bertopic/sentence-transformers
    def fake_fit(texts, min_topic_size=15, nr_topics=None):
        fake_ids = [0, 1, 0, -1]  # article 1&3 -> topic 0, article 2 -> topic 1, article 4 -> outlier
        return model, fake_ids, [1.0] * len(texts)

    topics.fit_topic_model = fake_fit
    topics.save_topic_model = lambda m: None  # skip real file save

    with tempfile.TemporaryDirectory() as tmpdir:
        topics.MODEL_DIR = tmpdir
        result_model, topic_info = topics.process_all_topics()

    assert cleared["called"] is True, "clear_article_topics should be called before refit"
    assert len(written_rows) == 4
    assert written_rows[0]["topic_id"] == 0
    assert written_rows[0]["topic_name"] == "fuel_prices_shortage"
    assert written_rows[1]["topic_id"] == 1
    assert written_rows[1]["topic_name"] == "election_president_results"
    assert written_rows[3]["topic_id"] == -1
    assert written_rows[3]["topic_name"] == "outlier_uncategorized"
    print(f"   OK -> wrote {len(written_rows)} rows, cleared old topics first: {cleared['called']}\n")

    print("4. apply_topic_names + load_topic_names round trip")
    name_updates = {}
    db.update_topic_names = lambda mapping: name_updates.update(mapping)

    with tempfile.TemporaryDirectory() as tmpdir:
        topics.MODEL_DIR = tmpdir
        topics.apply_topic_names({0: "Fuel Crisis", 1: "Elections"})
        assert name_updates == {0: "Fuel Crisis", 1: "Elections"}

        loaded = topics.load_topic_names()
        assert loaded == {0: "Fuel Crisis", 1: "Elections"}
    print(f"   OK -> saved and reloaded: {loaded}\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()