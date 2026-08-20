"""
test_temporal_analysis.py

How to run:
    python tests/test_temporal_analysis.py

This module is pure pandas logic (no ML model dependency), so this test
verifies the actual aggregation MATH with mocked database functions,
not just the wiring — this is the most important test in the whole
pipeline to get right, since it's the core of the research.
"""

import sys, os
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

import pandas as pd
import database as db
import temporal_analysis as ta


def run():
    # --- Synthetic 3-week dataset ---
    fake_articles = [
        # Week 01: 3 articles, 2 negative, 1 positive, 1 fuel topic
        {"article_id": 1, "publication_week": "2026-W01", "sentiment": "negative", "sentiment_score": 0.8, "topic_name": "Fuel & Oil Prices"},
        {"article_id": 2, "publication_week": "2026-W01", "sentiment": "negative", "sentiment_score": 0.6, "topic_name": "Cricket & Sports"},
        {"article_id": 3, "publication_week": "2026-W01", "sentiment": "positive", "sentiment_score": 0.5, "topic_name": "Cricket & Sports"},
        # Week 02: 4 articles, fuel topic frequency doubles (2 fuel articles)
        {"article_id": 4, "publication_week": "2026-W02", "sentiment": "negative", "sentiment_score": 0.9, "topic_name": "Fuel & Oil Prices"},
        {"article_id": 5, "publication_week": "2026-W02", "sentiment": "negative", "sentiment_score": 0.7, "topic_name": "Fuel QR System & Rationing"},
        {"article_id": 6, "publication_week": "2026-W02", "sentiment": "neutral", "sentiment_score": 0.5, "topic_name": "Cricket & Sports"},
        {"article_id": 7, "publication_week": "2026-W02", "sentiment": "positive", "sentiment_score": 0.4, "topic_name": "Cricket & Sports"},
    ]
    fake_keywords = [
        {"article_id": 1, "publication_week": "2026-W01", "keyword": "fuel shortage"},
        {"article_id": 4, "publication_week": "2026-W02", "keyword": "fuel prices"},
        {"article_id": 5, "publication_week": "2026-W02", "keyword": "fuel shortage"},
    ]
    fake_entities = [
        {"article_id": 1, "publication_week": "2026-W01", "entity_type": "ORGANIZATION"},
        {"article_id": 1, "publication_week": "2026-W01", "entity_type": "PERSON"},
        {"article_id": 4, "publication_week": "2026-W02", "entity_type": "ORGANIZATION"},
        {"article_id": 5, "publication_week": "2026-W02", "entity_type": "ORGANIZATION"},
    ]

    db.get_article_level_dataset = lambda: fake_articles
    db.get_keyword_dataset = lambda: fake_keywords
    db.get_entity_dataset = lambda: fake_entities

    print("1. build_weekly_panel: article_count")
    weekly = ta.build_weekly_panel()
    assert len(weekly) == 2
    w1 = weekly[weekly["week"] == "2026-W01"].iloc[0]
    w2 = weekly[weekly["week"] == "2026-W02"].iloc[0]
    assert w1["article_count"] == 3
    assert w2["article_count"] == 4
    print(f"   OK -> W01 count={w1['article_count']}, W02 count={w2['article_count']}\n")

    print("2. build_weekly_panel: sentiment ratios")
    assert w1["negative_sentiment_ratio"] == 2 / 3
    assert w1["positive_sentiment_ratio"] == 1 / 3
    print(f"   OK -> W01 negative_ratio={w1['negative_sentiment_ratio']:.3f}\n")

    print("3. build_weekly_panel: signed average sentiment")
    # W01: -0.8, -0.6, +0.5 -> mean = -0.3
    expected_avg = (-0.8 - 0.6 + 0.5) / 3
    assert abs(w1["average_sentiment"] - expected_avg) < 1e-9
    print(f"   OK -> W01 average_sentiment={w1['average_sentiment']:.4f} (expected {expected_avg:.4f})\n")

    print("4. build_weekly_panel: fuel_topic_frequency (multi-topic-name group)")
    assert w1["fuel_topic_frequency"] == 1   # article 1 only
    assert w2["fuel_topic_frequency"] == 2   # articles 4 AND 5 (two different topic names, same group)
    print(f"   OK -> W01={w1['fuel_topic_frequency']}, W02={w2['fuel_topic_frequency']} (doubled as expected)\n")

    print("5. build_weekly_panel: keyword frequency (distinct article count, not row count)")
    assert w1["fuel_keyword_frequency"] == 1
    assert w2["fuel_keyword_frequency"] == 2
    print(f"   OK -> W01={w1['fuel_keyword_frequency']}, W02={w2['fuel_keyword_frequency']}\n")

    print("6. build_weekly_panel: entity-type frequency")
    assert w1["organization_entity_frequency"] == 1
    assert w1["person_entity_frequency"] == 1
    assert w2["organization_entity_frequency"] == 2
    print(f"   OK -> W01 ORG={w1['organization_entity_frequency']}, W02 ORG={w2['organization_entity_frequency']}\n")

    print("7. add_trend_features: article_count_change")
    weekly = ta.add_trend_features(weekly)
    w1t = weekly[weekly["week"] == "2026-W01"].iloc[0]
    w2t = weekly[weekly["week"] == "2026-W02"].iloc[0]
    assert pd.isna(w1t["article_count_change"])  # first week has no prior week
    expected_change = (4 - 3) / 3
    assert abs(w2t["article_count_change"] - expected_change) < 1e-9
    print(f"   OK -> W01=NaN (no prior week), W02={w2t['article_count_change']:.4f}\n")

    print("8. add_trend_features: moving averages are NaN with insufficient history")
    assert pd.isna(w2t["ma_4week"])  # only 2 weeks of data, need 4
    assert pd.isna(w2t["ma_8week"])
    print("   OK -> ma_4week/ma_8week correctly NaN with only 2 weeks of history\n")

    print("9. add_trend_features: topic_growth_rate reflects the doubling")
    # W01 total tracked-topic count = 1 (fuel), W02 = 2 (fuel) -> growth = 1.0 (100% increase)
    assert abs(w2t["topic_growth_rate"] - 1.0) < 1e-9
    print(f"   OK -> topic_growth_rate={w2t['topic_growth_rate']:.4f} (expected 1.0, i.e. +100%)\n")

    print("10. process_temporal_analysis: end-to-end with mocked upsert")
    written_rows = []
    db.upsert_temporal_features = lambda rows: written_rows.extend(rows)
    ta.process_temporal_analysis()
    assert len(written_rows) == 2
    assert written_rows[0]["week"] in ("2026-W01", "2026-W02")
    print(f"   OK -> wrote {len(written_rows)} weekly rows\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()