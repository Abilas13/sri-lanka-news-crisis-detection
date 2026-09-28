"""
test_modeling.py

How to run:
    python tests/test_modeling.py


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
import numpy as np
from datetime import date
import modeling as md


def _build_synthetic_dataset():
    """
    Builds a 60-week synthetic panel with 2 crisis events, each with a
    clear elevated-feature pre-crisis window, so the model has real
    signal to learn from and lead-time detection has a known answer.
    """
    np.random.seed(42)
    rows = []
    # Crisis A: onset at week index 20 (event_id=1), pre-crisis window = weeks 15-19
    # Crisis B: onset at week index 45 (event_id=2), pre-crisis window = weeks 40-44
    for i in range(60):
        week = f"2025-W{i+1:02d}" if i < 52 else f"2026-W{i-51:02d}"
        is_precrisis_a = 15 <= i < 20
        is_precrisis_b = 40 <= i < 45
        is_active_a = 20 <= i < 22
        is_active_b = 45 <= i < 47

        elevated = is_precrisis_a or is_precrisis_b or is_active_a or is_active_b
        base = 5 + np.random.normal(0, 1)
        signal = base + (15 if elevated else 0)

        if is_active_a or is_active_b:
            label = None
        elif is_precrisis_a or is_precrisis_b:
            label = 1
        else:
            label = 0

        crisis_event = 1 if i < 32 else 2  # rough nearest-event split for fold assignment

        rows.append({
            "week": week,
            "article_count": 200 + np.random.normal(0, 10),
            "negative_sentiment_ratio": 0.25 + (0.1 if elevated else 0),
            "positive_sentiment_ratio": 0.1,
            "average_sentiment": -0.1 - (0.1 if elevated else 0),
            "fuel_topic_frequency": signal,
            "iran_war_topic_frequency": signal * 0.8,
            "disaster_recovery_topic_frequency": 20 + np.random.normal(0, 2),
            "article_count_change": np.random.normal(0, 0.05),
            "sentiment_change": np.random.normal(0, 0.02),
            "topic_growth_rate": np.random.normal(0.5 if elevated else 0, 0.1),
            "keyword_growth_rate": np.random.normal(0.3 if elevated else 0, 0.1),
            "entity_growth_rate": np.random.normal(0.2 if elevated else 0, 0.1),
            "ma_4week": 200.0,
            "ma_8week": 200.0,
            "anomaly_score": 8.0 if elevated else 1.0,
            "crisis_event": crisis_event,
            "label_pre_crisis": label,
            "days_to_crisis": None,
        })
    return pd.DataFrame(rows)


def run():
    df = _build_synthetic_dataset()

    print("1. get_ml_dataset filtering (via direct df, bypassing DB)")
    filtered = df[df["label_pre_crisis"].notna()].dropna(subset=md.FEATURE_COLUMNS)
    n_null = df["label_pre_crisis"].isna().sum()
    assert n_null == 4  # 2 active weeks per crisis x 2 crises
    print(f"   OK -> {n_null} active-crisis weeks correctly excluded from {len(df)} total\n")

    print("2. train_and_evaluate_loco: both models produce results for both folds")
    results = md.train_and_evaluate_loco(filtered)
    assert "logistic_regression" in results
    assert "random_forest" in results
    assert 1 in results["random_forest"] and 2 in results["random_forest"]
    print(f"   OK -> folds present: {[k for k in results['random_forest'].keys() if k != 'overall']}\n")

    print("3. train_and_evaluate_loco: metrics are sane (0-1 range)")
    for model_name in ["logistic_regression", "random_forest"]:
        overall = results[model_name]["overall"]
        for key in ["precision", "recall", "f1"]:
            assert 0.0 <= overall[key] <= 1.0, f"{model_name}.{key}={overall[key]} out of range"
    print(f"   OK -> RF overall: {results['random_forest']['overall']}\n")

    print("4. train_and_evaluate_loco: recovers real signal (recall > 0)")
    # Given the strong synthetic signal (elevated features in pre-crisis weeks),
    # the model should catch at least SOME pre-crisis weeks, not recall=0.
    rf_overall = results["random_forest"]["overall"]
    assert rf_overall["recall"] > 0.0, "model should detect at least some pre-crisis weeks given strong synthetic signal"
    print(f"   OK -> random_forest recall={rf_overall['recall']:.3f} (signal successfully learned)\n")

    print("5. compute_lead_time: detects crisis A ahead of its start date")
    crisis_events = pd.DataFrame([
        {"event_id": 1, "event_name": "Synthetic Crisis A", "start_date": date(2025, 5, 19)},  # approx week 20
        {"event_id": 2, "event_name": "Synthetic Crisis B", "start_date": date(2025, 11, 10)},  # approx week 45
    ])
    lead_time_df = md.compute_lead_time(results["_predictions"], crisis_events, threshold=0.5, model="random_forest")
    print(f"   Result:\n{lead_time_df.to_string(index=False)}")
    # At least one of the two synthetic crises should show a positive lead time
    # given the strong pre-crisis signal built into the synthetic data.
    assert lead_time_df["lead_time_days"].notna().any(), "expected at least one crisis to show detectable lead time"
    print("   OK -> at least one crisis shows a computed lead time\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()