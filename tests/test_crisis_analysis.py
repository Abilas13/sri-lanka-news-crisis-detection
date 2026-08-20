"""
test_crisis_analysis.py

How to run:
    python tests/test_crisis_analysis.py

Pure pandas/math logic — tests use synthetic data shaped like the real
fuel-crisis pattern actually observed in the dataset (baseline ~0-5,
jumping to 60+ at crisis onset), plus edge cases around window boundaries.
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
import utils
import crisis_analysis as ca


def run():
    print("1. utils.week_string_to_monday: known conversion")
    # 2026-W09 (Feb 28 2026 crisis) — verify it lands in the right calendar week
    monday = utils.week_string_to_monday("2026-W09")
    from datetime import date
    assert monday.isocalendar()[:2] == (2026, 9)
    print(f"   OK -> 2026-W09 Monday = {monday}\n")

    print("2. label_crisis_windows: pre-crisis window correctly flagged")
    crisis_events = pd.DataFrame([
        {"event_id": 1, "event_name": "Test Crisis", "start_date": date(2026, 2, 28),
         "end_date": date(2026, 3, 15)},
    ])
    weekly = pd.DataFrame({
        "week": ["2026-W05", "2026-W07", "2026-W08", "2026-W09", "2026-W10", "2026-W15"],
    })
    labeled = ca.label_crisis_windows(weekly, crisis_events, window_days=30)

    row_w05 = labeled[labeled["week"] == "2026-W05"].iloc[0]  # 33 days before -> OUTSIDE 30-day window -> normal
    row_w07 = labeled[labeled["week"] == "2026-W07"].iloc[0]  # 19 days before -> pre-crisis
    row_w15 = labeled[labeled["week"] == "2026-W15"].iloc[0]  # far after -> normal
    row_w09 = labeled[labeled["week"] == "2026-W09"].iloc[0]  # onset week -> active/excluded

    assert row_w07["label_pre_crisis"] == 1, f"W07 should be pre-crisis, got {row_w07['label_pre_crisis']}"
    assert row_w05["label_pre_crisis"] == 0, f"W05 (33 days out, outside window) should be normal, got {row_w05['label_pre_crisis']}"
    assert pd.isna(row_w09["label_pre_crisis"]), f"W09 (active) should be excluded (NaN/None), got {row_w09['label_pre_crisis']}"
    assert row_w15["label_pre_crisis"] == 0, f"W15 should be normal, got {row_w15['label_pre_crisis']}"
    print(f"   OK -> W05={row_w05['label_pre_crisis']} (33d out, correctly normal), "
          f"W07={row_w07['label_pre_crisis']} (19d out, correctly pre-crisis), "
          f"W09={row_w09['label_pre_crisis']} (excluded/active), "
          f"W15={row_w15['label_pre_crisis']} (normal)\n")

    print("3. label_crisis_windows: days_to_crisis sign convention")
    assert row_w07["days_to_crisis"] < 0, "days before crisis should be negative"
    print(f"   OK -> W07 days_to_crisis={row_w07['days_to_crisis']} (negative = before)\n")

    print("4. compute_anomaly_scores: realistic fuel-crisis-shaped data")
    # Mimics the real pattern observed: baseline ~0-5, jump to 60+ at week 9 of a 14-week series
    weeks = [f"2026-W{i:02d}" for i in range(1, 15)]
    fuel_values = [2, 0, 3, 1, 0, 3, 0, 5, 13, 14, 21, 61, 31, 13]
    weekly2 = pd.DataFrame({
        "week": weeks,
        "article_count": [200] * 14,
        "negative_sentiment_ratio": [0.25] * 14,
        "fuel_topic_frequency": fuel_values,
        "iran_war_topic_frequency": [1] * 14,
        "disaster_recovery_topic_frequency": [20] * 14,
    })
    result = ca.compute_anomaly_scores(weekly2, window=8)

    early_weeks = result[result["week"].isin(["2026-W02", "2026-W03"])]
    assert early_weeks["anomaly_score"].isna().all(), "weeks with <8 prior weeks should have NaN anomaly_score"
    print("   OK -> early weeks (insufficient history) correctly NaN\n")

    print("5. compute_anomaly_scores: crisis spike weeks show high anomaly_score")
    spike_week = result[result["week"] == "2026-W12"].iloc[0]  # fuel=61, way above baseline
    baseline_week = result[result["week"] == "2026-W09"].iloc[0]  # fuel=13, first sign of rise
    assert spike_week["anomaly_score"] > 2.0, f"spike week should have |z|>2, got {spike_week['anomaly_score']}"
    print(f"   OK -> W12 (fuel=61) anomaly_score={spike_week['anomaly_score']:.2f} (correctly flagged as anomalous)\n")

    print("6. compute_anomaly_scores: no leakage — current week excluded from own baseline")
    # Manually verify: W12's baseline should be mean/std of W04-W11 (8 weeks prior), NOT including W12 itself
    prior_8 = fuel_values[3:11]  # W04 through W11 (0-indexed: weeks[3:11])
    import numpy as np
    expected_mean = np.mean(prior_8)
    expected_std = np.std(prior_8, ddof=1)
    expected_z = abs((61 - expected_mean) / expected_std)
    assert abs(spike_week["anomaly_score"] - expected_z) < 0.01, \
        f"anomaly_score mismatch: got {spike_week['anomaly_score']}, expected ~{expected_z}"
    print(f"   OK -> manually verified z-score matches: {spike_week['anomaly_score']:.3f} ≈ {expected_z:.3f}\n")

    print("7. process_crisis_analysis: mocked DB end-to-end")
    import database as db
    db.get_temporal_features = lambda: weekly2.to_dict(orient="records")
    db.get_crisis_events = lambda: [
        {"event_id": 1, "event_name": "Test Fuel Crisis", "crisis_type": "Economic",
         "start_date": date(2026, 2, 23), "end_date": date(2026, 3, 20), "description": "test"}
    ]
    written_rows = []
    db.upsert_temporal_features = lambda rows: written_rows.extend(rows)

    ca.process_crisis_analysis()
    assert len(written_rows) == 14
    assert any(r["label_pre_crisis"] == 1 for r in written_rows)
    assert any(r["label_pre_crisis"] is None or pd.isna(r["label_pre_crisis"]) for r in written_rows)
    print(f"   OK -> wrote {len(written_rows)} weekly rows with labels and anomaly scores\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()