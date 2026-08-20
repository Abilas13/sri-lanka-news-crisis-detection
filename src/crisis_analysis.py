"""
crisis_analysis.py

Purpose
-------
Steps 14-16 and 18:
  14. Historical crisis dataset (crisis_events — already populated separately,
      see sql/cleanup_crisis_events.sql)
  15. Pre-crisis window labeling
  16. Early-signal analysis (rolling z-score anomaly detection)
  18. Avoid data leakage (all calculations here are strictly backward-looking)

Input
-----
The weekly temporal_features table (built by temporal_analysis.py) and
crisis_events.

Output
------
Updates temporal_features with: crisis_event, days_to_crisis,
label_pre_crisis, anomaly_score.

Methodology
-----------
- PRE_CRISIS_WINDOW_DAYS defines how many days before a crisis's start_date
  count as "pre-crisis" (label_pre_crisis=1). Weeks further away are
  "normal" (label_pre_crisis=0). Weeks DURING a crisis (between start_date
  and end_date) are explicitly excluded from either label (set to NULL) —
  per the brief's no-leakage rule, you're testing EARLY detection, not
  classifying the crisis period itself.
- anomaly_score: for each of a small set of key metrics, a rolling z-score
  is computed using a TRAILING window that excludes the current week
  (mean/std computed from weeks[t-window : t-1], never including week t
  itself or any future week). anomaly_score is the max absolute z-score
  across tracked metrics for that week — an interpretable "how unusual is
  this week" summary.
"""

import pandas as pd
import numpy as np
import utils

logger = utils.setup_logger(__name__)

PRE_CRISIS_WINDOW_DAYS = 30   # justified in the architecture doc: matches the
                                # brief's example window; adjust and re-justify
                                # in your write-up if you test other values

ANOMALY_ROLLING_WINDOW = 8    # weeks of trailing history used for the z-score baseline
ANOMALY_METRICS = [
    "article_count",
    "negative_sentiment_ratio",
    "fuel_topic_frequency",
    "iran_war_topic_frequency",
    "disaster_recovery_topic_frequency",
]


def get_crisis_events() -> pd.DataFrame:
    import database as db
    rows = db.get_crisis_events()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
        df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
    return df


def label_crisis_windows(weekly: pd.DataFrame, crisis_events: pd.DataFrame,
                          window_days: int = PRE_CRISIS_WINDOW_DAYS) -> pd.DataFrame:
    """
    Step 15: labels each week relative to the nearest crisis event.

    Adds:
        crisis_event      - event_id of the relevant crisis for this week
                              (active crisis takes priority, then pre-crisis
                              window, then just the nearest event for context)
        days_to_crisis     - negative = before the crisis's start_date,
                              positive = after
        label_pre_crisis   - 1 if within window_days BEFORE start_date,
                              0 if clearly outside any crisis window,
                              None if inside the crisis's active period
                              (start_date to end_date) — excluded from ML
                              training per the no-leakage / no-conflation rule.
    """
    weekly = weekly.copy()
    weekly["_week_date"] = weekly["week"].apply(utils.week_string_to_monday)

    crisis_event_col = []
    days_to_crisis_col = []
    label_col = []

    for week_date in weekly["_week_date"]:
        from datetime import timedelta
        week_end = week_date + timedelta(days=6)  # the week spans Monday through Sunday

        nearest_event_id, nearest_days = None, None
        active_event_id, active_days = None, None
        window_event_id, window_days_val = None, None
        is_active = False

        for _, event in crisis_events.iterrows():
            start = event["start_date"]
            end = event["end_date"] if pd.notna(event["end_date"]) else start
            days_from_start = (week_date - start).days

            # Track the nearest event overall, for descriptive days_to_crisis
            # even on "normal" weeks far from any crisis.
            if nearest_days is None or abs(days_from_start) < abs(nearest_days):
                nearest_event_id, nearest_days = event["event_id"], days_from_start

            # A week is "active" if its Mon-Sun span overlaps the crisis's
            # date range at all — NOT just if the week's Monday falls inside
            # it, since a crisis can start mid-week (e.g. a Saturday).
            if week_date <= end and week_end >= start:
                is_active = True
                active_event_id, active_days = event["event_id"], days_from_start

            if -window_days <= days_from_start < 0:
                if window_days_val is None or abs(days_from_start) < abs(window_days_val):
                    window_event_id, window_days_val = event["event_id"], days_from_start

        if is_active:
            crisis_event_col.append(active_event_id)
            days_to_crisis_col.append(active_days)
            label_col.append(None)   # excluded — inside the crisis itself
        elif window_event_id is not None:
            crisis_event_col.append(window_event_id)
            days_to_crisis_col.append(window_days_val)
            label_col.append(1)      # pre-crisis
        else:
            crisis_event_col.append(nearest_event_id)
            days_to_crisis_col.append(nearest_days)
            label_col.append(0)      # normal

    weekly["crisis_event"] = crisis_event_col
    weekly["days_to_crisis"] = days_to_crisis_col
    weekly["label_pre_crisis"] = label_col
    weekly = weekly.drop(columns=["_week_date"])
    return weekly


def compute_anomaly_scores(weekly: pd.DataFrame, metrics: list = None,
                            window: int = ANOMALY_ROLLING_WINDOW) -> pd.DataFrame:
    """
    Step 16: rolling z-score anomaly detection.

    For each metric, computes a trailing rolling mean/std using ONLY prior
    weeks (shift(1) before rolling — the current week is never included in
    its own baseline, preventing leakage). z = (value - rolling_mean) / rolling_std.

    anomaly_score per week = the max absolute z-score across all tracked
    metrics — a single interpretable summary of "how unusual is this week."
    Weeks with insufficient history (< window prior weeks) get NaN, not a
    misleading low-confidence score.
    """
    if metrics is None:
        metrics = ANOMALY_METRICS

    weekly = weekly.sort_values("week").reset_index(drop=True)
    z_score_cols = []

    for metric in metrics:
        if metric not in weekly.columns:
            logger.warning(f"Metric '{metric}' not found in weekly panel, skipping.")
            continue

        shifted = weekly[metric].shift(1)  # exclude current week from its own baseline
        rolling_mean = shifted.rolling(window=window, min_periods=window).mean()
        rolling_std = shifted.rolling(window=window, min_periods=window).std()

        z_col = f"_z_{metric}"
        # avoid divide-by-zero: where std is 0 (or NaN), z-score is undefined -> NaN
        weekly[z_col] = np.where(
            (rolling_std > 0) & rolling_std.notna(),
            (weekly[metric] - rolling_mean) / rolling_std,
            np.nan,
        )
        z_score_cols.append(z_col)

    if z_score_cols:
        weekly["anomaly_score"] = weekly[z_score_cols].abs().max(axis=1)
        weekly = weekly.drop(columns=z_score_cols)
    else:
        weekly["anomaly_score"] = None

    return weekly


def process_crisis_analysis():
    """
    Full Step 14-16/18 pipeline: fetches the current weekly panel and crisis
    events, labels pre-crisis windows, computes anomaly scores, writes back
    to temporal_features.
    """
    import database as db

    logger.info("Fetching weekly panel and crisis events...")
    weekly = pd.DataFrame(db.get_temporal_features())
    crisis_events = get_crisis_events()

    if weekly.empty:
        print("No temporal_features data found — run temporal_analysis.py first.")
        return weekly
    if crisis_events.empty:
        print("No crisis_events found — run sql/cleanup_crisis_events.sql first.")
        return weekly

    logger.info(f"Labeling pre-crisis windows ({PRE_CRISIS_WINDOW_DAYS}-day window)...")
    weekly = label_crisis_windows(weekly, crisis_events)

    logger.info("Computing rolling z-score anomaly scores...")
    weekly = compute_anomaly_scores(weekly)

    rows = weekly.to_dict(orient="records")
    for row in rows:
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None

    db.upsert_temporal_features(rows)

    n_pre_crisis = sum(1 for r in rows if r["label_pre_crisis"] == 1)
    n_normal = sum(1 for r in rows if r["label_pre_crisis"] == 0)
    n_excluded = sum(1 for r in rows if r["label_pre_crisis"] is None)
    print(f"Done. {len(rows)} weeks updated: "
          f"{n_pre_crisis} pre-crisis, {n_normal} normal, {n_excluded} excluded (active crisis periods).")
    return weekly


if __name__ == "__main__":
    process_crisis_analysis()