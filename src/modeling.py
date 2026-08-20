"""
modeling.py

Purpose
-------
Steps 17, 19, 20 — the final research output:
  17. ML classification (Logistic Regression + Random Forest)
  19. Evaluation (precision/recall/F1/ROC-AUC/PR-AUC/confusion matrix)
  20. Lead-time analysis — the headline result: how many days before each
      crisis did the model first flag an elevated pre-crisis probability?

Step 18 (avoid data leakage) is handled by construction: features are all
backward-looking (built in temporal_analysis.py / crisis_analysis.py), and
validation uses leave-one-crisis-out — the model NEVER sees the crisis
it's being tested on, in either its pre-crisis or normal weeks.

Small-sample caveat
---------------------
With only 2 usable crisis events and roughly 3-8 pre-crisis-labeled weeks
each, this is a small dataset for ML in the conventional sense. Results
here should be reported as exploratory/indicative, not a high-confidence
production classifier — say so explicitly in your write-up. This is also
why leave-one-crisis-out (not a large held-out test set) is the right
validation choice, per the original methodology.

Feature set
-----------
A deliberately compact set (not all 29+ columns in temporal_features) —
mirrors the brief's Step 17 example list, avoiding overfitting risk on
~80 rows:
    article_count, negative_sentiment_ratio, positive_sentiment_ratio,
    average_sentiment, fuel_topic_frequency, iran_war_topic_frequency,
    disaster_recovery_topic_frequency, article_count_change,
    sentiment_change, topic_growth_rate, keyword_growth_rate,
    entity_growth_rate, ma_4week, ma_8week, anomaly_score
"""

import pandas as pd
import numpy as np
import utils

logger = utils.setup_logger(__name__)

FEATURE_COLUMNS = [
    "article_count", "negative_sentiment_ratio", "positive_sentiment_ratio",
    "average_sentiment", "fuel_topic_frequency", "iran_war_topic_frequency",
    "disaster_recovery_topic_frequency", "article_count_change",
    "sentiment_change", "topic_growth_rate", "keyword_growth_rate",
    "entity_growth_rate", "ma_4week", "ma_8week", "anomaly_score",
]

TARGET_COLUMN = "label_pre_crisis"


def get_ml_dataset() -> pd.DataFrame:
    """
    Fetches temporal_features, excludes active-crisis weeks (label_pre_crisis
    IS NULL — per the no-conflation rule), and drops rows with missing
    values in any feature column (early weeks lacking enough history for
    ma_4week/ma_8week/anomaly_score, or the first week overall with no
    prior week for article_count_change/sentiment_change).
    """
    import database as db

    df = pd.DataFrame(db.get_temporal_features())
    if df.empty:
        return df

    df = df[df[TARGET_COLUMN].notna()].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    after = len(df)
    if before != after:
        logger.info(f"Dropped {before - after} rows with missing feature values "
                     f"(insufficient rolling-window history) — {after} rows remain.")

    return df.reset_index(drop=True)


def _compute_metrics(y_true, y_pred, y_proba) -> dict:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC-AUC/PR-AUC need both classes present in y_true to be meaningful
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metrics


def train_and_evaluate_loco(df: pd.DataFrame = None) -> dict:
    """
    Step 17 + 19: trains Logistic Regression and Random Forest with
    leave-one-crisis-out validation (fold = crisis_event column — each
    week already belongs to its nearest crisis from crisis_analysis.py).

    Returns a dict: {model_name: {fold_crisis_event: metrics_dict, ...}, ...}
    plus an 'overall' key averaging metrics across folds per model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    if df is None:
        df = get_ml_dataset()
    if df.empty:
        raise ValueError("No data available for modeling — run temporal_analysis.py "
                          "and crisis_analysis.py first.")

    folds = df["crisis_event"].dropna().unique()
    if len(folds) < 2:
        raise ValueError(f"Need at least 2 crisis events for leave-one-crisis-out "
                          f"validation, found {len(folds)}.")

    models = {
        "logistic_regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": lambda: RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42),
    }

    results = {name: {} for name in models}
    all_predictions = []  # collected for lead-time analysis later

    for held_out_fold in folds:
        train_df = df[df["crisis_event"] != held_out_fold]
        test_df = df[df["crisis_event"] == held_out_fold]

        if train_df[TARGET_COLUMN].nunique() < 2:
            logger.warning(f"Skipping fold {held_out_fold}: training data has only one class.")
            continue

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

        for model_name, model_factory in models.items():
            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = _compute_metrics(y_test.values, y_pred, y_proba)
            results[model_name][held_out_fold] = metrics

            for week, true_label, proba in zip(test_df["week"], y_test, y_proba):
                all_predictions.append({
                    "week": week, "crisis_event": held_out_fold,
                    "model": model_name, "true_label": true_label, "predicted_proba": proba,
                })

    # Average metrics across folds, per model
    for model_name in models:
        fold_metrics = results[model_name]
        if not fold_metrics:
            continue
        avg = {}
        for key in ["precision", "recall", "f1"]:
            values = [m[key] for m in fold_metrics.values()]
            avg[key] = sum(values) / len(values)
        roc_values = [m["roc_auc"] for m in fold_metrics.values() if m["roc_auc"] is not None]
        pr_values = [m["pr_auc"] for m in fold_metrics.values() if m["pr_auc"] is not None]
        avg["roc_auc"] = sum(roc_values) / len(roc_values) if roc_values else None
        avg["pr_auc"] = sum(pr_values) / len(pr_values) if pr_values else None
        results[model_name]["overall"] = avg

    results["_predictions"] = pd.DataFrame(all_predictions)
    return results


def compute_lead_time(predictions_df: pd.DataFrame, crisis_events: pd.DataFrame,
                       threshold: float = 0.5, model: str = "random_forest") -> pd.DataFrame:
    """
    Step 20: for each crisis event, finds the first week (walking forward
    in chronological order through the pre-crisis window) where the
    predicted probability crosses `threshold`, and computes how many days
    before the crisis's start_date that detection occurred.

    Returns a DataFrame: event_name, first_detection_week, lead_time_days.
    A crisis with no week crossing threshold gets lead_time_days = None
    (a real, honest result — not every crisis may show early signal).
    """
    rows = []
    model_preds = predictions_df[predictions_df["model"] == model].copy()
    model_preds["_week_date"] = model_preds["week"].apply(utils.week_string_to_monday)
    model_preds = model_preds.sort_values("_week_date")

    for _, event in crisis_events.iterrows():
        event_id = event["event_id"]
        event_preds = model_preds[
            (model_preds["crisis_event"] == event_id) & (model_preds["true_label"] == 1)
        ]
        detected = event_preds[event_preds["predicted_proba"] >= threshold]

        if detected.empty:
            rows.append({
                "event_name": event["event_name"], "first_detection_week": None,
                "lead_time_days": None,
            })
        else:
            first_week = detected.iloc[0]
            lead_days = (event["start_date"] - first_week["_week_date"]).days
            rows.append({
                "event_name": event["event_name"],
                "first_detection_week": first_week["week"],
                "lead_time_days": lead_days,
            })

    return pd.DataFrame(rows)


def run_full_evaluation(threshold: float = 0.5):
    """
    Full Step 17/19/20 pipeline: trains both models with LOCO validation,
    prints evaluation metrics, computes and prints lead-time results.
    """
    import crisis_analysis as ca

    df = get_ml_dataset()
    print(f"Dataset: {len(df)} weeks, {df[TARGET_COLUMN].sum()} pre-crisis, "
          f"{len(df) - df[TARGET_COLUMN].sum()} normal.\n")

    results = train_and_evaluate_loco(df)

    for model_name in ["logistic_regression", "random_forest"]:
        print(f"=== {model_name} ===")
        for fold, metrics in results[model_name].items():
            print(f"  Fold: {fold}")
            for key in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
                val = metrics[key] if fold != "overall" else metrics.get(key)
                if val is not None:
                    print(f"    {key}: {val:.3f}")
            if fold != "overall":
                print(f"    confusion_matrix: {metrics['confusion_matrix']}")
        print()

    crisis_events = ca.get_crisis_events()
    print("=== Lead-Time Analysis (Random Forest, threshold=0.5) ===")
    lead_time_df = compute_lead_time(results["_predictions"], crisis_events, threshold=threshold)
    print(lead_time_df.to_string(index=False))

    return results, lead_time_df


if __name__ == "__main__":
    run_full_evaluation()