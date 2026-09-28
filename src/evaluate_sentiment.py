"""
evaluate_sentiment.py

Purpose
-------
Compares the model's sentiment predictions against your hand-labeled
true_sentiment column, producing the evaluation metrics your research
write-up needs (Step 7 requirement).

"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def run(csv_path: str = "../data/processed/sentiment_eval_sample.csv"):
    df = pd.read_csv(csv_path)

    df["true_sentiment"] = df["true_sentiment"].astype(str).str.strip().str.lower()

    if "sentiment" not in df.columns:
        raise ValueError(
            "CSV doesn't have a 'sentiment' column (the model's prediction). "
            "If get_sentiment_eval_sample() only pulled from processed_news, "
            "you need to join in article_sentiment.sentiment first — see the "
            "join snippet in the message below this script."
        )

    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    valid_labels = {"positive", "neutral", "negative"}
    bad_rows = df[~df["true_sentiment"].isin(valid_labels)]
    if len(bad_rows) > 0:
        print(f"WARNING: {len(bad_rows)} rows have an unrecognized true_sentiment value "
              f"(expected positive/neutral/negative). These are excluded from scoring:")
        print(bad_rows[["article_id", "true_sentiment"]])
        df = df[df["true_sentiment"].isin(valid_labels)]

    y_true = df["true_sentiment"]
    y_pred = df["sentiment"]

    print(f"\nEvaluated on {len(df)} hand-labeled articles\n")

    acc = accuracy_score(y_true, y_pred)
    print(f"Overall accuracy: {acc:.3f}\n")

    labels = ["negative", "neutral", "positive"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    print("Per-class metrics:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    for i, label in enumerate(labels):
        print(f"{label:<10} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_df)

    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm_df}


if __name__ == "__main__":
    run()