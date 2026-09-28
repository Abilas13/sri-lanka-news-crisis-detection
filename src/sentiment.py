"""
sentiment.py

Purpose
-------
runs a pretrained transformer sentiment model over content_clean
and writes results to article_sentiment. No fine-tuning — pretrained model
only, per the project's time-limited scope.


"""

import utils

logger = utils.setup_logger(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# The model's raw output labels can come through as either the human-readable
# form or LABEL_0/1/2 depending on transformers version — map both to be safe.
_LABEL_MAP = {
    "negative": "negative", "LABEL_0": "negative",
    "neutral": "neutral", "LABEL_1": "neutral",
    "positive": "positive", "LABEL_2": "positive",
}

_pipeline = None  # lazy-loaded singleton — model load is expensive, do it once


def _get_pipeline():
    """
    Lazily loads the HuggingFace sentiment-analysis pipeline. Loading the
    model takes real time (downloading + initializing) — only happens once
    per process, on first call.
    """
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading sentiment model {MODEL_NAME} (device={'GPU' if device == 0 else 'CPU'})...")
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            truncation=True,
            max_length=512,
            device=device,
        )
        logger.info("Sentiment model loaded.")
    return _pipeline


def predict_sentiment_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Runs sentiment prediction on a list of texts.
    Returns a list of {'sentiment': str, 'sentiment_score': float}, same
    order as input. Empty strings get a safe default (neutral, score 0.0)
    without calling the model.
    """
    pipe = _get_pipeline()

    # Separate empty texts (model can't handle them meaningfully) from real ones
    results = [None] * len(texts)
    real_indices = []
    real_texts = []
    for i, t in enumerate(texts):
        if not t or not t.strip():
            results[i] = {"sentiment": "neutral", "sentiment_score": 0.0}
        else:
            real_indices.append(i)
            real_texts.append(t)

    if real_texts:
        raw_outputs = pipe(real_texts, batch_size=batch_size, truncation=True)
        for idx, output in zip(real_indices, raw_outputs):
            label = _LABEL_MAP.get(output["label"], output["label"].lower())
            results[idx] = {"sentiment": label, "sentiment_score": float(output["score"])}

    return results


def process_all_sentiment(batch_size: int = 32, limit: int = None):
    """
    Fetches all COMPLETED articles that don't have a sentiment row yet,
    runs prediction in batches, upserts into article_sentiment.
    Prints progress every batch, same pattern as pipeline.process_existing_articles.
    """
    import database as db

    articles = db.get_articles_needing_sentiment(limit=limit)
    total = len(articles)
    logger.info(f"process_all_sentiment: {total} articles to process")

    if total == 0:
        print("Nothing to process — all articles already have sentiment.")
        return

    processed = 0
    for start in range(0, total, batch_size):
        chunk = articles[start:start + batch_size]
        ids = [row["article_id"] for row in chunk]
        texts = [row["content_clean"] for row in chunk]

        predictions = predict_sentiment_batch(texts, batch_size=batch_size)

        rows = [
            {
                "article_id": aid,
                "sentiment": pred["sentiment"],
                "sentiment_score": pred["sentiment_score"],
                "sentiment_model": MODEL_NAME,
            }
            for aid, pred in zip(ids, predictions)
        ]
        db.upsert_article_sentiment(rows)
        processed += len(rows)
        print(f"  Progress: {min(start + batch_size, total)}/{total}")

    print(f"\nDone. Sentiment computed for {processed} articles.")


def get_sentiment_eval_sample(n: int = 120, random_state: int = 42):
    """
    Returns a random sample of N processed articles (article_id, title_clean,
    content_clean) for manual labeling — the evaluation step your research
    write-up needs. Not written to the DB; just returned for you to export
    (e.g. to a CSV) and label by hand.
    """
    import database as db
    import pandas as pd
    from sqlalchemy import text

    with db.get_connection() as conn:
        df = pd.read_sql(
            text("SELECT article_id, title_clean, content_clean FROM processed_news "
                 "WHERE processing_status = 'COMPLETED' ORDER BY RAND(:seed) LIMIT :n"),
            conn, params={"seed": random_state, "n": n}
        )
    return df


if __name__ == "__main__":
    process_all_sentiment()