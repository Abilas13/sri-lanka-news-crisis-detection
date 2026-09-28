"""
topics.py

Purpose
-------
identifies major topics across the corpus using BERTopic +
sentence-transformer embeddings, and tracks how they change over time
(via article_topics -> temporal_analysis.py later).

"""

import os
import json
import utils

logger = utils.setup_logger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "topics")

_embedding_model = None


def _get_embedding_model():
    """Lazily loads the sentence-transformer embedding model (shared with keywords.py)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embedding_model


def _auto_topic_name(model, topic_id: int, n_words: int = 3) -> str:
    """Generates a default name like 'fuel_shortage_prices' from a topic's top keywords."""
    if topic_id == -1:
        return "outlier_uncategorized"
    words = model.get_topic(topic_id)
    if not words:
        return f"topic_{topic_id}"
    top_words = [w for w, _score in words[:n_words]]
    return "_".join(top_words)


def fit_topic_model(
    texts: list[str],
    min_topic_size: int = 15,
    nr_topics: int = None,
    reduce_outliers: bool = True
):
    """
    Fits a fresh BERTopic model on the given texts.

    Pipeline:
        1. Generate sentence-transformer embeddings
        2. Fit BERTopic
        3. Optionally reduce the number of topics
        4. Reduce -1 outliers using embedding similarity

    Returns:
        (model, topic_ids, probabilities)
    """
    from bertopic import BERTopic

    embedder = _get_embedding_model()

    # ---------------------------------------------------------
    # 1. Generate embeddings
    # ---------------------------------------------------------
    logger.info(f"Encoding {len(texts)} documents...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True
    )

    # ---------------------------------------------------------
    # 2. Fit BERTopic
    # ---------------------------------------------------------
    logger.info("Fitting BERTopic model...")

    model = BERTopic(
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True
    )

    topic_ids, _probs = model.fit_transform(
        texts,
        embeddings
    )

    n_initial = len(set(topic_ids)) - (
        1 if -1 in topic_ids else 0
    )

    initial_outliers = sum(
        1 for topic in topic_ids if topic == -1
    )

    logger.info(
        f"Initial fit found {n_initial} topics "
        f"(+ outliers if any)."
    )

    logger.info(
        f"Initial outliers: "
        f"{initial_outliers}/{len(topic_ids)}"
    )

    # ---------------------------------------------------------
    # 3. Reduce number of topics
    # ---------------------------------------------------------
    if nr_topics and n_initial > nr_topics:

        logger.info(
            f"Reducing {n_initial} topics down to ~{nr_topics} "
            f"(reusing embeddings, no re-encode)..."
        )

        model.reduce_topics(
            texts,
            nr_topics=nr_topics
        )

        topic_ids = model.topics_

        n_final = len(set(topic_ids)) - (
            1 if -1 in topic_ids else 0
        )

        logger.info(
            f"Reduced to {n_final} topics."
        )

    # ---------------------------------------------------------
    # 4. Reduce outliers
    # ---------------------------------------------------------
    if reduce_outliers:

        before_outliers = sum(
            1 for topic in topic_ids if topic == -1
        )

        logger.info(
            f"Reducing {before_outliers} outliers "
            f"using embedding similarity..."
        )

        if before_outliers > 0:

            new_topics = model.reduce_outliers(
                texts,
                topic_ids,
                strategy="embeddings",
                embeddings=embeddings
            )

            # Update model's topic assignments
            model.update_topics(
                texts,
                topics=new_topics
            )

            topic_ids = new_topics

        after_outliers = sum(
            1 for topic in topic_ids if topic == -1
        )

        reassigned = before_outliers - after_outliers

        logger.info(
            f"Outlier reduction complete: "
            f"{before_outliers} -> {after_outliers}"
        )

        logger.info(
            f"Reassigned {reassigned} outlier articles."
        )

    # ---------------------------------------------------------
    # 5. Placeholder probabilities
    # ---------------------------------------------------------
    probabilities = [1.0] * len(topic_ids)

    final_topics = len(set(topic_ids)) - (
        1 if -1 in topic_ids else 0
    )

    final_outliers = sum(
        1 for topic in topic_ids if topic == -1
    )

    logger.info(
        f"Fitting complete. Found {final_topics} topics "
        f"(+ {final_outliers} outliers)."
    )

    return model, topic_ids, probabilities

def save_topic_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "bertopic_model")
    model.save(path, serialization="safetensors", save_ctfidf=True)
    logger.info(f"Model saved to {path}")


def load_topic_model():
    from bertopic import BERTopic
    path = os.path.join(MODEL_DIR, "bertopic_model")
    return BERTopic.load(path)


def save_topic_info_csv(model):
    """Saves a human-reviewable summary of every topic and its top keywords."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    info = model.get_topic_info()  # DataFrame: Topic, Count, Name, Representation...
    path = os.path.join(MODEL_DIR, "topic_info.csv")
    info.to_csv(path, index=False)
    logger.info(f"Topic info saved to {path} — review this to decide on manual names.")
    return info


def process_all_topics(min_topic_size: int = 15, nr_topics: int = None, limit: int = None):
    """
    Full Step 8 pipeline: fetches all COMPLETED articles, fits BERTopic on
    the whole corpus, writes article_topics, saves the model + topic_info.csv.

    nr_topics: consolidate down to roughly this many topics (recommended for
    corpora of this size — an unconstrained fit on 16k+ articles easily
    produces 100+ narrow topics, too many for weekly frequency tracking).
    """
    import database as db

    articles = db.get_articles_for_topic_modeling(limit=limit)
    total = len(articles)
    logger.info(f"process_all_topics: fitting on {total} articles")

    if total == 0:
        print("No articles available for topic modeling.")
        return

    ids = [a["article_id"] for a in articles]
    texts = [
        f"{(a['title_clean'] or '').strip()} {(a['content_clean'] or '').strip()}".strip()
        for a in articles
    ]

    model, topic_ids, probabilities = fit_topic_model(texts, min_topic_size=min_topic_size, nr_topics=nr_topics)

    # Build auto-generated names for every discovered topic, once
    unique_topics = set(topic_ids)
    auto_names = {tid: _auto_topic_name(model, tid) for tid in unique_topics}

    rows = [
        {
            "article_id": aid,
            "topic_id": tid,
            "topic_name": auto_names[tid],
            "topic_probability": prob,
        }
        for aid, tid, prob in zip(ids, topic_ids, probabilities)
    ]

    db.clear_article_topics()   # refit replaces all previous topic assignments
    db.upsert_article_topics(rows)

    save_topic_model(model)
    topic_info = save_topic_info_csv(model)

    print(f"\nDone. {total} articles assigned to {len(unique_topics)} topics (including outliers if present).")
    print("\nTop topics by article count:")
    print(topic_info.head(15).to_string(index=False))
    print(f"\nFull topic list saved to {os.path.join(MODEL_DIR, 'topic_info.csv')} — "
          f"review it, then call apply_topic_names({{topic_id: 'Your Name', ...}}) to rename.")

    return model, topic_info


def apply_topic_names(name_mapping: dict[int, str]):
    """
    Overrides auto-generated topic names with your manually chosen ones.
    Also saves the mapping to models/topics/topic_names.json for
    reproducibility (so re-running doesn't lose your naming work — reload
    it with load_topic_names() and pass to this function again after a refit).
    """
    import database as db

    db.update_topic_names(name_mapping)

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "topic_names.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in name_mapping.items()}, f, indent=2)

    print(f"Applied {len(name_mapping)} topic names and saved mapping to {path}")


def load_topic_names() -> dict:
    path = os.path.join(MODEL_DIR, "topic_names.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


if __name__ == "__main__":
    process_all_topics()