"""
keywords.py

Purpose
-------
Step 9: extracts the top N meaningful keywords/keyphrases per article using
KeyBERT, reusing the same sentence-transformer embedding model as topics.py
(all-MiniLM-L6-v2) — no need to load a second embedding model.

"""

import utils

logger = utils.setup_logger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # same as topics.py

_keybert_model = None


def _get_keybert_model():
    """Lazily loads KeyBERT, backed by the same sentence-transformer used in topics.py."""
    global _keybert_model
    if _keybert_model is None:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model {EMBEDDING_MODEL_NAME} for KeyBERT...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _keybert_model = KeyBERT(model=embedder)
        logger.info("KeyBERT model loaded.")
    return _keybert_model


def extract_keywords_batch(texts: list[str], top_n: int = 10) -> list[list[dict]]:
    """
    Extracts top-N keywords for each text in a batch.
    Returns a list (same order as input) of lists of
    {'keyword': str, 'score': float, 'rank': int}.
    Empty texts get an empty keyword list without calling the model.
    """
    kw_model = _get_keybert_model()

    results = [None] * len(texts)
    real_indices = []
    real_texts = []
    for i, t in enumerate(texts):
        if not t or not t.strip():
            results[i] = []
        else:
            real_indices.append(i)
            real_texts.append(t)

    if real_texts:
        raw_results = kw_model.extract_keywords(
            real_texts,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=top_n,
        )
        # extract_keywords returns a flat list of (kw, score) tuples for a
        # single doc, or a list-of-lists for multiple docs — normalize just in case.
        if real_texts and len(real_texts) == 1 and raw_results and isinstance(raw_results[0], tuple):
            raw_results = [raw_results]

        for idx, doc_keywords in zip(real_indices, raw_results):
            results[idx] = [
                {"keyword": kw, "score": float(score), "rank": rank}
                for rank, (kw, score) in enumerate(doc_keywords, start=1)
            ]

    return results


def process_all_keywords(top_n: int = 10, batch_size: int = 100, limit: int = None):
    """
    Fetches all COMPLETED articles without keywords yet, extracts top-N
    keywords per article in batches, upserts into article_keywords.
    """
    import database as db

    articles = db.get_articles_needing_keywords(limit=limit)
    total = len(articles)
    logger.info(f"process_all_keywords: {total} articles to process")

    if total == 0:
        print("Nothing to process — all articles already have keywords.")
        return

    processed = 0
    for start in range(0, total, batch_size):
        chunk = articles[start:start + batch_size]
        ids = [row["article_id"] for row in chunk]
        texts = [row["content_clean"] for row in chunk]

        keyword_lists = extract_keywords_batch(texts, top_n=top_n)

        rows = []
        for aid, kw_list in zip(ids, keyword_lists):
            for kw_entry in kw_list:
                rows.append({
                    "article_id": aid,
                    "keyword": kw_entry["keyword"],
                    "score": kw_entry["score"],
                    "rank": kw_entry["rank"],
                })

        db.upsert_article_keywords(rows)
        processed += len(ids)
        print(f"  Progress: {min(start + batch_size, total)}/{total}")

    print(f"\nDone. Keywords extracted for {processed} articles.")


if __name__ == "__main__":
    process_all_keywords()