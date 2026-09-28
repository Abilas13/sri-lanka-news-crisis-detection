"""
ner.py

Purpose
-------
Step 10: extracts named entities (people, organizations, locations, events,
money amounts) using spaCy, for tracking entity frequency over time.

"""

from collections import Counter
import utils

logger = utils.setup_logger(__name__)

SPACY_MODEL_NAME = "en_core_web_sm"

_ENTITY_TYPE_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "EVENT": "EVENT",
    "MONEY": "MONEY",
}

_nlp = None


def _get_nlp():
    """
    Lazily loads spaCy, with unneeded pipeline components disabled for
    speed (tagger/parser/lemmatizer aren't needed for NER alone).
    """
    global _nlp
    if _nlp is None:
        import spacy
        logger.info(f"Loading spaCy model {SPACY_MODEL_NAME}...")
        _nlp = spacy.load(SPACY_MODEL_NAME, disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
        logger.info("spaCy model loaded.")
    return _nlp


def extract_entities_batch(texts: list[str], batch_size: int = 64) -> list[list[dict]]:
    """
    Extracts entities for a batch of texts using spaCy's efficient nlp.pipe().
    Returns a list (same order as input) of lists of
    {'entity': str, 'entity_type': str, 'confidence': int} — confidence is
    the mention count within that article (see module docstring).
    Entities are deduplicated per article (same text+type counted once,
    with confidence = how many times it appeared).
    """
    nlp = _get_nlp()
    results = []

    for doc in nlp.pipe(texts, batch_size=batch_size):
        counts = Counter()
        for ent in doc.ents:
            mapped_type = _ENTITY_TYPE_MAP.get(ent.label_)
            if mapped_type is None:
                continue  # not one of our six tracked types, skip
            entity_text = ent.text.strip()
            if not entity_text:
                continue
            counts[(entity_text, mapped_type)] += 1

        results.append([
            {"entity": entity_text, "entity_type": entity_type, "confidence": float(count)}
            for (entity_text, entity_type), count in counts.items()
        ])

    return results


def process_all_ner(batch_size: int = 64, limit: int = None):
    """
    Fetches all COMPLETED articles without entities yet, extracts entities
    in batches via nlp.pipe(), upserts into article_entities.
    """
    import database as db

    articles = db.get_articles_needing_ner(limit=limit)
    total = len(articles)
    logger.info(f"process_all_ner: {total} articles to process")

    if total == 0:
        print("Nothing to process — all articles already have entities.")
        return

    processed = 0
    chunk_size = 500  # outer progress-reporting chunk; nlp.pipe handles its own internal batching
    for start in range(0, total, chunk_size):
        chunk = articles[start:start + chunk_size]
        ids = [row["article_id"] for row in chunk]
        texts = [row["content_clean"] or "" for row in chunk]

        entity_lists = extract_entities_batch(texts, batch_size=batch_size)

        rows = []
        for aid, ent_list in zip(ids, entity_lists):
            for ent in ent_list:
                rows.append({
                    "article_id": aid,
                    "entity": ent["entity"][:255],   # matches VARCHAR(255) column
                    "entity_type": ent["entity_type"],
                    "confidence": ent["confidence"],
                })

        db.upsert_article_entities(rows)
        processed += len(ids)
        print(f"  Progress: {min(start + chunk_size, total)}/{total}")

    print(f"\nDone. Entities extracted for {processed} articles.")


if __name__ == "__main__":
    process_all_ner()