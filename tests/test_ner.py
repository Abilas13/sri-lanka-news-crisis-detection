"""
test_ner.py

How to run:
    python tests/test_ner.py

Mocks spaCy's nlp.pipe() and the database layer — verifies entity-type
mapping, mention-count deduplication, and DB wiring without needing spaCy
installed or a real MySQL connection.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ner


class _FakeEnt:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNLP:
    """Stands in for a real spaCy Language object."""
    def pipe(self, texts, batch_size=64):
        for t in texts:
            if "CIABOC" in t:
                yield _FakeDoc([
                    _FakeEnt("CIABOC", "ORG"),
                    _FakeEnt("CIABOC", "ORG"),          # mentioned twice -> confidence 2
                    _FakeEnt("Colombo", "GPE"),
                    _FakeEnt("Rs. 1 million", "MONEY"),
                    _FakeEnt("yesterday", "DATE"),       # should be filtered out
                ])
            elif "empty" in t:
                yield _FakeDoc([])
            else:
                yield _FakeDoc([_FakeEnt("Sri Lanka", "GPE")])


def run():
    ner._nlp = _FakeNLP()
    ner._get_nlp = lambda: ner._nlp

    print("1. extract_entities_batch: type mapping and DATE filtering")
    texts = ["CIABOC arrested a suspect in Colombo for Rs. 1 million bribery yesterday."]
    results = ner.extract_entities_batch(texts)
    assert len(results) == 1
    entities = results[0]
    types_found = {e["entity_type"] for e in entities}
    assert "ORGANIZATION" in types_found
    assert "LOCATION" in types_found
    assert "MONEY" in types_found
    assert "DATE" not in types_found  # spaCy DATE label must be filtered
    print(f"   OK -> {entities}\n")

    print("2. extract_entities_batch: mention-count deduplication")
    ciaboc_entry = next(e for e in entities if e["entity"] == "CIABOC")
    assert ciaboc_entry["confidence"] == 2.0, "CIABOC mentioned twice should give confidence=2.0"
    print(f"   OK -> CIABOC confidence={ciaboc_entry['confidence']} (mentioned twice)\n")

    print("3. extract_entities_batch: empty entity list handled")
    results = ner.extract_entities_batch(["nothing of note here empty"])
    assert results[0] == []
    print("   OK -> empty list for no entities\n")

    print("4. process_all_ner: mocked DB end-to-end")
    fake_articles = [
        {"article_id": 1, "content_clean": "CIABOC arrested a suspect in Colombo for Rs. 1 million."},
        {"article_id": 2, "content_clean": "Sri Lanka won the match."},
        {"article_id": 3, "content_clean": "nothing of note here empty"},
    ]
    written_rows = []

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

    import database as db
    db.get_articles_needing_ner = lambda limit=None: fake_articles
    db.upsert_article_entities = lambda rows: written_rows.extend(rows)

    ner.process_all_ner()

    article_1_rows = [r for r in written_rows if r["article_id"] == 1]
    article_3_rows = [r for r in written_rows if r["article_id"] == 3]
    assert len(article_1_rows) == 3   # CIABOC, Colombo, Rs. 1 million (DATE filtered)
    assert len(article_3_rows) == 0   # no entities
    print(f"   OK -> wrote {len(written_rows)} entity rows total\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()