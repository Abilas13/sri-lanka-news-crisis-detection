"""
test_database.py

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlalchemy import text
import database as db


def run():
    print("1. Testing connection...")
    db.test_connection()
    print("   OK\n")

    print("2. Creating schema (sql/create_tables.sql + sql/indexes.sql)...")
    db.init_schema()
    print()

    print("3. Verifying all tables exist...")
    expected_tables = [
        "processed_news", "article_sentiment", "article_topics",
        "article_keywords", "article_entities", "crisis_events",
        "temporal_features", "processing_log",
    ]
    with db.get_connection() as conn:
        existing = {row[0] for row in conn.execute(text("SHOW TABLES"))}
    missing = [t for t in expected_tables if t not in existing]
    if missing:
        print(f"   MISSING TABLES: {missing}")
    else:
        print("   OK — all 8 tables present\n")

    print("4. Checking scraped_news row count...")
    with db.get_connection() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM scraped_news")).scalar()
    print(f"   scraped_news has {count} rows\n")

    print("5. Checking unprocessed article count...")
    unprocessed = db.get_unprocessed_article_ids()
    print(f"   {len(unprocessed)} articles pending processing")
    if unprocessed:
        print(f"   Example raw article: {db.get_raw_article(unprocessed[0])}")


if __name__ == "__main__":
    run()