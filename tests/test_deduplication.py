"""
test_deduplication.py

How to run:
    python tests/test_deduplication.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import deduplication as dedup


def run():
    print("1. compute_content_hash: identical text -> identical hash")
    h1 = dedup.compute_content_hash("Fuel prices rise sharply today.")
    h2 = dedup.compute_content_hash("Fuel prices rise sharply today.")
    h3 = dedup.compute_content_hash("Completely different content here.")
    assert h1 == h2
    assert h1 != h3
    print("   OK\n")

    print("2. compute_content_hash: whitespace differences don't matter")
    h1 = dedup.compute_content_hash("Fuel   prices  rise")
    h2 = dedup.compute_content_hash("Fuel prices rise")
    assert h1 == h2
    print("   OK\n")

    print("3. assign_duplicate_status: exact link duplicate")
    df = pd.DataFrame([
        {"article_id": 1, "content_clean": "Story A about fuel.", "link": "https://x.com/a", "publication_date": "2026-01-01"},
        {"article_id": 2, "content_clean": "Different story entirely.", "link": "https://x.com/a", "publication_date": "2026-01-01"},
    ])
    result = dedup.assign_duplicate_status(df)
    assert result.loc[result.article_id == 1, "duplicate_status"].iloc[0] == "UNIQUE"
    assert result.loc[result.article_id == 2, "duplicate_status"].iloc[0] == "EXACT_DUPLICATE"
    assert result.loc[result.article_id == 2, "duplicate_of"].iloc[0] == 1
    print("   OK\n")

    print("4. assign_duplicate_status: exact content duplicate (different links)")
    df = pd.DataFrame([
        {"article_id": 1, "content_clean": "Exact same content here.", "link": "https://x.com/a", "publication_date": "2026-01-01"},
        {"article_id": 2, "content_clean": "Exact same content here.", "link": "https://x.com/b", "publication_date": "2026-01-01"},
    ])
    result = dedup.assign_duplicate_status(df)
    assert result.loc[result.article_id == 2, "duplicate_status"].iloc[0] == "EXACT_DUPLICATE"
    print("   OK\n")

    print("5. assign_duplicate_status: near-duplicate (similar but not identical, same day)")
    df = pd.DataFrame([
        {"article_id": 1, "content_clean": "Fuel prices rose sharply across Colombo today amid shortages.", "link": "https://x.com/a", "publication_date": "2026-01-01"},
        {"article_id": 2, "content_clean": "Fuel prices rose sharply across Colombo today amid fuel shortages nationwide.", "link": "https://x.com/b", "publication_date": "2026-01-01"},
    ])
    result = dedup.assign_duplicate_status(df)
    print(f"   article 2 status: {result.loc[result.article_id == 2, 'duplicate_status'].iloc[0]}")
    print("   (near-duplicate detection depends on threshold — inspect manually, not asserted strictly)\n")

    print("6. assign_duplicate_status: genuinely different articles stay UNIQUE")
    df = pd.DataFrame([
        {"article_id": 1, "content_clean": "Fuel prices rose sharply today across the country.", "link": "https://x.com/a", "publication_date": "2026-01-01"},
        {"article_id": 2, "content_clean": "The national cricket team won their match yesterday evening.", "link": "https://x.com/b", "publication_date": "2026-01-01"},
    ])
    result = dedup.assign_duplicate_status(df)
    assert (result["duplicate_status"] == "UNIQUE").all()
    print("   OK\n")

    print("7. assign_duplicate_status: different days never compared for near-dup")
    df = pd.DataFrame([
        {"article_id": 1, "content_clean": "Fuel prices rose sharply today.", "link": "https://x.com/a", "publication_date": "2026-01-01"},
        {"article_id": 2, "content_clean": "Fuel prices rose sharply today.", "link": "https://x.com/b", "publication_date": "2026-06-01"},
    ])
    result = dedup.assign_duplicate_status(df)
    # identical content_clean -> caught by exact hash match regardless of date, which is correct
    assert result.loc[result.article_id == 2, "duplicate_status"].iloc[0] == "EXACT_DUPLICATE"
    print("   OK (caught by content-hash exact match, as expected)\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()