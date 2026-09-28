"""
test_date_processing.py


Tests normalize_date() against real values confirmed from the dataset,
plus edge cases (empty, garbage, wrong format) that must fail safely.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import date_processing as dp


def run():
    print("1. Valid real value: '01 Apr 2026'")
    r = dp.normalize_date("01 Apr 2026")
    assert r["date_parse_status"] == "OK"
    assert r["publication_year"] == 2026
    assert r["publication_month"] == 4
    assert r["publication_day"] == 1
    assert r["publication_week"] is not None
    print(f"   OK -> {r}\n")

    print("2. Valid real value: '27 Nov 2025'")
    r = dp.normalize_date("27 Nov 2025")
    assert r["date_parse_status"] == "OK"
    assert r["publication_year"] == 2025
    assert r["publication_month"] == 11
    print(f"   OK -> published_at={r['published_at']}, week={r['publication_week']}\n")

    print("3. Empty string")
    r = dp.normalize_date("")
    assert r["date_parse_status"] == "FAILED"
    assert r["published_at"] is None
    print("   OK -> correctly flagged as FAILED, no invented date\n")

    print("4. None input")
    r = dp.normalize_date(None)
    assert r["date_parse_status"] == "FAILED"
    print("   OK -> correctly flagged as FAILED\n")

    print("5. Garbage string")
    r = dp.normalize_date("not a date at all")
    assert r["date_parse_status"] == "FAILED"
    print("   OK -> correctly flagged as FAILED\n")

    print("6. Wrong format (ISO instead of DD Mon YYYY)")
    r = dp.normalize_date("2026-04-01")
    assert r["date_parse_status"] == "FAILED"
    print("   OK -> correctly flagged as FAILED (this format isn't in your data, "
          "but the function shouldn't silently misparse it either)\n")

    print("7. Batch processing")
    batch = dp.normalize_dates_batch(["01 Apr 2026", "27 Nov 2025", "garbage"])
    ok_count = sum(1 for r in batch if r["date_parse_status"] == "OK")
    print(f"   {ok_count}/3 parsed successfully (expected 2)\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()