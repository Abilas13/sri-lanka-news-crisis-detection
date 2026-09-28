"""
test_cleaning.py

"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cleaning


def run():
    print("1. validate_article: valid article")
    article = {
        "id": 1, "title": "Some title", "content": "A" * 100,
        "date": "01 Apr 2026", "link": "https://example.com/1"
    }
    ok, reason = cleaning.validate_article(article)
    assert ok is True and reason == ""
    print("   OK\n")

    print("2. validate_article: missing content")
    article2 = {"id": 2, "title": "Title", "content": None, "date": "x", "link": "y"}
    ok, reason = cleaning.validate_article(article2)
    assert ok is False and "MISSING_FIELD" in reason
    print(f"   OK -> correctly rejected: {reason}\n")

    print("3. validate_article: too short")
    article3 = {"id": 3, "title": "Title", "content": "short", "date": "x", "link": "y"}
    ok, reason = cleaning.validate_article(article3)
    assert ok is False and "CONTENT_TOO_SHORT" in reason
    print(f"   OK -> correctly rejected: {reason}\n")

    print("4. clean_text: HTML stripping")
    dirty = "<p>Breaking <b>news</b>: fuel prices <i>rise</i>.</p>"
    clean = cleaning.clean_text(dirty)
    assert "<" not in clean and ">" not in clean
    assert "Breaking" in clean and "news" in clean
    print(f"   OK -> {clean!r}\n")

    print("5. clean_text: excessive whitespace")
    dirty = "Too    much   \n\n\n  whitespace   here"
    clean = cleaning.clean_text(dirty)
    assert "  " not in clean
    print(f"   OK -> {clean!r}\n")

    print("6. clean_text: boilerplate removal")
    dirty = "Real article content here. Follow us on Facebook and Twitter for updates."
    clean = cleaning.clean_text(dirty)
    assert "Follow us on" not in clean
    assert "Real article content" in clean
    print(f"   OK -> {clean!r}\n")

    print("7. clean_text: encoding fix (mojibake)")
    dirty = "Itâ€™s a test"
    clean = cleaning.clean_text(dirty)
    if cleaning._FTFY_AVAILABLE:
        assert "â€™" not in clean
        print(f"   OK -> {clean!r}\n")
    else:
        print("   SKIPPED (ftfy not installed — run: pip install ftfy)\n")

    print("8. compute_basic_features")
    content = "This is sentence one. This is sentence two! Is this sentence three?"
    features = cleaning.compute_basic_features(content)
    assert features["sentence_count"] == 3
    assert features["article_length_words"] > 0
    print(f"   OK -> {features}\n")

    print("9. clean_article: full pipeline")
    result = cleaning.clean_article(
        "<b>Title</b>", "<p>Content here. Follow us on Facebook.</p>"
    )
    assert "title_clean" in result and "content_clean" in result
    assert "article_length_words" in result
    print(f"   OK -> {result}\n")

    print("10. clean_text: empty input doesn't crash")
    assert cleaning.clean_text("") == ""
    assert cleaning.clean_text(None) == ""
    print("   OK\n")

    print("All tests passed.")


if __name__ == "__main__":
    run()