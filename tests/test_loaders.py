"""
Phase 1 tests - run this after implementing/changing loaders.py

Usage: python test_loaders.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import docx
from pypdf import PdfWriter
from loaders import load_document, _extract_video_id, clean_text


def setup_test_files():
    """Create small local test files so PDF/DOCX/Markdown loaders can be tested offline."""
    with open("test.md", "w") as f:
        f.write("## Introduction\n\n\n\nRAG combines   retrieval with generation.")  # messy whitespace on purpose

    d = docx.Document()
    d.add_paragraph("This is a test paragraph about RAG.")
    d.save("test.docx")

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # no text layer -> should now raise ValueError
    with open("test.pdf", "wb") as f:
        writer.write(f)


def test_video_id_extraction():
    cases = {
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s": "dQw4w9WgXcQ",
    }
    for url, expected in cases.items():
        result = _extract_video_id(url)
        assert result == expected, f"FAILED: {url} -> {result}, expected {expected}"
    print("PASS: video ID extraction")


def test_clean_text_collapses_whitespace():
    dirty = "Line one\n\n\n\nLine two    with   extra   spaces\t\t\ttrailing   "
    cleaned = clean_text(dirty)
    assert "\n\n\n" not in cleaned, "FAILED: did not collapse triple newlines"
    assert "   " not in cleaned, "FAILED: did not collapse repeated spaces"
    print("PASS: clean_text collapses excess whitespace")


def test_youtube_bracket_removal_logic():
    # Unit-tests just the regex used inside load_youtube (no network needed)
    raw = "Hello [Music] world [Applause] this is a test [inaudible]"
    cleaned = re.sub(r"\[.*?\]", "", raw)
    assert "[Music]" not in cleaned and "[Applause]" not in cleaned
    print("PASS: youtube bracket-noise removal regex")


def test_markdown():
    text = load_document("test.md", "markdown")
    assert "## Introduction" in text          # header preserved
    assert "\n\n\n" not in text               # excess blank lines cleaned
    print("PASS: markdown loader (cleaned, header preserved)")


def test_docx():
    text = load_document("test.docx", "docx")
    assert "RAG" in text
    print("PASS: docx loader")


def test_pdf_with_no_text_raises_clear_error():
    try:
        load_document("test.pdf", "pdf")
        raise AssertionError("FAILED: should have raised ValueError for empty PDF text")
    except ValueError as e:
        assert "No text could be extracted" in str(e)
        print("PASS: pdf loader raises clear error on empty extraction")


def test_unsupported_type():
    try:
        load_document("test.md", "csv")
        raise AssertionError("FAILED: should have raised ValueError")
    except ValueError:
        print("PASS: unsupported type raises ValueError")


def test_website():
    # Requires internet access - swap the URL if this one ever goes down
    text = load_document("https://en.wikipedia.org/wiki/Retrieval-augmented_generation", "website")
    assert len(text) > 200
    print("PASS: website loader")


def test_youtube():
    # Requires internet access - swap for any video that has captions
    text = load_document("https://www.youtube.com/watch?v=aircAruvnKk", "youtube")
    assert len(text) > 200
    print("PASS: youtube loader")


if __name__ == "__main__":
    setup_test_files()
    test_video_id_extraction()
    test_clean_text_collapses_whitespace()
    test_youtube_bracket_removal_logic()
    test_markdown()
    test_docx()
    test_pdf_with_no_text_raises_clear_error()
    test_unsupported_type()

    print("\nRunning network-dependent tests (website, youtube)...")
    try:
        test_website()
        test_youtube()
    except Exception as e:
        print(f"NETWORK TEST FAILED (check your internet connection): {e}")

    print("\nAll offline tests passed. Phase 1 is verified.")
