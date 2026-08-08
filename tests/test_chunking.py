"""
Phase 2 tests - run this after implementing/changing chunking.py

Usage: python test_chunking.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunking import chunk_text, chunk_document


def test_empty_text_raises():
    for bad in ["", "   ", "\n\n"]:
        try:
            chunk_text(bad)
            raise AssertionError("FAILED: should have raised ValueError for empty text")
        except ValueError:
            pass
    print("PASS: empty/whitespace text raises ValueError")


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_text("word " * 500, chunk_size=100, overlap=100)
        raise AssertionError("FAILED: should have raised ValueError when overlap == chunk_size")
    except ValueError:
        pass
    print("PASS: overlap >= chunk_size raises ValueError")


def test_short_text_returns_single_chunk():
    text = "This is a short sentence with under three hundred words."
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text
    print("PASS: short text returns exactly one chunk")


def test_long_text_overlaps_correctly():
    words = [f"word{i}" for i in range(700)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert len(chunks) == 3, f"FAILED: expected 3 chunks, got {len(chunks)}"

    # Check overlap: last 50 words of chunk 0 should equal first 50 words of chunk 1
    chunk0_words = chunks[0].split()
    chunk1_words = chunks[1].split()
    assert chunk0_words[-50:] == chunk1_words[:50], "FAILED: overlap between chunk 0 and 1 is wrong"

    chunk2_words = chunks[2].split()
    assert chunk2_words[-1] == "word699", "FAILED: last chunk doesn't reach the end of the text"
    print("PASS: long text splits into correctly overlapping chunks")


def test_chunk_document_attaches_metadata():
    text = " ".join(f"word{i}" for i in range(700))
    result = chunk_document(text, source="test.md", source_type="markdown", chunk_size=300, overlap=50)

    assert len(result) == 3
    for i, chunk in enumerate(result):
        assert chunk["source"] == "test.md"
        assert chunk["source_type"] == "markdown"
        assert chunk["chunk_index"] == i
        assert isinstance(chunk["text"], str) and len(chunk["text"]) > 0
    print("PASS: chunk_document attaches correct metadata to every chunk")


if __name__ == "__main__":
    test_empty_text_raises()
    test_overlap_must_be_smaller_than_chunk_size()
    test_short_text_returns_single_chunk()
    test_long_text_overlaps_correctly()
    test_chunk_document_attaches_metadata()
    print("\nAll Phase 2 tests passed.")
