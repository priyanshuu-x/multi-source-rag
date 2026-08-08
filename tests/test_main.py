"""
Phase 7 tests - run this after implementing/changing main.py

Mocks the pipeline functions (load_document, index_chunks, hybrid_search,
rerank, generate_answer) so these tests run instantly with no real models,
API keys, or network access needed.

Usage: python test_main.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    print("PASS: /health returns 200 ok")


def test_root_serves_frontend_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    print("PASS: / serves frontend.html")


def test_upload_file_rejects_url_source_types():
    r = client.post(
        "/upload/file",
        files={"file": ("test.md", b"## hello", "text/markdown")},
        data={"source_type": "youtube"},
    )
    assert r.status_code == 400
    print("PASS: /upload/file rejects source_type=youtube (not a file type)")


def test_upload_file_rejects_empty_file():
    r = client.post(
        "/upload/file",
        files={"file": ("empty.md", b"", "text/markdown")},
        data={"source_type": "markdown"},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()
    print("PASS: /upload/file rejects an empty uploaded file")


@patch("main.index_chunks")
@patch("main.chunk_document")
@patch("main.load_document")
def test_upload_file_success_and_cleans_up_temp_file(mock_load, mock_chunk, mock_index):
    import os
    captured_tmp_path = {}

    def fake_load_document(path, source_type):
        captured_tmp_path["path"] = path
        assert os.path.exists(path)  # temp file should exist while processing
        return "some extracted text"

    mock_load.side_effect = fake_load_document
    mock_chunk.return_value = [{"text": "chunk1", "source": "test.md", "source_type": "markdown", "chunk_index": 0}]
    mock_index.return_value = 1

    r = client.post(
        "/upload/file",
        files={"file": ("test.md", b"## hello world", "text/markdown")},
        data={"source_type": "markdown"},
    )
    assert r.status_code == 200
    assert r.json() == {"source": "test.md", "chunks_indexed": 1}
    # temp file should be cleaned up after the request finishes
    assert not os.path.exists(captured_tmp_path["path"])
    print("PASS: /upload/file succeeds and cleans up its temp file afterward")


@patch("main.load_document")
def test_upload_file_value_error_becomes_400(mock_load):
    mock_load.side_effect = ValueError("No text could be extracted")
    r = client.post(
        "/upload/file",
        files={"file": ("test.pdf", b"%PDF-fake-bytes", "application/pdf")},
        data={"source_type": "pdf"},
    )
    assert r.status_code == 400
    assert "No text could be extracted" in r.json()["detail"]
    print("PASS: a ValueError from the pipeline becomes a 400, not a 500")


def test_upload_url_rejects_file_source_types():
    r = client.post("/upload/url", json={"url": "https://example.com", "source_type": "pdf"})
    assert r.status_code == 400
    print("PASS: /upload/url rejects source_type=pdf (not a URL type)")


@patch("main.index_chunks")
@patch("main.chunk_document")
@patch("main.load_document")
def test_upload_url_success(mock_load, mock_chunk, mock_index):
    mock_load.return_value = "extracted website text"
    mock_chunk.return_value = [{"text": "chunk1", "source": "https://example.com", "source_type": "website", "chunk_index": 0}]
    mock_index.return_value = 1

    r = client.post("/upload/url", json={"url": "https://example.com", "source_type": "website"})
    assert r.status_code == 200
    assert r.json() == {"source": "https://example.com", "chunks_indexed": 1}
    print("PASS: /upload/url succeeds with a mocked pipeline")


def test_ask_rejects_empty_query():
    r = client.post("/ask", json={"query": "   "})
    assert r.status_code == 400
    print("PASS: /ask rejects an empty query")


@patch("main.generate_answer")
@patch("main.rerank")
@patch("main.hybrid_search")
def test_ask_success(mock_hybrid, mock_rerank, mock_generate):
    mock_hybrid.return_value = [{"text": "chunk1", "metadata": {"source": "doc1"}}]
    mock_rerank.return_value = [{"text": "chunk1", "metadata": {"source": "doc1"}, "rerank_score": 0.9}]
    mock_generate.return_value = {"answer": "RAG is retrieval-augmented generation", "sources": ["doc1"]}

    r = client.post("/ask", json={"query": "what is RAG?"})
    assert r.status_code == 200
    assert r.json() == {"answer": "RAG is retrieval-augmented generation", "sources": ["doc1"]}
    # top_k is now a fixed internal default (5), not user-supplied - confirm hybrid_search over-fetches 2x that
    assert mock_hybrid.call_args.kwargs.get("top_k") == 10 or mock_hybrid.call_args.args[1] == 10
    print("PASS: /ask wires hybrid_search -> rerank -> generate_answer correctly")


@patch("main.hybrid_search")
def test_ask_no_documents_indexed_becomes_400(mock_hybrid):
    mock_hybrid.side_effect = ValueError("No documents in 'rag_chunks' yet - add chunks before searching")
    r = client.post("/ask", json={"query": "what is RAG?"})
    assert r.status_code == 400
    print("PASS: asking before anything is indexed returns a clear 400, not a 500")


@patch("main.hybrid_search")
def test_ask_unexpected_error_becomes_500(mock_hybrid):
    mock_hybrid.side_effect = RuntimeError("Groq API call failed: simulated")
    r = client.post("/ask", json={"query": "what is RAG?"})
    assert r.status_code == 500
    print("PASS: an unexpected pipeline error returns 500, not a crash")


if __name__ == "__main__":
    test_health_check()
    test_root_serves_frontend_html()
    test_upload_file_rejects_url_source_types()
    test_upload_file_rejects_empty_file()
    test_upload_file_success_and_cleans_up_temp_file()
    test_upload_file_value_error_becomes_400()
    test_upload_url_rejects_file_source_types()
    test_upload_url_success()
    test_ask_rejects_empty_query()
    test_ask_success()
    test_ask_no_documents_indexed_becomes_400()
    test_ask_unexpected_error_becomes_500()
    print("\nAll Phase 7 tests passed.")
