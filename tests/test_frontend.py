"""
Phase 8 tests - run this after implementing/changing frontend.py

Tests only the pure backend-calling functions (get_error_detail,
check_backend_health, upload_file_to_backend, upload_url_to_backend,
ask_backend) using a mocked requests module - not the Streamlit UI itself,
which isn't practical to unit test outside a running Streamlit session.

Usage: python test_frontend.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
import requests
from frontend import (
    get_error_detail, check_backend_health,
    upload_file_to_backend, upload_url_to_backend, ask_backend,
)

BACKEND_URL = "http://fake-backend:8000"


def fake_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("not json")
    return resp


def test_get_error_detail_with_json_body():
    resp = fake_response(400, json_data={"detail": "bad request"})
    assert get_error_detail(resp) == "bad request"
    print("PASS: get_error_detail extracts the 'detail' field from a JSON error body")


def test_get_error_detail_with_non_json_body():
    resp = fake_response(502, text="<html>Bad Gateway</html>")
    detail = get_error_detail(resp)
    assert "502" in detail
    print("PASS: get_error_detail falls back gracefully for a non-JSON error body")


@patch("requests.get")
def test_check_backend_health_true(mock_get):
    mock_get.return_value = fake_response(200, json_data={"status": "ok"})
    assert check_backend_health(BACKEND_URL) is True
    print("PASS: check_backend_health returns True on 200")


@patch("requests.get")
def test_check_backend_health_false_on_connection_error(mock_get):
    mock_get.side_effect = requests.ConnectionError("refused")
    assert check_backend_health(BACKEND_URL) is False
    print("PASS: check_backend_health returns False (not a crash) when the backend is down")


@patch("requests.post")
def test_upload_file_to_backend_success(mock_post):
    mock_post.return_value = fake_response(200, json_data={"source": "test.pdf", "chunks_indexed": 5})
    result = upload_file_to_backend(BACKEND_URL, "test.pdf", b"fake bytes", "pdf")
    assert result == {"source": "test.pdf", "chunks_indexed": 5}
    print("PASS: upload_file_to_backend returns parsed JSON on success")


@patch("requests.post")
def test_upload_file_to_backend_error_raises_runtime_error(mock_post):
    mock_post.return_value = fake_response(400, json_data={"detail": "Unsupported source type"})
    try:
        upload_file_to_backend(BACKEND_URL, "test.pdf", b"fake bytes", "pdf")
        raise AssertionError("FAILED: should raise RuntimeError on non-200 response")
    except RuntimeError as e:
        assert "Unsupported source type" in str(e)
        print("PASS: upload_file_to_backend raises a clear RuntimeError with the backend's message")


@patch("requests.post")
def test_upload_file_to_backend_connection_error(mock_post):
    mock_post.side_effect = requests.ConnectionError("refused")
    try:
        upload_file_to_backend(BACKEND_URL, "test.pdf", b"fake bytes", "pdf")
        raise AssertionError("FAILED: should raise RuntimeError when backend is unreachable")
    except RuntimeError as e:
        assert "Could not reach the backend" in str(e)
        print("PASS: upload_file_to_backend wraps connection errors clearly")


@patch("requests.post")
def test_upload_url_to_backend_success(mock_post):
    mock_post.return_value = fake_response(200, json_data={"source": "https://x.com", "chunks_indexed": 3})
    result = upload_url_to_backend(BACKEND_URL, "https://x.com", "website")
    assert result == {"source": "https://x.com", "chunks_indexed": 3}
    print("PASS: upload_url_to_backend returns parsed JSON on success")


@patch("requests.post")
def test_ask_backend_success(mock_post):
    mock_post.return_value = fake_response(200, json_data={"answer": "RAG is retrieval-augmented generation", "sources": ["doc1"]})
    result = ask_backend(BACKEND_URL, "what is RAG?")
    assert result["answer"] == "RAG is retrieval-augmented generation"
    assert result["sources"] == ["doc1"]
    print("PASS: ask_backend returns parsed JSON on success")


@patch("requests.post")
def test_ask_backend_error_raises_runtime_error(mock_post):
    mock_post.return_value = fake_response(400, json_data={"detail": "No documents indexed yet"})
    try:
        ask_backend(BACKEND_URL, "what is RAG?")
        raise AssertionError("FAILED: should raise RuntimeError on non-200 response")
    except RuntimeError as e:
        assert "No documents indexed yet" in str(e)
        print("PASS: ask_backend surfaces the backend's error message clearly")


if __name__ == "__main__":
    test_get_error_detail_with_json_body()
    test_get_error_detail_with_non_json_body()
    test_check_backend_health_true()
    test_check_backend_health_false_on_connection_error()
    test_upload_file_to_backend_success()
    test_upload_file_to_backend_error_raises_runtime_error()
    test_upload_file_to_backend_connection_error()
    test_upload_url_to_backend_success()
    test_ask_backend_success()
    test_ask_backend_error_raises_runtime_error()
    print("\nAll Phase 8 tests passed.")
