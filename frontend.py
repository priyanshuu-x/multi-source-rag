import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


# ---------- Backend-calling functions (pure, unit-testable - see test_frontend.py) ----------

def get_error_detail(response: requests.Response) -> str:
    """Safely extracts an error message from a response, even if it's not valid JSON."""
    try:
        return response.json().get("detail", f"Request failed with status {response.status_code}")
    except ValueError:
        return f"Request failed with status {response.status_code}: {response.text[:200]}"


def check_backend_health(backend_url: str) -> bool:
    try:
        r = requests.get(f"{backend_url}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def upload_file_to_backend(backend_url: str, filename: str, file_bytes: bytes, source_type: str) -> dict:
    try:
        r = requests.post(
            f"{backend_url}/upload/file",
            files={"file": (filename, file_bytes)},
            data={"source_type": source_type},
            timeout=120,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach the backend: {e}")
    if r.status_code != 200:
        raise RuntimeError(get_error_detail(r))
    return r.json()


def upload_url_to_backend(backend_url: str, url: str, source_type: str) -> dict:
    try:
        r = requests.post(
            f"{backend_url}/upload/url",
            json={"url": url, "source_type": source_type},
            timeout=120,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach the backend: {e}")
    if r.status_code != 200:
        raise RuntimeError(get_error_detail(r))
    return r.json()


def ask_backend(backend_url: str, query: str) -> dict:
    try:
        r = requests.post(
            f"{backend_url}/ask",
            json={"query": query},
            timeout=60,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach the backend: {e}")
    if r.status_code != 200:
        raise RuntimeError(get_error_detail(r))
    return r.json()


# ---------- Streamlit UI (thin layer on top of the functions above) ----------

def main():
    st.set_page_config(page_title="Multi-Source RAG", page_icon="📚")
    st.title("📚 Multi-Source RAG")

    if "history" not in st.session_state:
        st.session_state.history = []

    if not check_backend_health(BACKEND_URL):
        st.error(f"Can't reach the backend at {BACKEND_URL}. Make sure it's running (`python main.py`).")
        st.stop()

    with st.sidebar:
        st.header("Add a source")
        upload_mode = st.radio("Source type", ["File (PDF / DOCX / Markdown)", "URL (Website / YouTube)"])

        if upload_mode.startswith("File"):
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "md"])
            file_source_type = st.selectbox("File type", ["pdf", "docx", "markdown"])
            if st.button("Upload file", disabled=uploaded_file is None):
                with st.spinner("Processing file..."):
                    try:
                        result = upload_file_to_backend(
                            BACKEND_URL, uploaded_file.name, uploaded_file.getvalue(), file_source_type
                        )
                        st.success(f"Indexed {result['chunks_indexed']} chunks from {result['source']}")
                    except RuntimeError as e:
                        st.error(str(e))
        else:
            url = st.text_input("URL")
            url_source_type = st.selectbox("URL type", ["website", "youtube"])
            if st.button("Upload URL", disabled=not url.strip()):
                with st.spinner("Processing URL..."):
                    try:
                        result = upload_url_to_backend(BACKEND_URL, url, url_source_type)
                        st.success(f"Indexed {result['chunks_indexed']} chunks from {result['source']}")
                    except RuntimeError as e:
                        st.error(str(e))

    st.header("Ask a question")
    query = st.text_input("Your question")

    if st.button("Ask", disabled=not query.strip()):
        with st.spinner("Thinking..."):
            try:
                result = ask_backend(BACKEND_URL, query)
                st.session_state.history.append(
                    {"query": query, "answer": result["answer"], "sources": result["sources"]}
                )
            except RuntimeError as e:
                st.error(str(e))

    for item in reversed(st.session_state.history):
        st.markdown(f"**Q: {item['query']}**")
        st.write(item["answer"])
        st.caption("Sources: " + ", ".join(item["sources"]))
        st.divider()


if __name__ == "__main__":
    main()