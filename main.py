import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langsmith import traceable

from loaders import load_document
from chunking import chunk_document
from retrieval import index_chunks, hybrid_search
from reranker import rerank
from llm import generate_answer

app = FastAPI(title="Multi-Source RAG API")

# Allows the Streamlit frontend (different port = different origin) to call this API.
# "*" is fine for local/beginner use - a real production deploy should restrict this
# to the frontend's actual URL instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FILE_SOURCE_TYPES = {"pdf", "docx", "markdown"}
URL_SOURCE_TYPES = {"website", "youtube"}
DEFAULT_TOP_K = 5  # fixed internally now - no longer a user-facing option (removed per request)


class UrlUploadRequest(BaseModel):
    url: str
    source_type: str  # "website" or "youtube"


class AskRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("frontend.html")


@traceable(run_type="chain", name="upload_file_pipeline")
def _process_file_upload(tmp_path: str, filename: str, source_type: str) -> dict:
    text = load_document(tmp_path, source_type)
    chunks = chunk_document(text, source=filename, source_type=source_type)
    count = index_chunks(chunks)
    return {"source": filename, "chunks_indexed": count}


@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...), source_type: str = Form(...)):
    """For PDF, DOCX, and Markdown - anything that comes in as an uploaded file."""
    if source_type not in FILE_SOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"source_type must be one of {sorted(FILE_SOURCE_TYPES)} for file uploads",
        )

    suffix = os.path.splitext(file.filename or "")[1]
    tmp_path = None
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        return _process_file_upload(tmp_path, file.filename, source_type)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing file: {e}")
    finally:
        # Always clean up the temp file, whether processing succeeded or failed
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@traceable(run_type="chain", name="upload_url_pipeline")
def _process_url_upload(url: str, source_type: str) -> dict:
    text = load_document(url, source_type)
    chunks = chunk_document(text, source=url, source_type=source_type)
    count = index_chunks(chunks)
    return {"source": url, "chunks_indexed": count}


@app.post("/upload/url")
def upload_url(request: UrlUploadRequest):
    """For Website and YouTube sources - anything that comes in as a URL."""
    if request.source_type not in URL_SOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"source_type must be one of {sorted(URL_SOURCE_TYPES)} for URL uploads",
        )

    try:
        return _process_url_upload(request.url, request.source_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing URL: {e}")


@traceable(run_type="chain", name="ask_pipeline")
def _run_ask_pipeline(query: str) -> dict:
    candidates = hybrid_search(query, top_k=DEFAULT_TOP_K * 2)  # over-fetch before reranking
    reranked = rerank(query, candidates, top_k=DEFAULT_TOP_K)
    return generate_answer(query, reranked)


@app.post("/ask")
def ask(request: AskRequest):
    """Runs the full Hybrid Search -> Reranker -> LLM pipeline for one question."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        return _run_ask_pipeline(request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error answering question: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)