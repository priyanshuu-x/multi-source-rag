# 📚 Multi-Source RAG

A hybrid-search RAG (Retrieval-Augmented Generation) system that lets you ask questions over **Websites, YouTube videos, PDFs, DOCX files, and Markdown files** — all through one unified pipeline, one API, and one UI.

Built end-to-end as a learning project: from raw document loaders all the way through evaluation, monitoring, CI/CD, and Docker. 🚀

---

## ✨ Features

- 📄 **5 source types** — Website, YouTube, PDF, DOCX, Markdown, all normalized into the same pipeline
- 🔍 **Hybrid search** — dense (embeddings) + sparse (BM25) retrieval, fused with Reciprocal Rank Fusion
- 🎯 **Reranking** — a cross-encoder re-scores retrieved chunks before they reach the LLM
- 🤖 **Grounded answers** — Groq-powered generation that cites its sources, refuses to answer from outside the given context
- 📊 **Evaluation** — custom LLM-as-judge scoring faithfulness, answer relevancy, context precision, and context recall
- 🔭 **Monitoring** — every step of the pipeline is traced in LangSmith
- ✅ **Tested** — 8 test suites covering every phase, runnable in one command
- 🐳 **Dockerized** — one container runs the whole thing
- ⚙️ **CI** — GitHub Actions runs the full test suite on every push

---

## 🧠 How it works

```
User
  │
  ▼
FastAPI ──► Loader Selection ──► Chunking ──► Embedding
                                                  │
                                                  ▼
                                          Hybrid Search (dense + BM25)
                                                  │
                                                  ▼
                                              Reranker
                                                  │
                                                  ▼
                                                 LLM
                                                  │
                                                  ▼
                                              Response
```

1. **Loaders** (`loaders.py`) — turn any of the 5 source types into clean, normalized plain text
2. **Chunking** (`chunking.py`) — splits text into overlapping word-windows, tagged with source metadata
3. **Embedding** (`embedding.py`) — dense vectors via `sentence-transformers`, stored in an in-memory ChromaDB collection
4. **Hybrid Search** (`retrieval.py`) — BM25 keyword search + dense semantic search, fused via Reciprocal Rank Fusion
5. **Reranker** (`reranker.py`) — a cross-encoder re-scores the fused candidates for real relevance
6. **LLM** (`llm.py`) — Groq generates a grounded, cited answer from the top reranked chunks
7. **Evaluation** (`eval.py`) — a second LLM acts as a judge, scoring the whole pipeline's output on 4 metrics
8. **Monitoring** — every function above is wrapped in `@traceable`, so a full request trace shows up in LangSmith automatically

---

## 🛠️ Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI |
| Frontend | Single-page HTML/CSS/JS (`frontend.html`), served directly by FastAPI |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (in-memory) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Reranker | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Evaluation | Custom LLM-as-judge via Groq |
| Monitoring | LangSmith |
| Containerization | Docker |
| CI | GitHub Actions |

---

## 📁 Project structure

```
.
├── main.py              # FastAPI app - serves the API AND the frontend
├── loaders.py            # Website / YouTube / PDF / DOCX / Markdown loaders
├── chunking.py            # Text chunking
├── embedding.py            # ChromaDB dense store + dense search
├── retrieval.py            # BM25 sparse search + hybrid fusion
├── reranker.py            # Cross-encoder reranking
├── llm.py                # Groq answer generation
├── eval.py                # LLM-as-judge evaluation runner
├── eval_dataset.json        # Sample evaluation questions
├── frontend.html           # The entire UI (single file, no build step)
├── requirements.txt
├── .env.example            # Copy to .env and fill in your keys
├── .gitignore
├── Dockerfile
├── .dockerignore
├── .github/workflows/ci-cd.yml
└── tests/
    ├── run_all_tests.py     # Runs every test file below in one go
    ├── test_loaders.py
    ├── test_chunking.py
    ├── test_embedding.py
    ├── test_retrieval.py
    ├── test_reranker.py
    ├── test_llm.py
    ├── test_main.py
    └── test_eval.py
```

---

## 🚀 Getting started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### 2. Set up your environment

```bash
cp .env.example .env
```

Fill in:
- `GROQ_API_KEY` — free at [console.groq.com](https://console.groq.com)
- `LANGSMITH_API_KEY` — free at [smith.langchain.com](https://smith.langchain.com)

### 3. Run it 🎬

```bash
python main.py
```

Your browser should open automatically at `http://localhost:8000/`. If it doesn't, open that link manually (⚠️ not the `0.0.0.0` address shown in the logs — that's just the server's bind address, not a real link).

The first run will be slow — it's downloading the embedding and reranker models. After that, it's fast.

---

## 🐳 Running with Docker

```bash
docker build -t rag-app .
docker run -p 8000:8000 --env-file .env rag-app
```

Then visit `http://localhost:8000/` the same way.

---

## 🧪 Testing

Run the whole suite in one command:

```bash
python tests/run_all_tests.py
```

Every test uses fakes/mocks for models and APIs, so this runs instantly with no API key, no downloads, and no real network calls needed — safe to run anywhere, including CI.

---

## 📊 Running an evaluation

Edit `eval_dataset.json` with questions relevant to whatever you've actually indexed, then:

```bash
python eval.py
```

This prints faithfulness, answer relevancy, context precision, and context recall scores, and saves full results to `eval_results.json`.

---

## 🔌 API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the frontend UI |
| `/health` | GET | Health check |
| `/upload/file` | POST | Upload a PDF, DOCX, or Markdown file |
| `/upload/url` | POST | Ingest a Website or YouTube URL |
| `/ask` | POST | Ask a question over everything indexed so far |

Full interactive docs (Swagger UI) are available at `http://localhost:8000/docs` while the server is running.

---

## ⚠️ Known limitations

- **In-memory storage** — ChromaDB and the BM25 index both live in memory. Restarting the server clears everything that was indexed. This was a deliberate simplicity trade-off, not an oversight.
- **BM25 rebuilds on every add** — `rank_bm25` has no incremental-update API, so adding chunks rebuilds the whole sparse index. Fine at portfolio scale, would need a different approach for a large, frequently-updated corpus.
- **No OCR** — scanned PDFs with no text layer will raise a clear error rather than silently returning nothing.
- **Website scraping** — basic bot-protection bypass (a real browser User-Agent) is included, but sites with stronger anti-bot measures may still block requests.

---

## 🙏 Credits

Built as a hands-on learning project covering the full lifecycle of a production-minded RAG system — from raw document ingestion through evaluation, monitoring, and deployment tooling. 💜
