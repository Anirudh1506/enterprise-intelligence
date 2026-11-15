# Enterprise Intelligence

Upload organizational documents (PDF, TXT, Markdown), search them semantically, and ask questions that are answered **only** from the documents, with citations back to the source passages.

```
upload ─► store file ─► extract + clean text ─► chunk ─► embed (Gemini) ─► pgvector
                                                                              │
question ─► embed query ─► cosine top-k ─► numbered context ─► LLM ─► answer + citations
```

- **Backend:** FastAPI, SQLAlchemy 2, Alembic, PostgreSQL + pgvector, PyMuPDF, Google Gemini
- **Frontend:** React, TypeScript, Vite, Tailwind CSS

## Quick start

Prerequisites: Python 3.9+, Node 20+, PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector).

### 1. Database

macOS (Homebrew):

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
/opt/homebrew/opt/postgresql@17/bin/createdb enterprise_intelligence
/opt/homebrew/opt/postgresql@17/bin/createdb enterprise_intelligence_test   # for tests
```

Or with Docker (any OS):

```bash
docker run -d --name ei-postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17
# then use DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres
```

The migration runs `CREATE EXTENSION vector` itself; the database user needs permission to create extensions (the default local superuser does).

### 2. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then set GEMINI_API_KEY (optional, see below)
alembic upgrade head          # create tables + pgvector index
uvicorn app.main:app --reload # http://localhost:8000, API docs at /docs
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

The Vite dev server proxies `/api` and `/health` to `http://localhost:8000` (override with `VITE_BACKEND_URL`).

### 4. Try it

Upload the files in [`sample-docs/`](sample-docs/) on the **Documents** page, wait for `COMPLETED`, then go to **Ask** and try:

- "How many days of annual leave do I get and how many can I carry over?"
- "Can I deploy on a Friday?"
- "What is the company's stock price?" (not in the documents, so you should get an explicit "not enough information" answer)

## AI providers

| Setting | Values | Default |
|---|---|---|
| `EMBEDDING_PROVIDER` | `auto`, `gemini`, `mock` | `auto` |
| `LLM_PROVIDER` | `auto`, `gemini`, `mock` | `auto` |

`auto` uses Gemini when `GEMINI_API_KEY` is set, otherwise the mock providers. Get a key at <https://aistudio.google.com/apikey>.

**Mock providers (no API key).** These exist so the app runs end to end locally. They don't pretend to be AI:

- *Mock embeddings* are hashed bag-of-words vectors. Search becomes keyword overlap, not semantic.
- *Mock LLM* never writes an answer. It says that no LLM is configured and shows the retrieved passages.
- The UI shows a banner while mock providers are active.

**Embeddings from different providers are never mixed.** Each chunk stores the model that produced it (`google:gemini-embedding-001:768`), and search only compares vectors from the current model. If you switch providers, use **Reprocess** on existing documents.

**Changing embedding dimensions** (default 768, chosen to reduce storage and search cost) requires a new migration that changes the `vector(768)` column, and re-embedding every document.

## API

Interactive docs: <http://localhost:8000/docs>

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/documents` | Upload one or more files (`multipart/form-data`, field `files`). Returns `202`; processing runs in the background |
| `GET` | `/api/v1/documents` | List documents with status and metadata |
| `GET` | `/api/v1/documents/{id}` | Document detail, including chunks and an embedding preview |
| `DELETE` | `/api/v1/documents/{id}` | Delete the document, its file, chunks and embeddings (`409` while processing) |
| `POST` | `/api/v1/documents/{id}/reprocess` | Re-run the pipeline (e.g. after a failure or provider change) |
| `POST` | `/api/v1/search` | `{query, top_k?, document_ids?}` returns the top-k chunks with cosine similarity scores |
| `POST` | `/api/v1/ask` | `{question, top_k?, document_ids?}` returns an answer, `insufficient_context`, and numbered `sources` with `cited` flags |
| `GET` | `/health` | Database status and active embedding/LLM models |

Errors: `400` invalid input, `404` not found, `409` conflicting state, `413` file too large, `415` unsupported type, `422` validation, `502` AI provider failure.

## How it works

**Processing status:** `UPLOADED → PROCESSING → COMPLETED | FAILED`

- Uploads are validated (extension, size, PDF magic bytes, UTF-8 for text), then stored under `backend/storage/uploads/<document-id>.<ext>`. The user's filename is never used as a path.
- A background task extracts text (PyMuPDF for PDFs, page by page), cleans it (Unicode normalization, whitespace, re-joining hyphenated line breaks), chunks it (default 1000 characters with 150 overlap, per page, so chunks keep their page numbers), and embeds the chunks in batches of 100.
- Chunks are written **in the same transaction** that marks the document `COMPLETED`. A failure marks it `FAILED` with a readable error and leaves no partial chunks. Reprocessing replaces chunks atomically.
- Documents still `UPLOADED`/`PROCESSING` when the server restarts are marked `FAILED` on startup, so you can reprocess them.
- **Ask:** the question is embedded with the `RETRIEVAL_QUERY` task type, the top-k chunks are retrieved by cosine distance (HNSW index), and the LLM is told to answer only from those numbered sources, cite them as `[n]`, ignore instructions inside documents, and reply `INSUFFICIENT_CONTEXT` when they don't contain the answer. The API turns that reply into a clear message and marks which sources were actually cited.

## Project layout

```
backend/
  app/
    api/routes/        HTTP layer (documents, search + ask)
    api/deps.py        dependency injection / provider selection
    core/              settings, logging
    db/, models/       SQLAlchemy engine + ORM models
    repositories/      all database access
    schemas/           Pydantic request/response models
    services/          extraction, chunking, embedding, LLM, ingestion, search, ask
  alembic/             migrations
  tests/
frontend/src/
  pages/               Dashboard, Documents, Ask
  api.ts               typed API client
sample-docs/           small test dataset (fictional company)
```

## Running tests

```bash
cd backend && source .venv/bin/activate
pytest
```

- Unit tests mock every external API, so the suite needs no API key and makes no network calls.
- Integration tests use a real PostgreSQL database: `enterprise_intelligence_test` by default, or set `TEST_DATABASE_URL`. They are skipped if it isn't reachable.
- **The test database is wiped on every run.** Never point `TEST_DATABASE_URL` at real data.

Frontend checks:

```bash
cd frontend && npm run build && npm run lint
```

## Known limitations (MVP)

- **No OCR.** Scanned or image-only PDFs fail with "No extractable text".
- **In-process background tasks.** A server restart interrupts processing; affected documents are marked `FAILED` for reprocessing. There is no queue or worker.
- **No retries or backoff** on Gemini rate limits (`429`) or timeouts. The document fails with the provider's status code; use Reprocess.
- **Plain character-based chunking.** Chunks can split mid-sentence; the overlap reduces the impact.
- **No authentication or multi-tenancy.** This is intended for local or trusted-network use only.
- **No pagination.** The document list returns the 500 most recent documents.
