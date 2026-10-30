# RAG Chatbot

A document question answering API built on Retrieval-Augmented Generation. Upload documents, then ask questions about them in chat sessions. Answers are grounded in the documents and come with source citations (document name, pages, headings).

## Features

- **Document ingestion**: parses PDF, DOCX, PPTX, XLSX, HTML, Markdown and CSV with [Docling](https://github.com/docling-project/docling) and splits them into structure-aware chunks
- **Hybrid search**: dense embeddings (Qwen3-Embedding-0.6B) plus BM25 sparse vectors in Qdrant, fused with RRF
- **Reranking**: Qwen3-Reranker-0.6B cross-encoder reorders the retrieved chunks
- **Query router**: classifies each message (chitchat, standalone, follow-up, multi-question) and rewrites follow-ups into standalone queries
- **Chat history**: per-session history stored in Redis (last 8 turns, 1 hour TTL)
- **Semantic cache**: similar questions are answered from cache (cosine threshold 0.95, 7 day TTL); cleared whenever documents are added or removed
- **Local LLM**: Qwen3-8B-AWQ served by vLLM through an OpenAI-compatible API
- **Observability**: optional tracing with Langfuse

## Architecture

```
question ─► router ─► cache lookup ─► hybrid search (Qdrant) ─► rerank ─► LLM ─► answer + sources
               │                                                                      │
          chat history (Redis)                                                  save to cache/history
```

Services (see `docker-compose.yml`): `app` (FastAPI), `vllm`, `qdrant`, `redis`.


## Getting Started

1. Copy the example environment file and fill it in:

   ```bash
   cp .env.example .env
   ```

   | Variable | Description |
   |---|---|
   | `ADMIN_KEY` | Secret for admin endpoints (ingest, delete). Required. |
   | `BACKEND_API_KEY` | Secret for the chat endpoint. Required. |
   | `QDRANT_URL` | Qdrant address (default `http://qdrant:6333`) |
   | `REDIS_URL` | Redis address (default `redis://redis:6379/0`) |
   | `LLM_BASE_URL` | vLLM OpenAI-compatible endpoint (default `http://vllm:8000/v1`) |
   | `LANGFUSE_PUBLIC_KEY` | Langfuse public key (for tracing) |
   | `LANGFUSE_SECRET_KEY` | Langfuse secret key (for tracing) |

   For a self-hosted Langfuse or a non-default region, also set the host variable described in the Langfuse docs.

2. Start everything:

   ```bash
   docker compose up --build
   ```

   The first start downloads the models, so vLLM can take several minutes to become healthy. The app starts once vLLM is ready.

3. The API is now available at `http://localhost:8080`.

## API

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /ingest` (admin)
Upload a document. Header: `X-Admin-Key`.

```bash
curl -X POST http://localhost:8080/ingest \
  -H "X-Admin-Key: $ADMIN_KEY" \
  -F "file=@report.pdf"
```

Response: `{"filename": "report.pdf", "chunks": 42}`

Re-uploading a file with the same name replaces the existing version.

### `POST /chat`
Ask a question. Header: `X-API-Key`. Use the same `session_id` to keep conversation context.

```bash
curl -X POST http://localhost:8080/chat \
  -H "X-API-Key: $BACKEND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "session_id": "user-123"}'
```

Response:

```json
{
  "answer": "...",
  "sources": [{"doc_name": "report.pdf", "pages": [3, 4], "headings": ["Results"]}]
}
```

### `DELETE /documents/{doc_name}` (admin)
Remove a document and all its chunks. Header: `X-Admin-Key`. URL-encode filenames that contain spaces or special characters.

```bash
curl -X DELETE http://localhost:8080/documents/report.pdf \
  -H "X-Admin-Key: $ADMIN_KEY"
```

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, auth and endpoints |
| `rag.py` | Orchestrates ingestion and question answering |
| `parser.py` | Document parsing |
| `chunker.py` | Hybrid chunking sized for the embedding model's tokenizer |
| `embedder.py` | Embeddings |
| `vector_store.py` | Qdrant collection with dense + sparse hybrid search |
| `reranker.py` | reranking |
| `router.py` | Query classification and rewriting |
| `generator.py` | LLM client (vLLM via OpenAI API) |
| `semantic_cache.py` | Semantic answer cache in Qdrant |
| `history.py` | Chat history in Redis |
| `prompts.py` | System prompts |
| `helper.py` | Prompt building and structured-output schema |
| `errors.py` | Custom exceptions |

## Notes

- Information requests are answered only from the retrieved documents. If the documents don't contain the answer, the bot says so. Greetings and questions about the bot itself are handled without retrieval.
- Reranker, chat history and cache failures degrade gracefully (for example, the app falls back to the unreranked results if the reranker fails). A vector store failure during search returns an error.
- Models are cached in the `hf_cache` volume. Vector data and chat history persist in the `qdrant_data` and `redis_data` volumes.
