from rag import RAG
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, Request, Depends
from pydantic import BaseModel, Field, field_validator
from errors import ParseError, ChunkError, StoreError
import os
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
import secrets
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".csv"}
ADMIN_KEY = os.environ["ADMIN_KEY"]
BACKEND_API_KEY = os.environ["BACKEND_API_KEY"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAG()
    yield
    try:
        app.state.rag.store.client.close()
    except Exception:
        logger.warning("Failed to close Qdrant client", exc_info=True)

app = FastAPI(title="RAG Chatbot", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def require_api_key(x_api_key: str | None = Header(default=None)):
    if not x_api_key or not secrets.compare_digest(x_api_key, BACKEND_API_KEY):
        raise HTTPException(403, "Not allowed")

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=200)

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v

class Source(BaseModel):
    doc_name: str
    pages: list[int]
    headings: list[str]

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(body: ChatRequest, request: Request):
    return request.app.state.rag.ask(body.session_id, body.query)


def require_admin_key(x_admin_key: str | None = Header(default=None)):
    if not x_admin_key or not secrets.compare_digest(x_admin_key, ADMIN_KEY):
        raise HTTPException(403, "Not allowed")

class IngestResponse(BaseModel):
    filename: str
    chunks: int

@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin_key)])
def ingest(request: Request, file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(file.file.read())

        try:
            n = request.app.state.rag.ingest(str(path))
        except (ParseError, ChunkError):
            logger.warning(f"Ingest failed for {name}", exc_info=True)
            raise HTTPException(422, f"Could not process '{name}'.")
        except StoreError:
            logger.error(f"Store failed for {name}", exc_info=True)
            raise HTTPException(500, "Failed to store the document.")

    return {"filename": name, "chunks": n}


@app.delete("/documents/{doc_name}", dependencies=[Depends(require_admin_key)])
def delete_document(doc_name: str, request: Request):
    try:
        n = request.app.state.rag.remove_doc(doc_name)
    except StoreError:
        logger.error("Delete failed for %s", doc_name, exc_info=True)
        raise HTTPException(500, "Failed to delete the document.")

    if n == 0:
        raise HTTPException(404, f"No document named '{doc_name}'.")

    return {"doc_name": doc_name, "deleted_chunks": n}