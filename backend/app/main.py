from dotenv import load_dotenv

load_dotenv()

import logging  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.api.deps import get_embedding_service, get_llm_service  # noqa: E402
from app.api.routes.documents import router as documentRouter  # noqa: E402
from app.api.routes.search import router as searchRouter  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.repositories.document_repository import DocumentRepository  # noqa: E402
from app.services.embedding_service import EmbeddingConfigError, EmbeddingError  # noqa: E402
from app.services.llm_service import LLMError  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # background tasks don't survive restarts; fail them visibly instead of leaving them stuck
    with SessionLocal() as db:
        interrupted = DocumentRepository(db).fail_interrupted()
    if interrupted:
        logger.warning("Marked %s interrupted document(s) as FAILED", interrupted)
    yield


app = FastAPI(
    title="Enterprise Intelligence Platform",
    description="Evidence-driven enterprise AI platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documentRouter, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(searchRouter, prefix="/api/v1")


# provider errors carry client-safe messages; upstream failure -> 502
@app.exception_handler(EmbeddingError)
@app.exception_handler(LLMError)
async def provider_error_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(EmbeddingConfigError)
async def config_error_handler(request: Request, exc: EmbeddingConfigError):
    logger.error("Configuration error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Server is misconfigured"})


@app.get("/health")
async def health_check():
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "unavailable"

    try:
        embedding = get_embedding_service().model_id
    except Exception:
        embedding = "misconfigured"
    try:
        llm = get_llm_service()
        llm_model = f"{llm.provider}:{llm.model}"
    except Exception:
        llm_model = "misconfigured"

    return {
        "status": "healthy" if database == "ok" else "degraded",
        "service": "enterprise-intelligence-api",
        "database": database,
        "embedding_model": embedding,
        "llm_model": llm_model,
    }
