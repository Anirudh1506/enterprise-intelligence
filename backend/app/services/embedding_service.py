import logging
import math
import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)

# loads backend/.env if present; never overrides variables already set in the environment
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
# chosen to cut storage/retrieval cost, not because it is more accurate than 3072
DEFAULT_DIMENSIONS = 768
VALID_TASK_TYPES = {
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
}


class EmbeddingConfigError(RuntimeError):
    """Raised when configuration is missing or invalid. Never includes secret values."""


class EmbeddingError(Exception):
    """Raised when the embedding provider fails. Message is safe to show to clients."""


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or "").strip() or default


def _validate_task_type(task_type) -> str:
    normalized = task_type.strip().upper() if type(task_type) is str else None
    if normalized not in VALID_TASK_TYPES:
        # deliberately does not echo the rejected value
        raise EmbeddingConfigError(
            f"Invalid embedding task type. Allowed: {', '.join(sorted(VALID_TASK_TYPES))}"
        )
    return normalized


def _parse_dimensions(raw: str) -> int:
    if not raw.isdigit() or int(raw) <= 0:
        raise EmbeddingConfigError(
            "Invalid GEMINI_EMBEDDING_DIMENSIONS. Must be a positive integer."
        )
    return int(raw)


class EmbeddingService:

    provider = "google"

    def __init__(self):
        api_key = _env("GEMINI_API_KEY")
        if not api_key:
            raise EmbeddingConfigError(
                "Missing required environment variable: GEMINI_API_KEY. "
                "Set it in backend/.env or the shell environment."
            )

        self.model = _env("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.task_type = _validate_task_type(
            _env("GEMINI_EMBEDDING_TASK_TYPE", DEFAULT_TASK_TYPE)
        )
        self.dimensions = _parse_dimensions(
            _env("GEMINI_EMBEDDING_DIMENSIONS", str(DEFAULT_DIMENSIONS))
        )

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise EmbeddingConfigError(
                f"Failed to initialize Gemini client ({type(e).__name__})"
            ) from None

    async def embed_text(self, text: str, task_type: Optional[str] = None) -> list[float]:
        return (await self.embed_texts([text], task_type))[0]

    async def embed_texts(
        self, texts: list[str], task_type: Optional[str] = None
    ) -> list[list[float]]:
        """Embed many texts in one request. Output order matches input order."""
        if type(texts) is not list or not texts:
            raise ValueError("Texts to embed must be a non-empty list")
        if any(type(t) is not str or not t.strip() for t in texts):
            raise ValueError("Text to embed must not be empty")

        task = self.task_type if task_type is None else _validate_task_type(task_type)

        # ponytail: single request for all texts; Gemini caps batch size (~100),
        # add configurable batching + retries in the reliability milestone
        try:
            result = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task,
                    output_dimensionality=self.dimensions,
                ),
            )
        except errors.APIError as e:
            # log only the status code; provider messages can echo request details
            logger.error("Embedding request failed with status %s", e.code)
            raise EmbeddingError(f"Embedding provider error (status {e.code})") from None
        except httpx.HTTPError as e:
            logger.error("Embedding request failed: %s", type(e).__name__)
            raise EmbeddingError("Embedding provider unreachable") from None
        except Exception as e:
            logger.error("Unexpected embedding SDK failure: %s", type(e).__name__)
            raise EmbeddingError("Unexpected embedding failure") from None

        return _extract_vectors(result, len(texts), self.dimensions)


def _extract_vectors(result, expected_count: int, dimensions: int) -> list[list[float]]:
    embeddings = getattr(result, "embeddings", None)
    if not embeddings or len(embeddings) != expected_count:
        got = len(embeddings) if embeddings else 0
        raise EmbeddingError(
            f"Invalid embedding response: expected {expected_count} embeddings, got {got}"
        )
    return [_validate_vector(getattr(e, "values", None), dimensions) for e in embeddings]


def _validate_vector(values, dimensions: int) -> list[float]:
    if not values:
        raise EmbeddingError("Invalid embedding response: empty vector")
    if len(values) != dimensions:
        raise EmbeddingError(
            f"Invalid embedding response: expected {dimensions} dimensions, got {len(values)}"
        )

    vector = []
    for v in values:
        # bool is a subclass of int, so exclude it explicitly
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise EmbeddingError("Invalid embedding response: non-numeric value")
        if not math.isfinite(v):
            raise EmbeddingError("Invalid embedding response: non-finite value")
        vector.append(float(v))

    return vector
