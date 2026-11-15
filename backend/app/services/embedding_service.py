import hashlib
import logging
import math
import os
import re
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
# Gemini rejects batches above 100 texts per request
DEFAULT_BATCH_SIZE = 100
DEFAULT_TIMEOUT_SECONDS = 60
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


def _parse_positive_int(raw: str, name: str) -> int:
    if not raw.isdigit() or int(raw) <= 0:
        raise EmbeddingConfigError(f"Invalid {name}. Must be a positive integer.")
    return int(raw)


def _parse_dimensions(raw: str) -> int:
    return _parse_positive_int(raw, "GEMINI_EMBEDDING_DIMENSIONS")


def _validate_texts(texts) -> None:
    if type(texts) is not list or not texts:
        raise ValueError("Texts to embed must be a non-empty list")
    if any(type(t) is not str or not t.strip() for t in texts):
        raise ValueError("Text to embed must not be empty")


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
        self.batch_size = _parse_positive_int(
            _env("GEMINI_EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)),
            "GEMINI_EMBEDDING_BATCH_SIZE",
        )

        try:
            timeout = _parse_positive_int(
                _env("GEMINI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)), "GEMINI_TIMEOUT_SECONDS"
            )
            self.client = genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=timeout * 1000)
            )
        except Exception as e:
            raise EmbeddingConfigError(
                f"Failed to initialize Gemini client ({type(e).__name__})"
            ) from None

    async def embed_text(self, text: str, task_type: Optional[str] = None) -> list[float]:
        return (await self.embed_texts([text], task_type))[0]

    async def embed_texts(
        self, texts: list[str], task_type: Optional[str] = None
    ) -> list[list[float]]:
        """Embed texts in batches of `batch_size`. Output order matches input order."""
        _validate_texts(texts)
        task = self.task_type if task_type is None else _validate_task_type(task_type)

        # ponytail: sequential batches, no retry/backoff; add when rate limits bite
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(await self._embed_batch(texts[i:i + self.batch_size], task))
        return vectors

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}"

    async def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
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
        except httpx.TimeoutException:
            logger.error("Embedding request timed out")
            raise EmbeddingError("Embedding provider timed out") from None
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


class MockEmbeddingService:
    """Local-dev stand-in when no API key is configured.

    Hashed bag-of-words vectors: deterministic and lexical (shared words score
    higher), NOT semantic. Search quality is clearly worse than a real model.
    """

    provider = "mock"
    model = "hashed-bow-v1"

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS):
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}"

    async def embed_text(self, text: str, task_type: Optional[str] = None) -> list[float]:
        return (await self.embed_texts([text], task_type))[0]

    async def embed_texts(
        self, texts: list[str], task_type: Optional[str] = None
    ) -> list[list[float]]:
        _validate_texts(texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
            vec[h % self.dimensions] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            # text with no word characters; avoid a zero vector (cosine undefined)
            vec[0], norm = 1.0, 1.0
        return [v / norm for v in vec]
