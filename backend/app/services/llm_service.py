import logging

import httpx
from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM provider fails. Message is safe to show to clients."""


class GeminiLLMService:
    provider = "google"

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 60):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000)
        )

    async def generate(self, system: str, prompt: str) -> str:
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.1),
            )
        except errors.APIError as e:
            logger.error("LLM request failed status=%s", e.code)
            raise LLMError(f"LLM provider error (status {e.code})") from None
        except httpx.TimeoutException:
            logger.error("LLM request timed out")
            raise LLMError("LLM provider timed out") from None
        except httpx.HTTPError as e:
            logger.error("LLM request failed error=%s", type(e).__name__)
            raise LLMError("LLM provider unreachable") from None
        except Exception as e:
            logger.error("Unexpected LLM SDK failure error=%s", type(e).__name__)
            raise LLMError("Unexpected LLM failure") from None

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            # e.g. blocked by safety filters or empty candidate
            raise LLMError("LLM returned an empty response")
        return text


class MockLLMService:
    """Used when no LLM credentials are configured. Never fabricates an answer."""

    provider = "mock"
    model = "none"

    async def generate(self, system: str, prompt: str) -> str:
        return (
            "No LLM provider is configured (LLM_PROVIDER=mock), so no answer was generated. "
            "The retrieved passages below are the most relevant matches for your question."
        )
