import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.genai import errors

from app.services.llm_service import GeminiLLMService, LLMError, MockLLMService

FAKE_KEY = "fake-secret-key-123"


@pytest.fixture
def llm():
    with patch("app.services.llm_service.genai.Client"):
        svc = GeminiLLMService(FAKE_KEY, "test-model", timeout_seconds=5)
        svc.client.aio.models.generate_content = AsyncMock(
            return_value=SimpleNamespace(text="  The answer [1].  ")
        )
        yield svc


def run(coro):
    return asyncio.run(coro)


def test_generate_returns_stripped_text(llm):
    assert run(llm.generate("system", "prompt")) == "The answer [1]."
    kwargs = llm.client.aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["config"].system_instruction == "system"


def test_missing_key_rejected():
    with pytest.raises(LLMError):
        GeminiLLMService("", "m")


@pytest.mark.parametrize(
    "exc, message",
    [
        (errors.ClientError(429, {"error": {"message": f"quota for {FAKE_KEY}"}}), "status 429"),
        (httpx.ReadTimeout("slow"), "timed out"),
        (httpx.ConnectError("down"), "unreachable"),
        (RuntimeError(f"boom {FAKE_KEY}"), "Unexpected"),
    ],
)
def test_errors_are_safe(llm, exc, message):
    llm.client.aio.models.generate_content.side_effect = exc
    with pytest.raises(LLMError, match=message) as exc_info:
        run(llm.generate("s", "p"))
    assert FAKE_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_response_rejected(llm, text):
    llm.client.aio.models.generate_content.return_value = SimpleNamespace(text=text)
    with pytest.raises(LLMError, match="empty"):
        run(llm.generate("s", "p"))


def test_mock_llm_does_not_fabricate():
    answer = run(MockLLMService().generate("s", "Question: what is X?"))
    assert "no answer was generated" in answer
