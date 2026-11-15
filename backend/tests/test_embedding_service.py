import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.genai import errors

from app.services.embedding_service import (
    DEFAULT_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TASK_TYPE,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingService,
)

FAKE_KEY = "fake-secret-key-123"
ENV_VARS = (
    "GEMINI_API_KEY",
    "GEMINI_EMBEDDING_MODEL",
    "GEMINI_EMBEDDING_TASK_TYPE",
    "GEMINI_EMBEDDING_DIMENSIONS",
    "GEMINI_EMBEDDING_BATCH_SIZE",
    "GEMINI_TIMEOUT_SECONDS",
)
TEST_DIMS = 3


def response(values):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


@pytest.fixture
def env(monkeypatch):
    # start from a clean slate so the real .env never leaks into tests
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSIONS", str(TEST_DIMS))
    return monkeypatch


@pytest.fixture
def mock_client_cls():
    with patch("app.services.embedding_service.genai.Client") as client_cls:
        client_cls.return_value.aio.models.embed_content = AsyncMock(
            return_value=response([0.1, 0.2, 0.3])
        )
        yield client_cls


@pytest.fixture
def service(env, mock_client_cls):
    return EmbeddingService()


def embed_mock(svc):
    return svc.client.aio.models.embed_content


def run(coro):
    return asyncio.run(coro)


# --- configuration ---

def test_client_initialized_from_env(service, mock_client_cls):
    mock_client_cls.assert_called_once()
    kwargs = mock_client_cls.call_args.kwargs
    assert kwargs["api_key"] == FAKE_KEY
    assert kwargs["http_options"].timeout == 60_000


def test_defaults_when_optional_env_missing(env, mock_client_cls):
    env.delenv("GEMINI_EMBEDDING_DIMENSIONS")
    svc = EmbeddingService()
    assert svc.model == DEFAULT_EMBEDDING_MODEL
    assert svc.task_type == DEFAULT_TASK_TYPE
    assert svc.dimensions == DEFAULT_DIMENSIONS == 768


def test_dimensions_from_env(env, mock_client_cls):
    env.setenv("GEMINI_EMBEDDING_DIMENSIONS", "1536")
    assert EmbeddingService().dimensions == 1536


@pytest.mark.parametrize("value", ["abc", "0", "-5", "7.5"])
def test_invalid_dimensions_raise(env, mock_client_cls, value):
    env.setenv("GEMINI_EMBEDDING_DIMENSIONS", value)
    with pytest.raises(EmbeddingConfigError, match="GEMINI_EMBEDDING_DIMENSIONS"):
        EmbeddingService()


def test_model_and_task_type_from_env(env, mock_client_cls):
    env.setenv("GEMINI_EMBEDDING_MODEL", "custom-model")
    env.setenv("GEMINI_EMBEDDING_TASK_TYPE", "retrieval_query")
    svc = EmbeddingService()
    assert svc.model == "custom-model"
    assert svc.task_type == "RETRIEVAL_QUERY"


def test_blank_model_falls_back_to_default(env, mock_client_cls):
    env.setenv("GEMINI_EMBEDDING_MODEL", "   ")
    assert EmbeddingService().model == DEFAULT_EMBEDDING_MODEL


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_missing_api_key_raises(env, mock_client_cls, value):
    if value is None:
        env.delenv("GEMINI_API_KEY")
    else:
        env.setenv("GEMINI_API_KEY", value)
    with pytest.raises(EmbeddingConfigError, match="GEMINI_API_KEY"):
        EmbeddingService()
    mock_client_cls.assert_not_called()


def test_invalid_task_type_in_env_raises(env, mock_client_cls):
    env.setenv("GEMINI_EMBEDDING_TASK_TYPE", "NOT_A_TASK")
    with pytest.raises(EmbeddingConfigError, match="Invalid embedding task type") as exc_info:
        EmbeddingService()
    assert "NOT_A_TASK" not in str(exc_info.value)


def test_client_init_failure_hides_secret(env):
    with patch(
        "app.services.embedding_service.genai.Client",
        side_effect=ValueError(f"bad key {FAKE_KEY}"),
    ):
        with pytest.raises(EmbeddingConfigError) as exc_info:
            EmbeddingService()
    assert FAKE_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


# --- request ---

def test_returns_list_of_floats(service):
    result = run(service.embed_text("hello"))

    assert result == [0.1, 0.2, 0.3]
    assert all(type(v) is float for v in result)

    kwargs = embed_mock(service).await_args.kwargs
    assert kwargs["model"] == DEFAULT_EMBEDDING_MODEL
    assert kwargs["contents"] == ["hello"]
    assert kwargs["config"].task_type == DEFAULT_TASK_TYPE
    assert kwargs["config"].output_dimensionality == TEST_DIMS


def test_int_values_converted_to_float(service):
    embed_mock(service).return_value = response([1, 0, -1])
    result = run(service.embed_text("hello"))
    assert result == [1.0, 0.0, -1.0]
    assert all(type(v) is float for v in result)


def test_task_type_override_per_call(service):
    run(service.embed_text("what is rag?", task_type="retrieval_query"))
    assert embed_mock(service).await_args.kwargs["config"].task_type == "RETRIEVAL_QUERY"


@pytest.mark.parametrize("task_type", ["BOGUS", "", "   ", 123, ["RETRIEVAL_QUERY"]])
def test_invalid_task_type_per_call_raises(service, task_type):
    with pytest.raises(EmbeddingConfigError, match="Invalid embedding task type") as exc_info:
        run(service.embed_text("hello", task_type=task_type))
    assert "BOGUS" not in str(exc_info.value)
    embed_mock(service).assert_not_awaited()


@pytest.mark.parametrize("text", ["", "   \n\t", None, 123])
def test_empty_or_non_string_input_rejected(service, text):
    with pytest.raises(ValueError):
        run(service.embed_text(text))
    embed_mock(service).assert_not_awaited()


# --- provider errors ---

def test_api_error_hides_secret(service):
    embed_mock(service).side_effect = errors.ClientError(
        400, {"error": {"message": f"API key {FAKE_KEY} not valid"}}
    )
    with pytest.raises(EmbeddingError) as exc_info:
        run(service.embed_text("hello"))

    assert FAKE_KEY not in str(exc_info.value)
    assert "400" in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_network_error_wrapped(service):
    embed_mock(service).side_effect = httpx.ConnectError("boom")
    with pytest.raises(EmbeddingError, match="unreachable"):
        run(service.embed_text("hello"))


def test_unexpected_sdk_exception_wrapped(service):
    embed_mock(service).side_effect = KeyError(f"internal {FAKE_KEY}")
    with pytest.raises(EmbeddingError, match="Unexpected") as exc_info:
        run(service.embed_text("hello"))

    assert FAKE_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


# --- response validation ---

@pytest.mark.parametrize(
    "bad_response",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(embeddings=None),
        SimpleNamespace(embeddings=[]),
        SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1]), SimpleNamespace(values=[0.2])]),
        SimpleNamespace(embeddings=[SimpleNamespace(values=None)]),
        response([]),
    ],
    ids=["none", "no-attr", "embeddings-none", "embeddings-empty", "two-embeddings",
         "values-none", "values-empty"],
)
def test_malformed_response_rejected(service, bad_response):
    embed_mock(service).return_value = bad_response
    with pytest.raises(EmbeddingError, match="Invalid embedding response"):
        run(service.embed_text("hello"))


@pytest.mark.parametrize(
    "values",
    [[0.1, float("nan"), 0.3], [float("inf"), 0.2, 0.3], [0.1, 0.2, float("-inf")]],
    ids=["nan", "inf", "-inf"],
)
def test_non_finite_values_rejected(service, values):
    embed_mock(service).return_value = response(values)
    with pytest.raises(EmbeddingError, match="non-finite"):
        run(service.embed_text("hello"))


@pytest.mark.parametrize("values", [[0.1, "0.2", 0.3], [0.1, None, 0.3], [True, 0.2, 0.3]],
                         ids=["string", "none", "bool"])
def test_non_numeric_values_rejected(service, values):
    embed_mock(service).return_value = response(values)
    with pytest.raises(EmbeddingError, match="non-numeric"):
        run(service.embed_text("hello"))


@pytest.mark.parametrize("values", [[0.1], [0.1, 0.2], [0.1, 0.2, 0.3, 0.4]],
                         ids=["1-dim", "2-dim", "4-dim"])
def test_wrong_dimension_rejected(service, values):
    embed_mock(service).return_value = response(values)
    with pytest.raises(EmbeddingError, match=f"expected {TEST_DIMS} dimensions"):
        run(service.embed_text("hello"))


# --- batch ---

def batch_response(*vectors):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])


def test_batch_single_request_preserves_order(service):
    embed_mock(service).return_value = batch_response(
        [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]
    )

    result = run(service.embed_texts(["a", "b", "c"]))

    assert result == [[1.0] * 3, [2.0] * 3, [3.0] * 3]
    embed_mock(service).assert_awaited_once()
    assert embed_mock(service).await_args.kwargs["contents"] == ["a", "b", "c"]


@pytest.mark.parametrize("returned", [1, 3], ids=["fewer", "more"])
def test_batch_count_mismatch_rejected(service, returned):
    embed_mock(service).return_value = batch_response(*[[0.1, 0.2, 0.3]] * returned)
    with pytest.raises(EmbeddingError, match="expected 2 embeddings"):
        run(service.embed_texts(["a", "b"]))


@pytest.mark.parametrize(
    "bad_vector",
    [[0.1, 0.2], [0.1, float("nan"), 0.3], [0.1, "x", 0.3]],
    ids=["wrong-dim", "nan", "non-numeric"],
)
def test_batch_any_bad_vector_rejects_whole_batch(service, bad_vector):
    embed_mock(service).return_value = batch_response([0.1, 0.2, 0.3], bad_vector)
    with pytest.raises(EmbeddingError, match="Invalid embedding response"):
        run(service.embed_texts(["a", "b"]))


@pytest.mark.parametrize(
    "texts", [[], None, "hello", ["ok", ""], ["ok", "   "], ["ok", None]],
    ids=["empty-list", "none", "str-not-list", "one-empty", "one-blank", "one-none"],
)
def test_batch_invalid_input_rejected(service, texts):
    with pytest.raises(ValueError):
        run(service.embed_texts(texts))
    embed_mock(service).assert_not_awaited()


def test_timeout_maps_to_safe_error(service):
    embed_mock(service).side_effect = httpx.ReadTimeout("slow")
    with pytest.raises(EmbeddingError, match="timed out"):
        run(service.embed_text("hello"))


def test_batches_split_by_batch_size(env, mock_client_cls):
    env.setenv("GEMINI_EMBEDDING_BATCH_SIZE", "2")
    svc = EmbeddingService()
    embed_mock(svc).side_effect = lambda **kw: batch_response(
        *[[float(len(t))] * TEST_DIMS for t in kw["contents"]]
    )

    result = run(svc.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"]))

    assert [call.kwargs["contents"] for call in embed_mock(svc).await_args_list] == [
        ["a", "bb"], ["ccc", "dddd"], ["eeeee"]
    ]
    assert [v[0] for v in result] == [1.0, 2.0, 3.0, 4.0, 5.0]
