import asyncio
from unittest.mock import AsyncMock

import pytest

from app.schemas.search import SearchResult
from app.services.ask_service import (
    INSUFFICIENT_ANSWER,
    AskService,
    build_context,
    cited_numbers,
)
from app.services.llm_service import LLMError


def result(n, filename="policy.pdf", page=1):
    return SearchResult(
        chunk_id=f"c{n}", document_id="d1", filename=filename, chunk_index=n,
        page_number=page, content=f"passage {n}", score=0.9 - n * 0.1,
    )


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self, answer):
        self.generate = AsyncMock(return_value=answer) if not isinstance(answer, Exception) \
            else AsyncMock(side_effect=answer)


def make(results, answer):
    search = AsyncMock()
    search.search = AsyncMock(return_value=results)
    llm = FakeLLM(answer)
    return AskService(search, llm), llm


def ask(service, question="What is the policy?"):
    return asyncio.run(service.ask(question))


def test_answer_with_citations_marks_cited_sources():
    service, llm = make([result(0), result(1), result(2)], "Staff get 25 days [1] and carry over 5 [3].")
    data = ask(service)

    assert data.insufficient_context is False
    assert data.answer == "Staff get 25 days [1] and carry over 5 [3]."
    assert [(s.source_number, s.cited) for s in data.sources] == [(1, True), (2, False), (3, True)]
    llm.generate.assert_awaited_once()


def test_insufficient_token_replaced_with_clear_message():
    service, _ = make([result(0)], "INSUFFICIENT_CONTEXT")
    data = ask(service)

    assert data.insufficient_context is True
    assert data.answer == INSUFFICIENT_ANSWER
    assert len(data.sources) == 1  # still inspectable
    assert not data.sources[0].cited


def test_no_results_skips_llm():
    service, llm = make([], "should not be used")
    data = ask(service)

    assert data.insufficient_context is True
    assert data.sources == []
    llm.generate.assert_not_awaited()


def test_llm_error_propagates():
    service, _ = make([result(0)], LLMError("LLM provider error (status 429)"))
    with pytest.raises(LLMError, match="429"):
        ask(service)


def test_prompt_contains_numbered_sources_and_question():
    service, llm = make([result(0, "a.pdf", 4), result(1, "b.md", None)], "x [1]")
    ask(service, "Who approves leave?")

    system, prompt = llm.generate.await_args.args
    assert "ONLY the numbered sources" in system
    assert "[1] a.pdf (page 4)" in prompt
    assert "[2] b.md (chunk 1)" in prompt
    assert prompt.endswith("Question: Who approves leave?")


@pytest.mark.parametrize(
    "answer, expected",
    [("a [1] b [2][3]", {1, 2, 3}), ("see [1, 4]", {1, 4}), ("no refs", set()), ("[x] [10]", {10})],
)
def test_cited_numbers(answer, expected):
    assert cited_numbers(answer) == expected


def test_build_context_separates_sources():
    ctx = build_context([result(0), result(1)])
    assert ctx.count("---") == 1
    assert ctx.startswith("[1] policy.pdf (page 1)\npassage 0")
