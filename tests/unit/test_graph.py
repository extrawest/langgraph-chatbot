"""The graph as a whole: real LangGraph, real MessagesStore (Qdrant in memory),
scripted LLM, fake Mem0."""

import uuid

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from langgraph_chatbot.ai.context import ContextSchema
from langgraph_chatbot.ai.graph import (
    TOOL_OUTPUT_LIMIT,
    _custom_retry_on,
    build_graph,
    trim_tool_output,
)

_REQUEST = httpx.Request("POST", "https://llm.test")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST)


# --------------------------------------------------------- trim_tool_output
async def test_trim_tool_output_truncates_long_results_keeping_id():
    long = "x" * (TOOL_OUTPUT_LIMIT + 500)
    state = {
        "messages": [
            HumanMessage("q"),
            AIMessage("", tool_calls=[{"name": "web", "args": {}, "id": "c1"}]),
            ToolMessage(long, tool_call_id="c1", name="web", id="tool-msg-1"),
        ]
    }

    (trimmed,) = (await trim_tool_output(state))["messages"]

    assert trimmed.id == "tool-msg-1"  # same id -> replaces the original in state
    assert trimmed.tool_call_id == "c1"
    assert trimmed.name == "web"
    assert trimmed.content.startswith("x" * TOOL_OUTPUT_LIMIT)
    assert f"[truncated: {len(long)} chars total, first {TOOL_OUTPUT_LIMIT} shown]" in trimmed.content


async def test_trim_tool_output_leaves_short_results_alone():
    state = {"messages": [HumanMessage("q"), ToolMessage("short", tool_call_id="c1", id="t")]}
    assert await trim_tool_output(state) == {"messages": []}


async def test_trim_tool_output_only_looks_at_the_latest_tool_batch():
    long = "y" * (TOOL_OUTPUT_LIMIT + 1)
    state = {
        "messages": [
            ToolMessage(long, tool_call_id="old", id="old"),  # an earlier batch
            AIMessage("", tool_calls=[{"name": "a", "args": {}, "id": "c1"}]),
            ToolMessage(long, tool_call_id="c1", id="new-1"),
            ToolMessage("ok", tool_call_id="c2", id="new-2"),
        ]
    }

    trimmed = (await trim_tool_output(state))["messages"]

    assert [m.id for m in trimmed] == ["new-1"]


# --------------------------------------------------------------- retry_on
@pytest.mark.parametrize(
    ("exc", "retry"),
    [
        (UnexpectedResponse(500, "err", b"", httpx.Headers()), True),
        (UnexpectedResponse(503, "err", b"", httpx.Headers()), True),
        (UnexpectedResponse(404, "nf", b"", httpx.Headers()), False),
        (ResponseHandlingException(ConnectionError()), True),
        (openai.APIConnectionError(request=_REQUEST), True),
        (openai.APITimeoutError(request=_REQUEST), True),
        (openai.RateLimitError("slow down", response=_response(429), body=None), True),
        (openai.InternalServerError("oops", response=_response(500), body=None), True),
        (openai.BadRequestError("bad", response=_response(400), body=None), False),
        (ValueError("bug"), False),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v),
)
def test_custom_retry_on(exc, retry):
    assert _custom_retry_on(exc) is retry


# -------------------------------------------------------------- full graph
@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"result for {query}"


@tool
def huge_page(url: str) -> str:
    """Return a very large page."""
    return "z" * (TOOL_OUTPUT_LIMIT * 2)


def _tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def ctx():
    return ContextSchema(user_id=uuid.uuid4(), chat_id=uuid.uuid4())


async def _ask(graph, ctx, text):
    return await graph.ainvoke({"messages": [HumanMessage(text)]}, context=ctx)


async def test_graph_answers_and_persists_the_turn(scripted_llm, messages_store, user_memory, ctx):
    llm = scripted_llm(AIMessage("Hi! How can I help?"))
    graph = build_graph(llm, messages_store, user_memory, tools=[])

    state = await _ask(graph, ctx, "hello")

    assert state["messages"][-1].content == "Hi! How can I help?"
    stored = await messages_store.recent(user_id=ctx.user_id, chat_id=ctx.chat_id)
    assert [(m.role, m.content) for m in stored] == [
        ("user", "hello"),
        ("assistant", "Hi! How can I help?"),
    ]
    assert user_memory.remembered[0]["user_message"] == "hello"
    assert user_memory.remembered[0]["assistant_message"] == "Hi! How can I help?"


async def test_graph_feeds_stored_history_and_memories_to_the_model(
    scripted_llm, messages_store, user_memory, ctx
):
    user_memory.memories = ["prefers short answers"]
    llm = scripted_llm(AIMessage("first answer"), AIMessage("second answer"))
    graph = build_graph(llm, messages_store, user_memory, tools=[])

    await _ask(graph, ctx, "first question")
    await _ask(graph, ctx, "second question")

    second_prompt = llm.calls[1]
    assert "- prefers short answers" in second_prompt[0].content
    assert [(m.type, m.content) for m in second_prompt[1:]] == [
        ("human", "first question"),
        ("ai", "first answer"),
        ("human", "second question"),
    ]


async def test_graph_history_is_per_chat(scripted_llm, messages_store, user_memory, ctx):
    llm = scripted_llm(AIMessage("a1"), AIMessage("a2"))
    graph = build_graph(llm, messages_store, user_memory, tools=[])
    other_chat = ContextSchema(user_id=ctx.user_id, chat_id=uuid.uuid4())

    await _ask(graph, ctx, "in chat one")
    await _ask(graph, other_chat, "in chat two")

    assert [m.type for m in llm.calls[1]] == ["system", "human"]


async def test_graph_runs_tools_and_loops_back_to_the_agent(
    scripted_llm, messages_store, user_memory, ctx
):
    llm = scripted_llm(_tool_call("lookup", {"query": "weather"}), AIMessage("It is sunny."))
    graph = build_graph(llm, messages_store, user_memory, tools=[lookup])

    state = await _ask(graph, ctx, "weather?")

    assert [t.name for t in llm.bound_tools] == ["lookup"]
    assert state["messages"][-1].content == "It is sunny."
    tool_result = llm.calls[1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.content == "result for weather"
    # only the final answer is stored, not the tool traffic
    stored = await messages_store.recent(user_id=ctx.user_id, chat_id=ctx.chat_id)
    assert [m.content for m in stored] == ["weather?", "It is sunny."]


async def test_graph_trims_huge_tool_output_before_the_model_sees_it(
    scripted_llm, messages_store, user_memory, ctx
):
    llm = scripted_llm(_tool_call("huge_page", {"url": "https://x"}), AIMessage("done"))
    graph = build_graph(llm, messages_store, user_memory, tools=[huge_page])

    await _ask(graph, ctx, "read it")

    seen = llm.calls[1][-1].content
    assert len(seen) < TOOL_OUTPUT_LIMIT + 100
    assert "[truncated:" in seen


async def test_graph_tool_errors_are_reported_to_the_model(
    scripted_llm, messages_store, user_memory, ctx
):
    llm = scripted_llm(_tool_call("lookup", {"wrong_arg": 1}), AIMessage("sorry"))
    graph = build_graph(llm, messages_store, user_memory, tools=[lookup])

    state = await _ask(graph, ctx, "q")

    assert state["messages"][-1].content == "sorry"
    assert llm.calls[1][-1].status == "error"


async def test_graph_retries_transient_llm_errors(scripted_llm, messages_store, user_memory, ctx):
    llm = scripted_llm(openai.APIConnectionError(request=_REQUEST), AIMessage("recovered"))
    graph = build_graph(llm, messages_store, user_memory, tools=[])

    state = await _ask(graph, ctx, "q")

    assert state["messages"][-1].content == "recovered"
    assert len(llm.calls) == 2


async def test_graph_does_not_retry_programming_errors(scripted_llm, messages_store, user_memory, ctx):
    llm = scripted_llm(ValueError("bug"), AIMessage("never"))
    graph = build_graph(llm, messages_store, user_memory, tools=[])

    with pytest.raises(ValueError, match="bug"):
        await _ask(graph, ctx, "q")
    assert len(llm.calls) == 1
