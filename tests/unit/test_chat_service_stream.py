"""ChatService.stream_reply: graph stream -> UI events. Uses a fake graph."""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from langgraph_chatbot.ai.context import ContextSchema, Mode
from langgraph_chatbot.chat.events import (
    ChatEvent,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    ToolEvent,
)
from langgraph_chatbot.chat.service import ChatService, _user_message
from langgraph_chatbot.logger import log_context
from tests.helpers import FakeGraph

USER_ID = uuid.uuid4()
CHAT = SimpleNamespace(id=uuid.uuid4(), title="My chat")


async def collect(graph, message="hi"):
    service = ChatService(MagicMock(), MagicMock(), graph, MagicMock())
    return [event async for event in service.stream_reply(CHAT, USER_ID, message)]


async def test_streams_tokens_between_chat_and_done():
    graph = FakeGraph([(AIMessageChunk("Hel"), "agent"), (AIMessageChunk("lo"), "agent")])

    events = await collect(graph)

    assert events == [
        ChatEvent(chat_id=CHAT.id, title="My chat"),
        TokenEvent(text="Hel"),
        TokenEvent(text="lo"),
        DoneEvent(),
    ]


async def test_calls_graph_with_question_and_context():
    graph = FakeGraph()

    await collect(graph, "what's up?")

    (call,) = graph.calls
    (message,) = call["input"]["messages"]
    assert isinstance(message, HumanMessage)
    assert message.content == "what's up?"
    assert call["context"] == ContextSchema(user_id=USER_ID, chat_id=CHAT.id, mode=Mode.TEXT)
    assert call["stream_mode"] == "messages"
    assert call["config"] == {"metadata": {"chat_id": str(CHAT.id)}}


async def test_tool_calls_become_started_and_done_events():
    graph = FakeGraph(
        [
            (AIMessageChunk("", tool_call_chunks=[{"name": "tavily_search", "args": "", "id": "c1", "index": 0}]), "agent"),
            # follow-up chunks of the same call carry only args, no name
            (AIMessageChunk("", tool_call_chunks=[{"name": None, "args": '{"q":1}', "id": None, "index": 0}]), "agent"),
            (ToolMessage("page text", tool_call_id="c1", name="tavily_search"), "agent_tools"),
            (AIMessageChunk("Found it"), "agent"),
        ]
    )

    events = (await collect(graph))[1:-1]

    assert events == [
        ToolEvent(name="tavily_search", status="started"),
        ToolEvent(name="tavily_search", status="done"),
        TokenEvent(text="Found it"),
    ]


async def test_block_content_is_streamed_as_text():
    chunk = AIMessageChunk([{"type": "text", "text": "block text"}])
    events = await collect(FakeGraph([(chunk, "agent")]))
    assert events[1] == TokenEvent(text="block text")


async def test_messages_from_other_nodes_are_not_streamed():
    graph = FakeGraph(
        [
            (ToolMessage("4000 chars", tool_call_id="c1", name="x"), "trim_tool_output"),
            (AIMessageChunk("internal"), "save"),
        ]
    )
    events = await collect(graph)
    assert [type(e) for e in events] == [ChatEvent, DoneEvent]


async def test_failure_mid_stream_ends_with_error_and_no_done(caplog):
    graph = FakeGraph([(AIMessageChunk("partial"), "agent"), RuntimeError("llm exploded")])

    events = await collect(graph)

    assert events[-2:] == [TokenEvent(text="partial"), ErrorEvent(detail="Something went wrong")]
    assert not any(isinstance(e, DoneEvent) for e in events)
    assert "Chat stream failed" in caplog.text


async def test_internal_error_details_are_not_leaked():
    events = await collect(FakeGraph([RuntimeError("password=hunter2")]))
    assert "hunter2" not in events[-1].detail


async def test_sets_log_context_for_the_request():
    seen = {}

    class SpyGraph(FakeGraph):
        async def astream(self, input, **kwargs):
            seen.update(log_context.get())
            return
            yield

    await collect(SpyGraph())

    assert seen == {"chat_id": str(CHAT.id), "user_id": str(USER_ID), "mode": "text"}


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (TimeoutError(), "The request took too long, please try again"),
        (SimpleNamespace(status_code=429), "The service is busy right now, please try again in a moment"),
        (SimpleNamespace(code=429), "The service is busy right now, please try again in a moment"),
        (SimpleNamespace(status_code=500), "Something went wrong"),
        (ValueError("x"), "Something went wrong"),
    ],
)
def test_user_message(exc, message):
    assert _user_message(exc) == message
