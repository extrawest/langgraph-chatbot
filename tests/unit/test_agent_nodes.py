import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph_chatbot.ai.context import ContextSchema, Mode
from langgraph_chatbot.ai.nodes import agent as agent_module
from langgraph_chatbot.ai.nodes.agent import (
    HISTORY_LIMIT,
    SYSTEM_PROMPT,
    VOICE_SYSTEM_PROMPT,
    _current_turn,
    _question,
    _system_prompt,
    _turn_id,
    make_agent_node,
    make_load_history_node,
    make_recall_node,
    make_save_node,
    make_save_user_question_node,
)
from langgraph_chatbot.chat.vector_models.stored_message import StoredMessage

USER_ID = uuid.uuid4()
CHAT_ID = uuid.uuid4()


def runtime(mode: Mode = Mode.TEXT):
    # nodes only read runtime.context
    return SimpleNamespace(context=ContextSchema(user_id=USER_ID, chat_id=CHAT_ID, mode=mode))


def conversation():
    """Earlier session messages (voice keeps them), then the current turn with a tool call."""
    return [
        HumanMessage("old question", id="h-old"),
        AIMessage("old answer", id="a-old"),
        HumanMessage("what is 2+2?", id="h-now"),
        AIMessage("", tool_calls=[{"name": "calc", "args": {}, "id": "call-1"}], id="a-tool"),
        ToolMessage("4", tool_call_id="call-1", id="t-1"),
    ]


# ---------------------------------------------------------------- helpers
def test_current_turn_starts_at_newest_human_message():
    turn = _current_turn(conversation())
    assert [m.id for m in turn] == ["h-now", "a-tool", "t-1"]


def test_question_and_turn_id_come_from_newest_human_message():
    assert _question(conversation()) == "what is 2+2?"
    assert _turn_id(conversation()) == "h-now"


def test_system_prompt_by_mode():
    assert _system_prompt(Mode.TEXT, []) == SYSTEM_PROMPT
    assert _system_prompt(Mode.VOICE, []) == VOICE_SYSTEM_PROMPT


def test_system_prompt_lists_memories():
    prompt = _system_prompt(Mode.TEXT, ["likes tea", "name is Ann"])

    assert prompt.startswith(SYSTEM_PROMPT)
    assert "- likes tea\n- name is Ann" in prompt
    assert "do not mention that you store memories" in prompt


# ------------------------------------------------------------ load_history
async def test_load_history_returns_plain_dicts():
    store = AsyncMock()
    store.recent.return_value = [
        StoredMessage(role="user", content="hi"),
        StoredMessage(role="assistant", content="hello"),
        StoredMessage(role="something-else", content="?"),
    ]

    result = await make_load_history_node(store)({"messages": []}, runtime())

    store.recent.assert_awaited_once_with(user_id=USER_ID, chat_id=CHAT_ID, limit=HISTORY_LIMIT)
    assert result == {
        "history": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "assistant", "content": "?"},
        ]
    }


# ------------------------------------------------------------------ recall
async def test_recall_queries_with_current_question(user_memory):
    user_memory.memories = ["likes tea"]

    result = await make_recall_node(user_memory)({"messages": conversation()}, runtime())

    assert result == {"memories": ["likes tea"]}
    assert user_memory.recall_calls == [{"user_id": USER_ID, "query": "what is 2+2?", "limit": 5}]


async def test_recall_failure_does_not_break_the_chat(caplog):
    broken = AsyncMock()
    broken.recall.side_effect = RuntimeError("mem0 down")

    result = await make_recall_node(broken)({"messages": conversation()}, runtime())

    assert result == {"memories": []}
    assert "Mem0 recall failed" in caplog.text


async def test_recall_times_out(monkeypatch):
    monkeypatch.setattr(agent_module, "RECALL_TIMEOUT_SECONDS", 0.01)
    async def never_answers(**kwargs):
        await asyncio.sleep(10)

    slow = AsyncMock()
    slow.recall.side_effect = never_answers

    result = await make_recall_node(slow)({"messages": conversation()}, runtime())

    assert result == {"memories": []}


# ------------------------------------------------------------------- agent
@pytest.mark.parametrize(("mode", "prompt"), [(Mode.TEXT, SYSTEM_PROMPT), (Mode.VOICE, VOICE_SYSTEM_PROMPT)])
async def test_agent_prompt_is_system_history_then_current_turn(scripted_llm, mode, prompt):
    llm = scripted_llm(AIMessage("the answer"))
    state = {
        "messages": conversation(),
        "history": [{"role": "user", "content": "h1"}, {"role": "assistant", "content": "h2"}],
        "memories": [],
    }

    result = await make_agent_node(llm)(state, runtime(mode))

    assert result["messages"][0].content == "the answer"
    (sent,) = llm.calls
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == prompt
    assert [(m.type, m.content) for m in sent[1:3]] == [("human", "h1"), ("ai", "h2")]
    # the older session messages ("old question") are not sent: Qdrant history is used
    assert [m.id for m in sent[3:]] == ["h-now", "a-tool", "t-1"]


async def test_agent_works_without_history_and_memories_keys(scripted_llm):
    llm = scripted_llm(AIMessage("ok"))

    await make_agent_node(llm)({"messages": [HumanMessage("hi", id="h")]}, runtime())

    (sent,) = llm.calls
    assert [m.type for m in sent] == ["system", "human"]


async def test_agent_includes_memories_in_system_prompt(scripted_llm):
    llm = scripted_llm(AIMessage("ok"))
    state = {"messages": [HumanMessage("hi", id="h")], "memories": ["likes tea"]}

    await make_agent_node(llm)(state, runtime())

    assert "- likes tea" in llm.calls[0][0].content


# ------------------------------------------------------------------- saving
async def test_save_user_question():
    store = AsyncMock()

    await make_save_user_question_node(store, None)({"messages": conversation()}, runtime())

    store.add.assert_awaited_once_with(
        user_id=USER_ID, chat_id=CHAT_ID, turn_id="h-now", role="user", content="what is 2+2?"
    )


async def test_save_stores_answer_and_remembers_in_background(user_memory):
    store = AsyncMock()
    messages = [*conversation(), AIMessage("It is 4.", id="a-final")]

    result = await make_save_node(store, user_memory)({"messages": messages}, runtime())

    assert result == {}
    store.add.assert_awaited_once_with(
        user_id=USER_ID, chat_id=CHAT_ID, turn_id="h-now", role="assistant", content="It is 4."
    )
    assert user_memory.remembered == [
        {
            "user_id": USER_ID,
            "chat_id": CHAT_ID,
            "user_message": "what is 2+2?",
            "assistant_message": "It is 4.",
        }
    ]


async def test_save_skips_empty_answer(user_memory):
    store = AsyncMock()
    messages = [HumanMessage("hi", id="h"), AIMessage("", id="a")]

    await make_save_node(store, user_memory)({"messages": messages}, runtime())

    store.add.assert_not_awaited()
    assert len(user_memory.remembered) == 1


async def test_save_uses_text_of_block_content(user_memory):
    store = AsyncMock()
    answer = AIMessage([{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}])

    await make_save_node(store, user_memory)(
        {"messages": [HumanMessage("hi", id="h"), answer]}, runtime()
    )

    assert store.add.await_args.kwargs["content"] == "Hello there"
