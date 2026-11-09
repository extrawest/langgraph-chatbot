"""Voice pieces that run without LiveKit: token, tool-output filter, entrypoint guards."""

import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest

from langgraph_chatbot.ai.context import ContextSchema, Mode
from langgraph_chatbot.voice.constants import VOICE_AGENT_NAME
from langgraph_chatbot.voice.spoken_graph import SpokenOnly
from langgraph_chatbot.voice.utils import build_voice_token
from tests.helpers import FakeGraph


# ------------------------------------------------------------------ token
def test_voice_token_claims(settings):
    user_id, chat_id = uuid.uuid4(), uuid.uuid4()

    token = build_voice_token(settings, user_id=user_id, chat_id=chat_id)
    claims = jwt.decode(
        token,
        settings.livekit_api_secret.get_secret_value(),
        algorithms=["HS256"],
        options={"verify_aud": False},
    )

    assert claims["iss"] == settings.livekit_api_key.get_secret_value()
    assert claims["sub"] == str(user_id)
    assert claims["attributes"] == {"chat_id": str(chat_id)}
    assert claims["video"]["roomJoin"] is True
    assert claims["video"]["room"].startswith(f"voice-{chat_id}-")
    assert claims["roomConfig"]["agents"][0]["agentName"] == VOICE_AGENT_NAME
    assert claims["exp"] - claims["nbf"] == timedelta(minutes=10).total_seconds()


def test_voice_token_uses_a_fresh_room_each_time(settings):
    def room(token):
        return jwt.decode(token, options={"verify_signature": False})["video"]["room"]

    ids = {"user_id": uuid.uuid4(), "chat_id": uuid.uuid4()}
    assert room(build_voice_token(settings, **ids)) != room(build_voice_token(settings, **ids))


# ------------------------------------------------------------- SpokenOnly
async def test_spoken_only_keeps_agent_tokens_and_forwards_args():
    graph = FakeGraph([("hello", "agent"), ("<tool page>", "agent_tools"), ("<trimmed>", "trim_tool_output"), ("bye", "agent")])

    out = [t async for t, _ in SpokenOnly(graph).astream({"messages": []}, stream_mode="messages")]

    assert out == ["hello", "bye"]
    assert graph.calls == [{"input": {"messages": []}, "stream_mode": "messages"}]


async def test_spoken_only_custom_node():
    graph = FakeGraph([("a", "agent"), ("b", "other")])
    assert [t async for t, _ in SpokenOnly(graph, node="other").astream({})] == ["b"]


# ------------------------------------------------------------- entrypoint
@pytest.fixture
def voice_agent(monkeypatch, settings):
    from langgraph_chatbot.voice import agent

    monkeypatch.setattr(agent, "get_settings", lambda: settings)
    return agent


def job_context(*, attributes=None, identity="user-1", participant_error=None):
    ctx = MagicMock()
    ctx.connect = AsyncMock()
    participant = SimpleNamespace(identity=identity, attributes=attributes or {})
    ctx.wait_for_participant = AsyncMock(side_effect=participant_error, return_value=participant)
    return ctx


async def test_entrypoint_room_closed_before_participant(voice_agent, monkeypatch):
    build = AsyncMock()
    monkeypatch.setattr(voice_agent, "build_runtime", build)
    ctx = job_context(participant_error=RuntimeError("closed"))

    await voice_agent.entrypoint(ctx)

    build.assert_not_awaited()
    ctx.shutdown.assert_not_called()


async def test_entrypoint_requires_chat_id(voice_agent, monkeypatch):
    build = AsyncMock()
    monkeypatch.setattr(voice_agent, "build_runtime", build)
    ctx = job_context(attributes={})

    await voice_agent.entrypoint(ctx)

    ctx.shutdown.assert_called_once_with(reason="missing chat_id")
    build.assert_not_awaited()


async def test_entrypoint_runtime_failure_shuts_down(voice_agent, monkeypatch, caplog):
    monkeypatch.setattr(voice_agent, "build_runtime", AsyncMock(side_effect=RuntimeError("qdrant down")))
    ctx = job_context(attributes={"chat_id": "chat-1"})

    await voice_agent.entrypoint(ctx)

    ctx.shutdown.assert_called_once_with(reason="runtime failed to start")
    assert "Voice runtime failed to start" in caplog.text


async def test_entrypoint_starts_session_with_voice_context(voice_agent, monkeypatch, settings):
    runtime = SimpleNamespace(graph=FakeGraph(), aclose=AsyncMock())
    build = AsyncMock(return_value=runtime)
    session = MagicMock()
    session.start = AsyncMock()
    adapter = MagicMock()
    tts = MagicMock()
    monkeypatch.setattr(voice_agent, "build_runtime", build)
    monkeypatch.setattr(voice_agent, "AgentSession", MagicMock(return_value=session))
    monkeypatch.setattr(voice_agent, "silero", MagicMock())
    monkeypatch.setattr(voice_agent, "inference", MagicMock())
    monkeypatch.setattr(voice_agent, "TurnHandlingOptions", MagicMock())
    monkeypatch.setattr(voice_agent, "elevenlabs", SimpleNamespace(STT=MagicMock(), TTS=tts))
    monkeypatch.setattr(voice_agent, "langchain", SimpleNamespace(LLMAdapter=adapter))
    ctx = job_context(identity="user-1", attributes={"chat_id": "chat-1"})

    await voice_agent.entrypoint(ctx)

    build.assert_awaited_once_with(settings, ensure_collection=False)
    adapter_kwargs = adapter.call_args.kwargs
    assert isinstance(adapter_kwargs["graph"], SpokenOnly)
    assert adapter_kwargs["context"] == ContextSchema(user_id="user-1", chat_id="chat-1", mode=Mode.VOICE)
    assert tts.call_args.kwargs["voice_id"] == settings.eleven_voice_id
    ctx.add_shutdown_callback.assert_called_once_with(runtime.aclose)
    session.start.assert_awaited_once()
    assert session.start.await_args.kwargs["room"] is ctx.room
    ctx.shutdown.assert_not_called()


def test_assistant_has_no_instructions_of_its_own(voice_agent):
    # the real prompt lives in the graph; see VOICE_SYSTEM_PROMPT
    assert voice_agent.Assistant().instructions == ""
