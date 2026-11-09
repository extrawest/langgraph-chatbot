"""Test doubles and small helpers shared by the test modules."""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

DEFAULT_PASSWORD = "correct-horse-battery"


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted responses in order and records every prompt it gets.

    A response that is an exception instance is raised instead of returned.
    """

    responses: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        response = self.responses.pop(0) if self.responses else AIMessage("(no script)")
        if isinstance(response, BaseException):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self


class FakeUserMemory:
    """In-memory stand-in for UserMemory (Mem0)."""

    def __init__(self, memories: list[str] | None = None) -> None:
        self.memories = memories or []
        self.recall_calls: list[dict] = []
        self.remembered: list[dict] = []

    async def recall(self, *, user_id, query, limit=5):
        self.recall_calls.append({"user_id": user_id, "query": query, "limit": limit})
        return list(self.memories)

    def remember_in_background(self, **kwargs) -> None:
        self.remembered.append(kwargs)

    async def aclose(self) -> None:
        pass


class FakeGraph:
    """Stands in for the compiled LangGraph graph: replays `(chunk, metadata)` pairs.

    Each item of `script` is either a `(chunk, node_name)` tuple or an exception to raise.
    """

    def __init__(self, script: list | None = None) -> None:
        self.script = script or []
        self.calls: list[dict] = []

    async def astream(self, input, **kwargs):
        self.calls.append({"input": input, **kwargs})
        for item in self.script:
            if isinstance(item, BaseException):
                raise item
            chunk, node = item
            yield chunk, {"langgraph_node": node}


def parse_sse(text: str) -> list[tuple[str | None, Any]]:
    """`event: x` / `data: {...}` blocks -> [(event_name, parsed_data), ...]."""
    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if name or data is not None:
            events.append((name, data))
    return events
