"""Lets only the assistant's own words reach the voice pipeline (TTS).

LiveKit's LangChain adapter turns every message with text that the graph streams
into speech. In `stream_mode="messages"` that includes ToolMessages (a whole web
page, from `agent_tools` and again from `trim_tool_output`), so without a filter the
agent would read tool output aloud. Only tokens produced by the `agent` node are kept.
The model itself still sees the tool results: they stay in the graph state.
"""

from typing import Any


class SpokenOnly:
    def __init__(self, graph: Any, node: str = "agent") -> None:
        self._graph = graph
        self._node = node

    async def astream(self, *args: Any, **kwargs: Any):
        async for token, metadata in self._graph.astream(*args, **kwargs):
            if metadata.get("langgraph_node") == self._node:
                yield token, metadata
