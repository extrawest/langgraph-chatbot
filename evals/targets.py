"""What LangSmith runs for each dataset example.

One AppRuntime is built per process and reused across examples - building it per
example would reconnect the MCP servers every time. Only the tenant ids are fresh.
"""

from uuid import UUID, uuid4, uuid5

from langchain_core.messages import HumanMessage

from langgraph_chatbot.ai.bootstrap import AppRuntime
from langgraph_chatbot.ai.context import ContextSchema, Mode
from langgraph_chatbot.voice.spoken_graph import SpokenOnly


class EvalTenants:
    """Fresh user/chat ids for one experiment run.

    Qdrant and Mem0 keep whatever a run writes. Reusing ids across runs would let
    run N recall facts written by run N-1, and the numbers would stop being
    comparable - the memory suite would start passing for the wrong reason.
    """

    def __init__(self, run_id: UUID | None = None) -> None:
        self.run_id = run_id or uuid4()
        self._ids: dict[tuple[str, str], UUID] = {}

    def _id(self, kind: str, label: str) -> UUID:
        key = (kind, label)
        if key not in self._ids:
            self._ids[key] = uuid5(self.run_id, f"{kind}:{label}")
        return self._ids[key]

    def user(self, label: str) -> UUID:
        return self._id("user", label)

    def chat(self, label: str) -> UUID:
        return self._id("chat", label)


async def _run_turn(graph, user_id: UUID, chat_id: UUID, text: str, mode: Mode) -> dict:
    """One turn through the graph, collecting what each consumer would have seen.

    Mirrors ChatService.stream_reply: `agent` tokens are the visible answer,
    `agent_tools` output is what the UI shows only as a status badge.
    """
    answer: list[str] = []
    tool_output: list[str] = []
    tools: list[str] = []
    nodes: set[str] = set()

    async for chunk, meta in graph.astream(
        {"messages": [HumanMessage(content=text)]},
        context=ContextSchema(user_id=user_id, chat_id=chat_id, mode=mode),
        stream_mode="messages",
    ):
        node = meta.get("langgraph_node")
        if node:
            nodes.add(node)
        if node == "agent":
            for call in getattr(chunk, "tool_call_chunks", None) or []:
                if call.get("name"):
                    tools.append(call["name"])
            if chunk.text:
                answer.append(str(chunk.text))
        elif node == "agent_tools":
            tool_output.append(str(chunk.text))

    return {
        "answer": "".join(answer),
        "tools": sorted(set(tools)),
        "tool_output": "".join(tool_output),
        "nodes": sorted(nodes),
    }


def make_target(runtime: AppRuntime, tenants: EvalTenants, *, voice: bool = False):
    """Build the callable LangSmith invokes per example.

    An example is a list of turns, so one shape covers single questions, the
    cross-chat memory cases (same user, two chats) and the isolation cases
    (two users). Only the last turn is scored.
    """
    graph = SpokenOnly(runtime.graph) if voice else runtime.graph
    mode = Mode.VOICE if voice else Mode.TEXT

    async def target(inputs: dict) -> dict:
        result: dict = {}
        for turn in inputs["turns"]:
            result = await _run_turn(
                graph,
                tenants.user(turn.get("user", "a")),
                tenants.chat(turn.get("chat", "c1")),
                turn["text"],
                mode,
            )
            # Mem0 writes run in the background (remember_in_background), so the
            # next turn would race them. aclose() only awaits the pending tasks,
            # which makes it a flush - the object stays usable afterwards.
            await runtime.user_memory.aclose()
        return result

    return target
