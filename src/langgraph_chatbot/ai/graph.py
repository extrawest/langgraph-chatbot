from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy, TimeoutPolicy
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from langgraph_chatbot.ai.context import ContextSchema
from langgraph_chatbot.ai.memory import UserMemory
from langgraph_chatbot.ai.nodes.agent import (
    make_agent_node,
    make_load_history_node,
    make_recall_node,
    make_save_node,
    make_save_user_question_node,
)
from langgraph_chatbot.ai.state import AppState
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore

TOOL_OUTPUT_LIMIT = 8000


async def trim_tool_output(state: AppState) -> dict:
    """Cap tool results so one huge page cannot flood the model context."""
    trimmed = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break
        text = message.text
        if len(text) > TOOL_OUTPUT_LIMIT:
            note = f"\n\n[truncated: {len(text)} chars total, first {TOOL_OUTPUT_LIMIT} shown]"
            # same id -> add_messages replaces the original message
            trimmed.append(
                ToolMessage(
                    content=text[:TOOL_OUTPUT_LIMIT] + note,
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                    id=message.id,
                )
            )
    return {"messages": trimmed}


def _custom_retry_on(exc: BaseException) -> bool:
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code >= 500
    return isinstance(
        exc,
        (
            ResponseHandlingException,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
        ),
    )


def build_graph(
    llm: BaseChatModel,
    messages_store: MessagesStore,
    user_memory: UserMemory,
    tools,
):
    workflow = StateGraph(AppState, context_schema=ContextSchema)
    tool_node = ToolNode(
        tools,
        handle_tool_errors=True,
    )

    agent_llm = llm.bind_tools(tools) if tools else llm
    workflow.add_node("load_history", make_load_history_node(messages_store))
    workflow.add_node("recall", make_recall_node(user_memory))
    workflow.add_node(
        "agent",
        make_agent_node(agent_llm),
        retry_policy=_custom_retry_on,
    )
    workflow.add_node(
        "save_user_question", make_save_user_question_node(messages_store, user_memory)
    )
    workflow.add_node(
        "save",
        make_save_node(messages_store, user_memory),
        retry_policy=RetryPolicy(retry_on=_custom_retry_on),
    )
    workflow.add_node("agent_tools", tool_node, timeout=TimeoutPolicy(run_timeout=20.0))
    workflow.add_node("trim_tool_output", trim_tool_output)

    # history (Qdrant) and long-term memory (Mem0) load in parallel; agent waits for both
    workflow.add_edge(START, "load_history")
    workflow.add_edge(START, "recall")
    workflow.add_edge(["load_history", "recall"], "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "agent_tools",
            END: "save_user_question",
        },
    )
    workflow.add_edge("agent_tools", "trim_tool_output")
    workflow.add_edge("trim_tool_output", "agent")

    workflow.add_edge("save_user_question", "save")
    workflow.add_edge("save", END)
    return workflow.compile()
