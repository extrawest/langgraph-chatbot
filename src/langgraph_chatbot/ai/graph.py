from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_chatbot.ai.context import ContextSchema
from langgraph_chatbot.ai.nodes.agent import (
    make_agent_node,
    make_load_history_node,
    make_save_node,
)
from langgraph_chatbot.ai.state import AppState
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore


def build_graph(llm: BaseChatModel, messages_store: MessagesStore):
    workflow = StateGraph(AppState, context_schema=ContextSchema)
    tool_node = ToolNode([])

    workflow.add_node("load_history", make_load_history_node(messages_store))
    workflow.add_node("agent", make_agent_node(llm))
    workflow.add_node("save", make_save_node(messages_store))
    workflow.add_node("agent_tools", tool_node)

    workflow.add_edge(START, "load_history")
    workflow.add_edge("load_history", "agent")
    workflow.add_edge("agent", "save")
    workflow.add_edge("save", END)
    # workflow.add_conditional_edges(
    #     "agent",
    #     tools_condition,
    #     {
    #         "tools": "agent_tools",
    #         END: END,
    #     },
    # )

    return workflow.compile()
