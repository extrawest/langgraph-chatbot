from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_chatbot.ai.nodes.agent import make_agent_node
from langgraph_chatbot.ai.state import AppState


def build_graph(llm: BaseChatModel, memory, tools):
    workflow = StateGraph(AppState)
    tool_node = ToolNode([])

    workflow.add_node("agent", make_agent_node(llm.bind_tools(tools)))
    workflow.add_node("agent_tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "agent_tools",
            END: END,
        },
    )

    return workflow.compile()
