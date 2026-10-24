from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from langgraph_chatbot.ai.context import ContextSchema
from langgraph_chatbot.ai.state import AppState

SYSTEM_PROMPT = ""


def make_agent_node(llm: BaseChatModel):
    async def chat(state: AppState, runtime: Runtime[ContextSchema]):
        user_id = runtime.context.user_id
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    return chat
