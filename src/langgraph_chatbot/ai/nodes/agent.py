from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from langgraph_chatbot.ai.context import ContextSchema
from langgraph_chatbot.ai.state import AppState

SYSTEM_PROMPT = """
You are a helpful, concise AI assistant in a chat application.

- Reply in the same language the user writes in.
- Be direct: answer first, then add details only if they help.
- Use Markdown when it improves readability (lists, code blocks); the chat renders it.
- If you are not sure about something or lack information, say so instead of guessing.
- If a request is ambiguous, ask one short clarifying question.
"""


def make_agent_node(llm: BaseChatModel):
    async def chat(state: AppState, runtime: Runtime[ContextSchema]):
        user_id = runtime.context.user_id
        response = await llm.astream(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    return chat
