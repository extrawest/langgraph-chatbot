from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from langgraph_chatbot.ai.context import ContextSchema
from langgraph_chatbot.ai.state import AppState
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore

SYSTEM_PROMPT = """
You are a helpful, concise AI assistant in a chat application.

- Reply in the same language the user writes in.
- Be direct: answer first, then add details only if they help.
- Use Markdown when it improves readability (lists, code blocks); the chat renders it.
- If you are not sure about something or lack information, say so instead of guessing.
- If a request is ambiguous, ask one short clarifying question.
"""


def make_load_history_node(messages_store: MessagesStore):
    async def load_history(state: AppState, runtime: Runtime[ContextSchema]):
        ctx = runtime.context
        stored = await messages_store.recent(
            user_id=ctx.user_id, chat_id=ctx.chat_id, limit=20
        )
        history = [
            HumanMessage(content=m.content)
            if m.role == "user"
            else AIMessage(content=m.content)
            for m in stored
        ]
        # history (oldest -> newest) + the question that triggered this run
        return {"messages": [*history, HumanMessage(content=state["user_input"])]}

    return load_history


def make_agent_node(llm: BaseChatModel):
    async def chat(state: AppState):
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    return chat


def make_save_node(messages_store: MessagesStore):
    async def save(state: AppState, runtime: Runtime[ContextSchema]):
        ctx = runtime.context
        answer = state["messages"][-1]
        # sequential on purpose: user message must get the earlier created_at
        await messages_store.add(
            user_id=ctx.user_id,
            chat_id=ctx.chat_id,
            role="user",
            content=state["user_input"],
        )
        await messages_store.add(
            user_id=ctx.user_id,
            chat_id=ctx.chat_id,
            role="assistant",
            content=answer.content,
        )
        return {}

    return save
