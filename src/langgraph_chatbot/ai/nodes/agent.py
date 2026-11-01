import asyncio
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from langgraph_chatbot.ai.context import ContextSchema, Mode
from langgraph_chatbot.ai.memory import UserMemory
from langgraph_chatbot.ai.state import AppState
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 20
RECALL_TIMEOUT_SECONDS = 3.0

SYSTEM_PROMPT = """
You are a helpful, concise AI assistant in a chat application.

- Reply in the same language the user writes in.
- Be direct: answer first, then add details only if they help.
- Use Markdown when it improves readability (lists, code blocks); the chat renders it.
- If you are not sure about something or lack information, say so instead of guessing.
- If a request is ambiguous, ask one short clarifying question.
"""

VOICE_SYSTEM_PROMPT = """
You are a helpful voice AI assistant. The user hears your answer; nothing is shown
on a screen.

- Reply in the same language the user speaks.
- Keep it short: one or two sentences, and offer more only if the user asks.
- Speak plain text: no Markdown, no lists, no code blocks, no emojis, no URLs.
- Write numbers, dates and units the way a person would say them out loud.
- If you are not sure about something, say so instead of guessing.
"""


def _current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The turn in progress: the newest user message and everything after it
    (later: tool calls and their results).

    Everything before it is history. In voice mode LiveKit keeps the whole session
    in `messages`; we ignore that part on purpose: Qdrant is the single source of
    history, so text and voice see exactly the same conversation.
    """
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[last_human:]


def _question(messages: list[BaseMessage]) -> str:
    """Text of the newest user message - the question this run answers."""
    return _current_turn(messages)[0].content


def make_load_history_node(messages_store: MessagesStore):
    async def load_history(state: AppState, runtime: Runtime[ContextSchema]):
        ctx = runtime.context
        stored = await messages_store.recent(
            user_id=ctx.user_id, chat_id=ctx.chat_id, limit=HISTORY_LIMIT
        )
        history = [
            HumanMessage(content=m.content)
            if m.role == "user"
            else AIMessage(content=m.content)
            for m in stored
        ]
        return {"history": history}

    return load_history


def make_recall_node(user_memory: UserMemory):
    async def recall(state: AppState, runtime: Runtime[ContextSchema]):
        try:
            # memory must never break the chat, nor keep a voice answer waiting
            memories = await asyncio.wait_for(
                user_memory.recall(
                    user_id=runtime.context.user_id,
                    query=_question(state["messages"]),
                ),
                timeout=RECALL_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Mem0 recall failed")
            memories = []
        return {"memories": memories}

    return recall


def _system_prompt(mode: Mode, memories: list[str]) -> str:
    base = VOICE_SYSTEM_PROMPT if mode is Mode.VOICE else SYSTEM_PROMPT
    if not memories:
        return base
    facts = "\n".join(f"- {m}" for m in memories)
    return (
        f"{base}\n"
        "Known facts about the user from earlier conversations "
        "(use them only when relevant, do not mention that you store memories):\n"
        f"{facts}\n"
    )


def make_agent_node(llm: BaseChatModel):
    async def chat(state: AppState, runtime: Runtime[ContextSchema]):
        prompt = _system_prompt(runtime.context.mode, state.get("memories", []))
        response = await llm.ainvoke(
            [
                SystemMessage(content=prompt),
                *state.get("history", []),
                *_current_turn(state["messages"]),
            ]
        )
        return {"messages": [response]}

    return chat


def make_save_node(messages_store: MessagesStore, user_memory: UserMemory):
    async def save(state: AppState, runtime: Runtime[ContextSchema]):
        ctx = runtime.context
        question = _question(state["messages"])
        answer = state["messages"][-1].content

        # sequential on purpose: the user message must get the earlier created_at
        await messages_store.add(
            user_id=ctx.user_id, chat_id=ctx.chat_id, role="user", content=question
        )
        await messages_store.add(
            user_id=ctx.user_id, chat_id=ctx.chat_id, role="assistant", content=answer
        )
        user_memory.remember_in_background(
            user_id=ctx.user_id,
            chat_id=ctx.chat_id,
            user_message=question,
            assistant_message=answer,
        )
        return {}

    return save
