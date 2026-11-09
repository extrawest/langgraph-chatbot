import logging
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph_chatbot.ai.context import ContextSchema
from langgraph_chatbot.chat.events import (
    ChatEvent,
    DoneEvent,
    ErrorEvent,
    StreamEvent,
    TokenEvent,
    ToolEvent,
)
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore
from langgraph_chatbot.chat.repositories.repository import ChatRepository
from langgraph_chatbot.chat.vector_models.stored_message import StoredMessage
from langgraph_chatbot.logger import log_context
from langgraph_chatbot.models.chat import Chat

logger = logging.getLogger(__name__)


def _user_message(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "The request took too long, please try again"
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return "The service is busy right now, please try again in a moment"
    return "Something went wrong"


class ChatService:
    def __init__(
        self,
        session: AsyncSession,
        repo: ChatRepository,
        graph: CompiledStateGraph,
        messages_store: MessagesStore,
    ) -> None:
        self._session = session
        self._repo = repo
        self._graph = graph
        self._messages_store = messages_store

    async def list_chats(self, user_id: uuid.UUID) -> list[Chat]:
        chats = await self._repo.list_for_user(user_id)
        return chats

    async def get_chat(self, chat_id: uuid.UUID, user_id: uuid.UUID) -> Chat | None:
        chat = await self._repo.get_by_id(chat_id, user_id)
        return chat

    async def get_or_create_chat(
        self, chat_id: uuid.UUID | None, user_id: uuid.UUID, title: str
    ):
        if chat_id is None:
            return await self.create_chat(user_id, title)
        return await self.get_chat(chat_id, user_id)

    async def create_chat(self, user_id: uuid.UUID, title: str) -> Chat:
        chat = Chat(user_id=user_id, title=title)

        await self._repo.add(chat)
        await self._session.commit()
        return chat

    async def rename_chat(
        self, chat_id: uuid.UUID, user_id: uuid.UUID, title: str
    ) -> Chat | None:
        chat = await self._repo.get_by_id(chat_id, user_id)
        if chat is None:
            return None

        chat.title = title.strip()
        await self._session.commit()
        await self._session.refresh(chat)
        return chat

    async def delete_chat(self, chat_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Delete the chat and its message history. False if it is not the user's."""
        if not await self._repo.delete(chat_id, user_id):
            return False
        await self._session.commit()
        # Postgres first: once the row is gone the chat is unreachable, so if Qdrant
        # fails below, the leftover points are orphans nobody can read (the chat_id is
        # a fresh UUID and every read is scoped by it), not a half-deleted visible chat.
        await self._messages_store.delete_chat(user_id=user_id, chat_id=chat_id)
        return True

    async def get_messages(
        self, chat_id: uuid.UUID, user_id: uuid.UUID, limit: int = 100
    ) -> list[StoredMessage] | None:
        """History of a chat, oldest first. None if the chat is not the user's."""
        chat = await self._repo.get_by_id(chat_id, user_id)
        if chat is None:
            return None
        return await self._messages_store.recent(
            user_id=user_id, chat_id=chat_id, limit=limit
        )

    async def stream_reply(
        self, chat: Chat, user_id: uuid.UUID, message
    ) -> AsyncIterator[StreamEvent]:
        chat_id = chat.id
        # every log record of this request (nodes included) now carries these fields;
        # no reset: each request runs in its own task, so nothing leaks to others
        log_context.set(
            {"chat_id": str(chat_id), "user_id": str(user_id), "mode": "text"}
        )
        yield ChatEvent(chat_id=chat_id, title=chat.title)
        config = {"metadata": {"chat_id": str(chat_id)}}
        context = ContextSchema(user_id=user_id, chat_id=chat_id)

        try:
            async for chunk, meta in self._graph.astream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                context=context,
                stream_mode="messages",
            ):
                node = meta["langgraph_node"]
                if node == "agent":
                    for call in chunk.tool_call_chunks or []:
                        if call.get("name"):
                            yield ToolEvent(name=call["name"], status="started")
                    # .text, not .content: content may be a list of blocks
                    if chunk.text:
                        yield TokenEvent(text=str(chunk.text))
                elif node == "agent_tools":
                    result = str(chunk.text)
                    logger.info(
                        "Tool %s returned %d chars: %s",
                        chunk.name,
                        len(result),
                        result[:2000],
                    )
                    yield ToolEvent(name=chunk.name, status="done")
        except Exception as exc:
            logger.exception("Chat stream failed (chat_id=%s)", chat_id)
            yield ErrorEvent(detail=_user_message(exc))
            return

        yield DoneEvent()
