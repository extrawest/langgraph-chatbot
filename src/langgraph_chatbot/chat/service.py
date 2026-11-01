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
)
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore
from langgraph_chatbot.chat.repositories.repository import ChatRepository
from langgraph_chatbot.chat.vector_models.stored_message import StoredMessage
from langgraph_chatbot.models.chat import Chat


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

    async def delete_chat(self, chat_id: uuid.UUID, user_id: uuid.UUID) -> None:
        await self._repo.delete(chat_id, user_id)
        await self._session.commit()

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
                if (
                    meta["langgraph_node"] == "agent"
                    and isinstance(chunk.content, str)
                    and chunk.content
                ):
                    yield TokenEvent(text=chunk.content)
        except Exception as e:
            print(e)
            yield ErrorEvent(detail="Something went wrong")
            return

        yield DoneEvent()
