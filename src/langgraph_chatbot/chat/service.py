import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from langgraph_chatbot.chat.repository import ChatRepository
from langgraph_chatbot.models.chat import Chat


class ChatService:
    def __init__(self, session: AsyncSession, repo: ChatRepository) -> None:
        self._session = session
        self._repo = repo

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
