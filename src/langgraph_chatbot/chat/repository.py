import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph_chatbot.models.chat import Chat
from langgraph_chatbot.models.user import User


class EmailAlreadyExists(Exception):
    """User with this email already exists."""


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, id: uuid.UUID, user_id) -> Chat | None:
        stmt = select(Chat).where(Chat.id == id, Chat.user_id == user_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: uuid.UUID) -> list[Chat] | None:
        stmt = (
            select(Chat).where(Chat.user_id == user_id).order_by(Chat.updated_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def add(self, chat: Chat) -> Chat:
        self._session.add(chat)
        await self._session.flush()
        return chat

    async def delete(self, id: uuid.UUID, user_id: uuid.UUID) -> None:
        chat = await self.get_by_id(id, user_id)
        if not chat:
            return

        await self._session.delete(chat)
