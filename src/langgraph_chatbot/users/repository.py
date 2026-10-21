import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph_chatbot.models.user import User


class EmailAlreadyExists(Exception):
    """User with this email already exists."""


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, id: uuid.UUID) -> User | None:
        user = await self._session.get(User, id)
        return user

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def add(self, user: User) -> User:
        self._session.add(user)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            raise EmailAlreadyExists(user.email) from exc
        return user
