import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from langgraph_chatbot.auth.security import get_password_hash, verify_password
from langgraph_chatbot.models.user import User
from langgraph_chatbot.users.repository import UserRepository


class AuthService:
    def __init__(self, session: AsyncSession, repo: UserRepository) -> None:
        self._session = session
        self._repo = repo

    async def get_user(self, email: str):
        user = await self._repo.get_by_email(email)
        return user

    async def authenticate_user(self, email: str, password: str):
        user = await self._repo.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def register(self, email: str, password: str) -> User:
        user = User(
            email=email,
            hashed_password=get_password_hash(password),
        )
        await self._repo.add(user)
        await self._session.commit()
        return user
