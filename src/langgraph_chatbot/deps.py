from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph_chatbot.auth.security import decode_access_token
from langgraph_chatbot.auth.service import AuthService
from langgraph_chatbot.config import Settings, get_settings
from langgraph_chatbot.models.user import User
from langgraph_chatbot.schemas.user import UserRead
from langgraph_chatbot.users.repository import UserRepository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def get_user_repo(session: SessionDep) -> UserRepository:
    return UserRepository(session)


def get_auth_service(session: SessionDep, users: UserRepoDep) -> AuthService:
    return AuthService(session, users)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.sessionmaker() as session:
        yield session


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    user_repo: UserRepoDep,
    settings: SettingsDep,
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(
            token,
            settings.jwt_secret,
            settings.jwt_algorithm,
        )
        user_id = UUID(payload["sub"])
    except InvalidTokenError, KeyError, ValueError:
        raise credentials_exception

    user = await user_repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[UserRead, Depends(get_current_user)],
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_graph(request: Request):
    return request.app.state.graph


SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
UserRepoDep = Annotated[UserRepository, Depends(get_user_repo)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
