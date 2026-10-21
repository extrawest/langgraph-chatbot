from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph_chatbot.auth.security import decode_access_token
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.main import UserDB
from langgraph_chatbot.schemas.user import UserRead

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.sessionmaker() as session:
        yield session


async def get_user(db, username: str):
    faked_user = {
        "username": "johndoe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
    }
    return UserDB(**faked_user)


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    settings = get_settings()
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
    except InvalidTokenError:
        raise credentials_exception

    user = get_user({}, username=payload.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[UserRead, Depends(get_current_user)],
):
    if current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
