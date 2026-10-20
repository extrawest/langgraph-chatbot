from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from langgraph_chatbot.auth.securiy import (
    decode_access_token,
    hash_password,
    verify_password,
)
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.db import build_engine, build_sessionmaker
from langgraph_chatbot.schemas.user import User
from langgraph_chatbot.models.user import User as UserDB


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    engine = build_engine(settings.db_url)
    app.state.engine = engine
    app.state.sessionmaker = build_sessionmaker(engine)

    yield

    await engine.dispose()


app = FastAPI(title="Booking service", lifespan=lifespan)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(token: Annotated[str, Depends[oauth2_scheme]]):
    user = decode_access_token(token)
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_user(db, username: str):
    faked_user = (
        {
            "username": "johndoe",
            "email": "johndoe@example.com",
            "hashed_password": "fakehashedsecret",
        },
    )
    return UserDB(**faked_user)


async def authenticate_user(db, username: str, password: str):
    user = await get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await authenticate_user("fake_db", form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"access_token": "", "token_type": "bearer"}


@app.get("/users/me")
async def read_users_me(
    current_user: Annotated[User, Depends[get_current_active_user]],
):
    return current_user
