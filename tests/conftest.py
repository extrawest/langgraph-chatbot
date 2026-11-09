"""Shared fixtures.

Hermetic by design: no Postgres, Qdrant server, LLM, Mem0, LiveKit or MCP server is
needed. Postgres -> SQLite in memory, Qdrant -> qdrant-client local mode, the LLM ->
a scripted fake model, the LangGraph graph -> a scripted fake (API tests only).
"""

import os

# --- must run before any langgraph_chatbot import -------------------------------
# main.py / voice/agent.py / ui/app.py call load_dotenv() at import time, which would
# pull the developer's real .env (API keys, LangSmith tracing) into the test process.
import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False
for _var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
    os.environ[_var] = "false"
# ---------------------------------------------------------------------------------

import uuid  # noqa: E402
from collections.abc import AsyncIterator, Callable  # noqa: E402
from datetime import timedelta  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool  # noqa: E402

import langgraph_chatbot.models  # noqa: E402,F401  (registers all tables on Base)
from langgraph_chatbot.auth.security import (  # noqa: E402
    create_access_token,
    get_password_hash,
)
from langgraph_chatbot.chat.repositories.messages_store import (  # noqa: E402
    MessagesStore,
)
from langgraph_chatbot.config import Settings  # noqa: E402
from langgraph_chatbot.db import Base  # noqa: E402
from langgraph_chatbot.models.chat import Chat  # noqa: E402
from langgraph_chatbot.models.user import User  # noqa: E402
from tests.helpers import (  # noqa: E402
    DEFAULT_PASSWORD,
    FakeGraph,
    FakeUserMemory,
    ScriptedChatModel,
)

# Optional settings that would switch real integrations on if they leaked in from the
# shell environment.
_OPTIONAL_SECRET_VARS = (
    "VECTOR_STORE_API_KEY",
    "TAVILY_API_KEY",
    "FIRECRAWL_API_KEY",
    "ELEVEN_API_KEY",
)

# ------------------------------------------------------------------------ settings
@pytest.fixture
def settings(monkeypatch) -> Settings:
    for var in _OPTIONAL_SECRET_VARS:
        monkeypatch.delenv(var, raising=False)
    return Settings(
        _env_file=None,
        db_url="sqlite+aiosqlite://",
        vector_store_url="http://qdrant.test:6333",
        jwt_secret="test-jwt-secret-that-is-long-enough-for-hs256",
        llm_api_key="test-llm-key",
        livekit_url="wss://livekit.test",
        livekit_api_key="test-livekit-key",
        livekit_api_secret="test-livekit-secret-that-is-long-enough",
        eleven_voice_id="test-voice",
    )

# ------------------------------------------------------------------------ database
@pytest.fixture
async def engine():
    # StaticPool: one shared connection, otherwise every connection gets its own
    # empty in-memory database
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest.fixture
def sessionmaker(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, expire_on_commit=False)

@pytest.fixture
async def session(sessionmaker) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as session:
        yield session

@pytest.fixture
def user_factory(sessionmaker) -> Callable[..., Any]:
    async def _create(
        email: str | None = None,
        password: str = DEFAULT_PASSWORD,
        is_active: bool = True,
    ) -> User:
        user = User(
            email=email or f"user-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash(password),
            is_active=is_active,
        )
        async with sessionmaker() as s:
            s.add(user)
            await s.commit()
        return user

    return _create

@pytest.fixture
def chat_factory(sessionmaker) -> Callable[..., Any]:
    async def _create(user: User, title: str = "Test chat", **kwargs) -> Chat:
        chat = Chat(user_id=user.id, title=title, **kwargs)
        async with sessionmaker() as s:
            s.add(chat)
            await s.commit()
        return chat

    return _create

@pytest.fixture
def make_token(settings) -> Callable[..., str]:
    def _make(subject: Any, expires_delta: timedelta | None = None) -> str:
        return create_access_token(
            {"sub": str(subject)},
            settings.jwt_secret,
            settings.jwt_algorithm,
            expires_delta=expires_delta or timedelta(minutes=5),
        )

    return _make

@pytest.fixture
def auth_headers(make_token) -> Callable[[User], dict[str, str]]:
    return lambda user: {"Authorization": f"Bearer {make_token(user.id)}"}

# --------------------------------------------------------------- vector store
@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()

@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=768)

@pytest.fixture
async def messages_store(qdrant, embeddings) -> MessagesStore:
    store = MessagesStore(collection_name="messages", embeddings=embeddings, client=qdrant)
    await store.ensure_collection()
    return store

# ---------------------------------------------------------------- test doubles
@pytest.fixture
def scripted_llm() -> Callable[..., ScriptedChatModel]:
    return lambda *responses: ScriptedChatModel(responses=list(responses))

@pytest.fixture
def user_memory() -> FakeUserMemory:
    return FakeUserMemory()

@pytest.fixture
def fake_graph() -> FakeGraph:
    return FakeGraph()

# ------------------------------------------------------------------ FastAPI app
@pytest.fixture
def app(settings, sessionmaker, messages_store, fake_graph):
    """The real FastAPI app with test state; lifespan (real runtime) is not run."""
    from langgraph_chatbot.config import get_settings
    from langgraph_chatbot.main import app

    app.state.sessionmaker = sessionmaker
    app.state.graph = fake_graph
    app.state.messages_store = messages_store
    app.dependency_overrides[get_settings] = lambda: settings
    yield app
    app.dependency_overrides.clear()
    for attr in ("sessionmaker", "graph", "messages_store"):
        delattr(app.state, attr)

@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
