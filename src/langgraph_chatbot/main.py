from contextlib import asynccontextmanager

from fastapi import FastAPI

from langgraph_chatbot.ai.bootstrap import build_runtime
from langgraph_chatbot.auth.router import router as auth_router
from langgraph_chatbot.chat.router import router as chat_router
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.db import build_engine, build_sessionmaker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    runtime = await build_runtime(settings)
    app.state.runtime = runtime
    app.state.graph = runtime.graph
    app.state.messages_store = runtime.messages_store
    app.state.user_memory = runtime.user_memory

    engine = build_engine(settings.db_url)
    app.state.engine = engine
    app.state.sessionmaker = build_sessionmaker(engine)

    yield

    await runtime.aclose()
    await engine.dispose()


app = FastAPI(title="Langgraph Chatbot", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(chat_router)
