import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain.mcp import MCPAdapter

from langgraph_chatbot.ai.bootstrap import build_runtime
from langgraph_chatbot.auth.router import router as auth_router
from langgraph_chatbot.chat.router import router as chat_router
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.db import build_engine, build_sessionmaker
from langgraph_chatbot.voice.router import router as voice_router

# our own logs at INFO (e.g. which MCP tools were loaded), libraries stay quiet
logging.basicConfig(level=logging.WARNING)
logging.getLogger("langgraph_chatbot").setLevel(logging.INFO)


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
app.include_router(voice_router)

app.mount(
    "/voice-ui",
    StaticFiles(directory=Path(__file__).parent / "voice" / "static", html=True),
    name="voice-ui",
)
