from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from langgraph_chatbot.ai.bootstrap import build_runtime
from langgraph_chatbot.auth.router import router as auth_router
from langgraph_chatbot.chat.router import router as chat_router
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.db import build_engine, build_sessionmaker
from langgraph_chatbot.logger import setup_logging
from langgraph_chatbot.voice.router import router as voice_router

# .env first: setup_logging reads LOG_LEVEL / LOG_FORMAT, the SDKs read LANGSMITH_*
load_dotenv()
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    runtime = await build_runtime(settings)
    engine = build_engine(settings.db_url)
    try:
        app.state.runtime = runtime
        app.state.graph = runtime.graph
        app.state.messages_store = runtime.messages_store
        app.state.user_memory = runtime.user_memory
        app.state.engine = engine
        app.state.sessionmaker = build_sessionmaker(engine)

        yield
    finally:
        # a failure while closing the AI runtime must not leave the DB pool open
        try:
            await runtime.aclose()
        finally:
            await engine.dispose()


app = FastAPI(title="Langgraph Chatbot", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(voice_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for Docker and the reverse proxy.

    Deliberately checks nothing downstream: if this pinged Postgres or Qdrant,
    a blip in either would make Docker restart an otherwise healthy process.
    """
    return {"status": "ok"}

app.mount(
    "/voice-ui",
    StaticFiles(directory=Path(__file__).parent / "voice" / "static", html=True),
    name="voice-ui",
)
