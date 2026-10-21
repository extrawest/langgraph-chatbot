from contextlib import asynccontextmanager

from fastapi import FastAPI

from langgraph_chatbot.auth.router import router as auth_router
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.db import build_engine, build_sessionmaker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    engine = build_engine(settings.db_url)
    app.state.engine = engine
    app.state.sessionmaker = build_sessionmaker(engine)

    yield

    await engine.dispose()


app = FastAPI(title="Booking service", lifespan=lifespan)

app.include_router(auth_router)
