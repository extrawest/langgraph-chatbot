from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient

from langgraph_chatbot.ai.graph import build_graph
from langgraph_chatbot.ai.llm import build_llm
from langgraph_chatbot.auth.router import router as auth_router
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore
from langgraph_chatbot.chat.router import router as chat_router
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.db import build_engine, build_sessionmaker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    llm = build_llm(settings)

    qdrant = AsyncQdrantClient(url=settings.vector_store_url)
    messages_store = MessagesStore(
        collection_name="messages",
        client=qdrant,
        embeddings=OpenAIEmbeddings(
            model="gemini-embedding-001",
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            dimensions=768,
            check_embedding_ctx_length=False,
        ),
    )
    await messages_store.ensure_collection()
    app.state.graph = build_graph(llm, messages_store)
    app.state.messages_store = messages_store

    engine = build_engine(settings.db_url)
    app.state.engine = engine
    app.state.sessionmaker = build_sessionmaker(engine)

    yield

    await engine.dispose()
    await qdrant.close()


app = FastAPI(title="Langgraph Chatbot", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(chat_router)
