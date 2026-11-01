from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import AsyncQdrantClient

from langgraph_chatbot.ai.graph import build_graph
from langgraph_chatbot.ai.llm import build_llm
from langgraph_chatbot.ai.memory import UserMemory
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore
from langgraph_chatbot.config import Settings


@dataclass
class AppRuntime:
    """Everything the AI side needs, built once per process.

    The FastAPI app builds one in lifespan; every voice job (separate process)
    builds its own in the entrypoint. Same code, separate objects.
    """

    qdrant: AsyncQdrantClient
    messages_store: MessagesStore
    user_memory: UserMemory
    graph: CompiledStateGraph

    async def aclose(self) -> None:
        await self.user_memory.aclose()
        await self.qdrant.close()


async def build_runtime(
    settings: Settings, *, ensure_collection: bool = True
) -> AppRuntime:
    llm = build_llm(settings)

    qdrant = AsyncQdrantClient(
        url=settings.vector_store_url,
        api_key=(
            settings.vector_store_api_key.get_secret_value()
            if settings.vector_store_api_key
            else None
        ),
    )
    messages_store = MessagesStore(
        collection_name="messages",
        client=qdrant,
        embeddings=OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            dimensions=settings.embedding_dims,
            check_embedding_ctx_length=False,
        ),
    )
    if ensure_collection:
        await messages_store.ensure_collection()

    user_memory = UserMemory.from_settings(settings)

    return AppRuntime(
        qdrant=qdrant,
        messages_store=messages_store,
        user_memory=user_memory,
        graph=build_graph(llm, messages_store, user_memory),
    )
