import asyncio
import logging
from urllib.parse import urlparse
from uuid import UUID

from mem0 import AsyncMemory

from langgraph_chatbot.config import Settings

logger = logging.getLogger(__name__)

MEM0_COLLECTION = "mem0"


def build_mem0_config(settings: Settings) -> dict:
    api_key = settings.llm_api_key.get_secret_value()

    vector_cfg: dict = {
        "collection_name": MEM0_COLLECTION,
        "embedding_model_dims": settings.embedding_dims,
    }
    if settings.vector_store_api_key:
        vector_cfg["url"] = settings.vector_store_url
        vector_cfg["api_key"] = settings.vector_store_api_key.get_secret_value()
    else:
        parsed = urlparse(settings.vector_store_url)
        vector_cfg["host"] = parsed.hostname or "localhost"
        vector_cfg["port"] = parsed.port or 6333

    return {
        "vector_store": {"provider": "qdrant", "config": vector_cfg},
        "llm": {
            "provider": "openai",
            "config": {
                "model": settings.llm_model,
                "api_key": api_key,
                "openai_base_url": settings.llm_base_url,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": settings.embedding_model,
                "embedding_dims": settings.embedding_dims,
                "api_key": api_key,
                "openai_base_url": settings.llm_base_url,
            },
        },
    }


class UserMemory:
    """Cross-chat memory of one user (facts, preferences). Thin wrapper over Mem0.

    Isolation: every call takes user_id (from the JWT) and passes it to Mem0.
    """

    def __init__(self, memory: AsyncMemory) -> None:
        self._memory = memory
        self._tasks: set[asyncio.Task] = set()

    @classmethod
    def from_settings(cls, settings: Settings) -> UserMemory:
        return cls(AsyncMemory.from_config(build_mem0_config(settings)))

    async def recall(self, *, user_id: UUID, query: str, limit: int = 5) -> list[str]:
        result = await self._memory.search(
            query, filters={"user_id": str(user_id)}, top_k=limit
        )
        return [item["memory"] for item in result.get("results", [])]

    async def remember(
        self, *, user_id: UUID, chat_id: UUID, user_message: str, assistant_message: str
    ) -> None:
        await self._memory.add(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            user_id=str(user_id),
            metadata={"chat_id": str(chat_id)},
        )

    def remember_in_background(self, **kwargs) -> None:
        """Mem0.add calls an LLM (seconds): never make the user wait for it."""
        task = asyncio.create_task(self._safe_remember(**kwargs))
        self._tasks.add(task)  # keep a reference, otherwise the task may be GC'd
        task.add_done_callback(self._tasks.discard)

    async def _safe_remember(self, **kwargs) -> None:
        try:
            await self.remember(**kwargs)
        except Exception:
            logger.exception("Mem0 add failed")

    async def aclose(self) -> None:
        """Let pending writes finish on shutdown."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
