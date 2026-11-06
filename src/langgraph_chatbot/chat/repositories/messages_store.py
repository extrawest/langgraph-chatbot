import time
from uuid import UUID, uuid5

from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import (
    IntegerIndexParams,
    IntegerIndexType,
    KeywordIndexParams,
    PayloadSchemaType,
    PointStruct,
)

from langgraph_chatbot.chat.vector_models.stored_message import StoredMessage


# fixed namespace for deterministic point ids (any constant UUID works, it must not change)
_POINT_NAMESPACE = UUID("6f1c3b7e-5a2d-4c8e-9b41-2d7a0e5f8c13")


class MessagesStore:
    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        client: AsyncQdrantClient,
    ) -> None:
        self._client = client
        self._embeddings = embeddings
        self._collection = collection_name

    async def ensure_collection(self) -> None:
        if await self._client.collection_exists(self._collection):
            return

        await self._client.create_collection(
            self._collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="group_id",
            field_schema=KeywordIndexParams(type="keyword", is_tenant=True),
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="chat_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="created_at",
            field_schema=IntegerIndexParams(
                type=IntegerIndexType.INTEGER,
                lookup=False,
                range=True,
            ),
        )

    async def add(
        self,
        *,
        user_id: UUID,
        chat_id: UUID,
        turn_id: str,
        role: str,
        content: str,
    ) -> None:
        """Store one message. Idempotent: the same (user, chat, turn, role) is the same
        point, so a retried graph node overwrites it instead of adding a duplicate.

        `turn_id` must come from durable state (the id of the user's message), never
        be generated here - a retry would get a new one.
        """
        vector = await self._embeddings.aembed_query(content[:2000])
        point = PointStruct(
            id=str(uuid5(_POINT_NAMESPACE, f"{user_id}:{chat_id}:{turn_id}:{role}")),
            vector=vector,
            payload={
                "group_id": str(user_id),
                "chat_id": str(chat_id),
                "role": role,
                "content": content,
                "created_at": int(time.time() * 1_000_000),
            },
        )

        await self._client.upsert(self._collection, points=[point])

    async def recent(
        self, *, user_id: UUID, chat_id: UUID, limit: int = 20
    ) -> list[StoredMessage]:
        points, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="group_id", match=models.MatchValue(value=str(user_id))
                    ),
                    models.FieldCondition(
                        key="chat_id", match=models.MatchValue(value=str(chat_id))
                    ),
                ]
            ),
            order_by=models.OrderBy(key="created_at", direction=models.Direction.DESC),
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )

        return [StoredMessage(**p.payload) for p in reversed(points)]

    @staticmethod
    def _tenant_filter(user_id: UUID, chat_id: UUID | None = None) -> models.Filter:
        """group_id is always required - the only place tenant isolation lives."""
        must = [
            models.FieldCondition(
                key="group_id", match=models.MatchValue(value=str(user_id))
            )
        ]
        if chat_id is not None:
            must.append(
                models.FieldCondition(
                    key="chat_id", match=models.MatchValue(value=str(chat_id))
                )
            )
        return models.Filter(must=must)

    async def search(
        self,
        *,
        user_id: UUID,
        query: str,
        chat_id: UUID | None = None,
        limit: int = 5,
    ) -> list[StoredMessage]:
        """Semantic search over the user's messages (all chats unless chat_id given)."""
        vector = await self._embeddings.aembed_query(query[:2000])
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=self._tenant_filter(user_id, chat_id),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [StoredMessage(**p.payload) for p in response.points]

    async def delete_chat(self, *, user_id: UUID, chat_id: UUID) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=self._tenant_filter(user_id, chat_id)
            ),
        )
