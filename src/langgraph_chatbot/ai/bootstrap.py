import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from urllib.parse import quote

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain.mcp import MCPAdapter
from langchain_openai import OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import AsyncQdrantClient

from langgraph_chatbot.ai.graph import build_graph
from langgraph_chatbot.ai.llm import build_llm
from langgraph_chatbot.ai.memory import UserMemory
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore
from langgraph_chatbot.config import Settings

logger = logging.getLogger(__name__)

# Only quick tools go to the model: crawl/research-style ones take minutes, which is
# useless in a chat and dead air in a voice call. The names of the tools each server
# really exposes are logged at startup - if one of these is wrong, the list is empty.
TAVILY_TOOLS = {"tavily_search", "tavily_extract"}
FIRECRAWL_TOOLS = {"firecrawl_scrape", "firecrawl_search"}


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
    # owns everything that must be closed: MCP adapters, Mem0, Qdrant
    stack: AsyncExitStack

    async def aclose(self) -> None:
        # closes in reverse order of creation; a failure in one callback does not
        # skip the others
        await self.stack.aclose()


async def _load_tools(
    stack: AsyncExitStack, name: str, target, allowed: set[str]
) -> list:
    """Connect one MCP server and return its tools; [] if it is unavailable.

    An MCP server being down must not take the chat (or a voice call) down with it.
    The log line carries only the error type: messages can contain the URL with the key.
    """
    try:
        adapter = await stack.enter_async_context(MCPAdapter(target))
        tools = [t for t in await adapter.list_tools() if t.name in allowed]
    except Exception as exc:
        logger.warning(
            "%s MCP unavailable (%s), continuing without it", name, type(exc).__name__
        )
        return []
    logger.info("%s MCP tools: %s", name, [t.name for t in tools])
    return tools


async def build_runtime(
    settings: Settings, *, ensure_collection: bool = True
) -> AppRuntime:
    llm = build_llm(settings)

    # Every resource goes into the stack right after it is created. If anything below
    # fails, `async with` closes what already exists (reverse order) and the original
    # error propagates. On success pop_all() hands the ownership to the runtime.
    async with AsyncExitStack() as stack:
        qdrant = AsyncQdrantClient(
            url=settings.vector_store_url,
            api_key=(
                settings.vector_store_api_key.get_secret_value()
                if settings.vector_store_api_key
                else None
            ),
        )
        stack.push_async_callback(qdrant.close)

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
        # registered after Qdrant, so it closes first: pending Mem0 writes finish
        # while the Qdrant client is still open
        stack.push_async_callback(user_memory.aclose)

        tools = []
        if settings.tavily_api_key:
            # Tavily takes the key as a query parameter
            key = quote(settings.tavily_api_key.get_secret_value(), safe="")
            url = f"{settings.tavily_mcp_url}?tavilyApiKey={key}"
            tools += await _load_tools(stack, "Tavily", url, TAVILY_TOOLS)
        if settings.firecrawl_api_key:
            firecrawl = Client(
                StreamableHttpTransport(
                    settings.firecrawl_mcp_url,
                    auth=settings.firecrawl_api_key.get_secret_value(),
                )
            )
            tools += await _load_tools(stack, "Firecrawl", firecrawl, FIRECRAWL_TOOLS)

        graph = build_graph(llm, messages_store, user_memory, tools)

        return AppRuntime(
            qdrant=qdrant,
            messages_store=messages_store,
            user_memory=user_memory,
            graph=graph,
            stack=stack.pop_all(),
        )
