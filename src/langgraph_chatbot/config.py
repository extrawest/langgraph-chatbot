from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_url: str
    vector_store_url: str
    vector_store_api_key: SecretStr | None = None

    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 30

    llm_api_key: SecretStr
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.5-flash-lite"

    embedding_model: str = "gemini-embedding-001"
    embedding_dims: int = 768

    livekit_url: str
    livekit_api_key: SecretStr | None = None
    livekit_api_secret: SecretStr | None = None

    eleven_voice_id: str
    eleven_api_key: SecretStr | None = None

    # hosted MCP servers; no API key = that server is switched off
    tavily_api_key: SecretStr | None = None
    tavily_mcp_url: str = "https://mcp.tavily.com/mcp/"
    firecrawl_api_key: SecretStr | None = None
    firecrawl_mcp_url: str = "https://mcp.firecrawl.dev/v2/mcp"


@lru_cache
def get_settings() -> Settings:
    return Settings()
