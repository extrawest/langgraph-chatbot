from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_url: str

    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
