from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    google_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(
        "quizmate-notes-gemini-001",
        alias="PINECONE_INDEX_NAME",
    )
    pinecone_cloud: str = Field("aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field("us-east-1", alias="PINECONE_REGION")

    embedding_model: str = Field("models/gemini-embedding-001", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(3072, alias="EMBEDDING_DIMENSIONS")
    chat_model: str = Field("gemini-2.5-flash-lite", alias="CHAT_MODEL")
    pinecone_metric: str = Field("cosine", alias="PINECONE_METRIC")

    chunk_size_tokens: int = Field(800, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(100, alias="CHUNK_OVERLAP_TOKENS")
    upsert_batch_size: int = Field(100, alias="UPSERT_BATCH_SIZE")

    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR / ".env", ROOT_DIR / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
