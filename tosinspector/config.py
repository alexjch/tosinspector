"""Configuration settings for ToS Inspector using Pydantic BaseSettings."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    All Ollama endpoints, model names, chunking parameters, and retrieval settings
    are configurable here. Change these values in a .env file or via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Ollama server configuration
    ollama_base_url: str = Field(
        default="https://localhost:11434",
        description="Base URL for the Ollama server (use HTTPS for security)"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Model name for generating embeddings"
    )
    ollama_generation_model: str = Field(
        default="llama2",
        description="Model name for text generation/QA"
    )
    ollama_timeout: int = Field(
        default=60,
        description="Timeout in seconds for Ollama API requests"
    )
    ollama_verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates for HTTPS requests"
    )
    ollama_api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for Ollama requests"
    )

    # Chunking configuration
    chunk_size: int = Field(
        default=1000,
        description="Number of characters per chunk"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Number of overlapping characters between chunks"
    )

    # Retrieval configuration
    top_k: int = Field(
        default=5,
        description="Number of top chunks to retrieve for a query"
    )
    max_context_tokens: int = Field(
        default=4000,
        description="Maximum tokens allowed in the prompt context"
    )

    # Index storage
    index_dir: str = Field(
        default="indexes",
        description="Directory to store index manifests"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )


# Global settings instance - import this in other modules
settings = Settings()
