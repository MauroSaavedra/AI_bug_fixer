"""Pydantic settings configuration for AgenticSource.

This module provides centralized configuration management using Pydantic Settings.
Environment variables are loaded from .env file for sensitive values like API keys.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Configuration priority (highest to lowest):
    1. Environment variables
    2. .env file values
    3. Default values defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Provider Configuration
    # ─────────────────────────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider to use: 'openai' or 'ollama'",
    )

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Ensure valid LLM provider."""
        allowed = {"openai", "ollama"}
        if v.lower() not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got '{v}'")
        return v.lower()

    # ─────────────────────────────────────────────────────────────────────────
    # OpenAI Configuration
    # ─────────────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (from .env)",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    openai_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for OpenAI",
    )
    openai_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens per response",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Ollama Configuration
    # ─────────────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model to use",
    )
    ollama_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for Ollama",
    )
    ollama_timeout: int = Field(
        default=120,
        ge=1,
        description="Request timeout in seconds",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Vector Store Configuration
    # ─────────────────────────────────────────────────────────────────────────
    chroma_db_path: Path = Field(
        default=Path("./chroma_db"),
        description="Path to ChromaDB persistence directory",
    )
    chroma_collection: str = Field(
        default="agentic_source_repo",
        description="ChromaDB collection name",
    )
    chroma_similarity_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar chunks to retrieve",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Orchestration Configuration
    # ─────────────────────────────────────────────────────────────────────────
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum review-reject retry cycles",
    )
    enable_streaming: bool = Field(
        default=False,
        description="Enable streaming responses (if supported)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
    result_folder: str = Field(
        default="report",
        description="Folder to save the fix result"
    )
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure valid logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper_v

    # ─────────────────────────────────────────────────────────────────────────
    # Bug Detection Configuration
    # ─────────────────────────────────────────────────────────────────────────
    detection_tools: list[str] = Field(
        default_factory=lambda: ["mypy", "pylint", "ruff"],
        description="Tools to use for bug detection",
    )
    detection_severity_threshold: str = Field(
        default="WARNING",
        description="Minimum severity to report: ERROR, WARNING, INFO",
    )

    @field_validator("detection_severity_threshold")
    @classmethod
    def validate_detection_threshold(cls, v: str) -> str:
        """Ensure valid severity threshold."""
        allowed = {"ERROR", "WARNING", "INFO"}
        upper_v = v.upper()
        if upper_v not in allowed:
            raise ValueError(f"threshold must be one of {allowed}, got '{v}'")
        return upper_v

    # ─────────────────────────────────────────────────────────────────────────
    # LangSmith Observability Configuration
    # ─────────────────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    langchain_api_key: str = Field(
        default="",
        description="LangSmith API key",
    )
    langchain_project: str = Field(
        default="agentic-source",
        description="LangSmith project name",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Using lru_cache ensures settings are loaded once and reused,
    avoiding repeated .env file reads.
    """
    return Settings()
