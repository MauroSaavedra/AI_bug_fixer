"""Unit tests for configuration settings.

Tests the Pydantic Settings implementation for proper configuration
management and validation.
"""

import os
from pathlib import Path
from pydantic import ValidationError
import pytest

from src.config.settings import Settings


class TestSettings:
    """Test suite for Settings configuration."""

    def test_llm_provider_validation(self):
        """Test that invalid LLM provider is rejected."""
        with pytest.raises(ValidationError, match="llm_provider must be one of"):
            Settings(llm_provider="invalid")

    def test_llm_provider_case_insensitive(self):
        """Test that LLM provider is case insensitive."""
        settings = Settings(llm_provider="OPENAI")
        assert settings.llm_provider == "openai"

        settings = Settings(llm_provider="Ollama")
        assert settings.llm_provider == "ollama"

    def test_log_level_validation(self):
        """Test that invalid log level is rejected."""
        with pytest.raises(ValueError, match="log_level must be one of"):
            Settings(log_level="invalid")

    def test_log_level_normalization(self):
        """Test that log level is normalized to uppercase."""
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_temperature_range(self):
        """Test temperature validation."""
        # Valid range
        Settings(openai_temperature=0.5)
        Settings(openai_temperature=0.0)
        Settings(openai_temperature=2.0)

        # Out of range should fail
        with pytest.raises(ValueError):
            Settings(openai_temperature=-0.1)

        with pytest.raises(ValueError):
            Settings(openai_temperature=2.1)

    def test_max_retries_range(self):
        """Test max_retries validation."""
        Settings(max_retries=0)
        Settings(max_retries=10)

        with pytest.raises(ValueError):
            Settings(max_retries=-1)

        with pytest.raises(ValueError):
            Settings(max_retries=11)


class TestSettingsEnvironmentVariables:
    """Test settings loading from environment variables."""

    def test_openai_api_key_from_env(self, monkeypatch):
        """Test that OpenAI API key can be loaded from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        settings = Settings()
        assert settings.openai_api_key == "sk-from-env"

    def test_ollama_model_from_env(self, monkeypatch):
        """Test that Ollama model can be loaded from environment."""
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        settings = Settings()
        assert settings.ollama_model == "mistral"

    def test_llm_provider_from_env(self, monkeypatch):
        """Test that LLM provider can be loaded from environment."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        settings = Settings()
        assert settings.llm_provider == "ollama"

    def test_chroma_path_from_env(self, monkeypatch):
        """Test that ChromaDB path can be loaded from environment."""
        monkeypatch.setenv("CHROMA_DB_PATH", "/custom/path")
        settings = Settings()
        assert str(settings.chroma_db_path) == "/custom/path"
