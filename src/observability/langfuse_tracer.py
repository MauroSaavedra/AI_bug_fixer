"""Langfuse tracer configuration and client management.

This module initializes and manages the Langfuse client based on application settings.
It provides a singleton tracer factory that gracefully handles missing or disabled configurations.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None  # type: ignore

from src.config.settings import Settings


class LangfuseTracer:
    """Manages Langfuse client and provides tracing utilities.

    Features:
    - Singleton pattern per configuration set.
    - Graceful degradation when tracing is disabled.
    - Centralized access to the Langfuse client.

    Usage:
        tracer = LangfuseTracer.get_instance(settings)
        if tracer.is_enabled:
            # Use tracer.langfuse to create traces/spans/generations
    """

    _instance: LangfuseTracer | None = None

    def __init__(self, settings: Settings) -> None:
        """Initialize the tracer with application settings.

        Args:
            settings: Pydantic settings object containing Langfuse configuration.
        """
        self._settings = settings
        self._langfuse_client: Any | None = None
        self._is_tracing_enabled = False

        if Langfuse is None:
            logger.warning("Langfuse package not installed. Observability disabled.")
            return

        # Validate configuration
        if not settings.langfuse_tracing:
            logger.debug("Langfuse tracing is disabled in settings.")
            return

        # Ensure credentials are present
        has_secret = bool(settings.langfuse_secret_key and settings.langfuse_secret_key.strip())
        has_public = bool(settings.langfuse_public_key and settings.langfuse_public_key.strip())

        if not has_secret or not has_public:
            logger.warning(
                "Langfuse credentials (LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY) are missing. "
                "Observability disabled."
            )
            return

        # Initialize Langfuse client
        try:
            self._langfuse_client = Langfuse(
                secret_key=settings.langfuse_secret_key,
                public_key=settings.langfuse_public_key,
                host=settings.langfuse_base_url,
                # enabled=True,
            )
            self._is_tracing_enabled = True
            logger.info(
                f"Langfuse observability enabled. Host: {settings.langfuse_base_url}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self._langfuse_client = None

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and the client is functional."""
        return self._is_tracing_enabled and self._langfuse_client is not None

    @property
    def langfuse(self) -> Any:
        """Get the raw Langfuse client for advanced use."""
        return self._langfuse_client

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> LangfuseTracer:
        """Get the singleton tracer instance.

        Args:
            settings: Optional settings; if not provided, global settings are fetched.

        Returns:
            The singleton LangfuseTracer instance.
        """
        if cls._instance is None:
            if settings is None:
                from src.config.settings import get_settings
                settings = get_settings()
            cls._instance = cls(settings)
        return cls._instance


def get_tracer() -> LangfuseTracer:
    """Factory function to get the global Langfuse tracer.

    Returns:
        Configured LangfuseTracer instance.
    """
    return LangfuseTracer.get_instance()
