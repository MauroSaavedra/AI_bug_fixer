import time
from typing import AsyncIterator
import ollama
from ollama import AsyncClient

from src.agent_orchestration.domain.interfaces import ILLMClient, LLMResponse


class OllamaClient(ILLMClient):
    """Ollama API client for local models.

    Features:
    - Local model inference (no API keys needed)
    - Support for models like Llama 3.2, CodeLlama, etc.
    - Async operations with configurable timeout
    - Streaming support
    - Automatic fallback handling

    Note: Requires Ollama server running locally at the configured base_url.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        """Initialize Ollama client.

        Args:
            model: Model name (llama3.2, codellama, etc.)
            base_url: Ollama server URL
            timeout: Request timeout in seconds

        Note:
            The model must be pulled locally before use:
            `ollama pull llama3.2`
        """
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

        # Initialize async client
        self._client = AsyncClient(host=base_url)

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "ollama"

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Send chat completion request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate (mapped to Ollama options)
            stream: Whether to stream response

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMError: If request fails or model not found
        """
        start_time = time.time()

        # Build options
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens  # Ollama's term for max_tokens

        # Convert messages to Ollama format (already compatible)
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]

        try:
            if stream:
                # Handle streaming by collecting all chunks
                content_chunks = []
                async for chunk in self.chat_stream(messages, temperature, max_tokens):
                    content_chunks.append(chunk)
                content = "".join(content_chunks)
                latency_ms = (time.time() - start_time) * 1000

                return LLMResponse(
                    content=content,
                    provider=self.provider_name,
                    model=self._model,
                    tokens_used=None,
                    finish_reason="stop",
                    latency_ms=latency_ms,
                )
            else:
                # Non-streaming request
                response = await self._client.chat(
                    model=self._model,
                    messages=ollama_messages,  # type: ignore
                    options=options if options else None,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract content
                content = response.message.content if response.message else ""

                return LLMResponse(
                    content=content,
                    provider=self.provider_name,
                    model=self._model,
                    tokens_used=None,
                    finish_reason="stop",
                    latency_ms=latency_ms,
                )

        except ollama.ResponseError as e:
            if "model not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{self._model}' not found. Pull it first: ollama pull {self._model}"
                ) from e
            raise RuntimeError(f"Ollama API error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}") from e

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion from Ollama.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Content chunks as they're generated
        """
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]

        try:
            stream = await self._client.chat(
                model=self._model,
                messages=ollama_messages,  # type: ignore
                options=options if options else None,
                stream=True,
            )

            async for chunk in stream:
                if chunk.message and chunk.message.content:
                    yield chunk.message.content

        except Exception as e:
            raise RuntimeError(f"Ollama streaming error: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama service is available.

        Returns:
            True if Ollama server is reachable
        """
        try:
            # Try to list models as a health check
            import httpx

            response = httpx.get(
                f"{self._base_url}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models on the Ollama server.

        Returns:
            List of model names
        """
        try:
            models = await self._client.list()
            return [m.model for m in models.models] if models.models else []
        except Exception:
            return []

    async def close(self) -> None:
        """Close the client (no-op for Ollama)."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
