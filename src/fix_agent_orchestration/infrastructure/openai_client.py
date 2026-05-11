import time
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from src.fix_agent_orchestration.domain.interfaces import ILLMClient, LLMResponse
from src.observability.langfuse_utils import update_current_generation


class OpenAIClient(ILLMClient):
    """OpenAI API client with async support.

    Features:
    - Async/await for non-blocking operations
    - Streaming responses for real-time output
    - Automatic retry with exponential backoff
    - Token usage tracking
    - Configurable temperature and max_tokens
    - Langfuse observability for prompt/response tracking
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o, gpt-4o-mini, etc.)
            base_url: Optional custom base URL (for Azure/proxy)
            timeout: Request timeout in seconds
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

        # Initialize async client
        client_kwargs = {
            "api_key": api_key,
            "timeout": httpx.Timeout(timeout),
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "openai"

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
        """Send chat completion request to OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMError: If request fails
        """
        start_time = time.time()

        # Build request parameters
        request_params = {
            "model": self._model,
            "messages": messages,  # type: ignore
        }

        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens

        try:
            if stream:
                # Handle streaming (not implemented for non-streaming method)
                content_chunks = []
                async for chunk in self.chat_stream(messages, temperature, max_tokens):
                    content_chunks.append(chunk)
                content = "".join(content_chunks)
                latency_ms = (time.time() - start_time) * 1000

                return LLMResponse(
                    content=content,
                    provider=self.provider_name,
                    model=self._model,
                    tokens_used=None,  # Not available in streaming
                    finish_reason="stop",
                    latency_ms=latency_ms,
                )
            else:
                # Non-streaming request
                response: ChatCompletion = await self._client.chat.completions.create(
                    **request_params
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract response data
                choice = response.choices[0]
                content = choice.message.content or ""

                llm_response = LLMResponse(
                    content=content,
                    provider=self.provider_name,
                    model=self._model,
                    tokens_used=response.usage.total_tokens if response.usage else None,
                    finish_reason=choice.finish_reason,
                    latency_ms=latency_ms,
                )

                # Update Langfuse generation (v4 API)
                try:
                    update_current_generation(
                        model=self._model,
                        model_parameters={
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "stream": stream,
                            "provider": "openai",
                        },
                        usage_details={
                            "total": response.usage.total_tokens if response.usage else None
                        } if response.usage else None,
                        metadata={
                            "latency_ms": latency_ms,
                            "finish_reason": choice.finish_reason,
                        },
                    )
                except Exception:
                    pass

                return llm_response

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion from OpenAI.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Content chunks as they're generated
        """
        request_params = {
            "model": self._model,
            "messages": messages,  # type: ignore
            "stream": True,
        }

        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens

        try:
            stream = await self._client.chat.completions.create(**request_params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming error: {e}") from e

    def is_available(self) -> bool:
        """Check if OpenAI service is available.

        Returns:
            True if API key is configured
        """
        return bool(self._api_key and self._api_key.strip().startswith("sk-"))

    async def close(self) -> None:
        """Close the client connection."""
        await self._client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()