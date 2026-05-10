"""Domain interfaces for agent orchestration.

These abstract interfaces define the contracts that agent implementations
must fulfill, enabling dependency inversion and easy swapping of agent
implementations or LLM providers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.ingestion.domain.entities import CodeEntity
from src.fix_agent_orchestration.domain.state import AgentState


class IAgent(ABC):
    """Abstract interface for all agents in the orchestration system."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent's name for logging and identification."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get a brief description of the agent's purpose."""
        pass

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's logic on the current state.

        The agent analyzes the current state, performs its specialized task,
        and returns an updated state with the results.

        Args:
            state: Current agent state with all accumulated context

        Returns:
            Updated agent state with new information

        Raises:
            AgentExecutionError: If agent fails to complete its task
        """
        pass


class ILLMClient(ABC):
    """Abstract interface for LLM client implementations."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of the LLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model being used."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> "LLMResponse":
        """Send a chat completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0-2), None for default
            max_tokens: Maximum tokens to generate, None for default
            stream: Whether to stream the response

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMError: If the request fails
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Chunks of the response as they're generated

        Raises:
            LLMError: If the request fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available.

        Returns:
            True if the service can be reached and is operational
        """
        pass


class IRAGSearch(ABC):
    """Abstract interface for RAG (Retrieval-Augmented Generation) search."""

    @abstractmethod
    async def search(
        self,
        query: str,
        keywords: list[str] | None = None,
        limit: int = 5,
        file_filter: str | None = None,
    ) -> list["SearchResult"]:
        """Search for relevant code entities.

        Performs hybrid search combining:
        - Semantic similarity on the query
        - Keyword matching for precision
        - Optional file path filtering

        Args:
            query: Semantic search query
            keywords: Optional keywords for boosting
            limit: Maximum results to return
            file_filter: Optional glob pattern for file filtering

        Returns:
            List of search results with relevance scores
        """
        pass


class IVectorStore(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        limit: int = 5,
        filter_dict: dict | None = None,
    ) -> list["CodeEntity"]:
        """Search for similar code entities.

        Args:
            query: The search query text
            limit: Maximum number of results
            filter_dict: Optional metadata filters

        Returns:
            List of matching code entities, ranked by relevance

        Raises:
            VectorStoreError: If search operation fails
        """
        pass

class LLMResponse:
    """Response from LLM client."""

    def __init__(
        self,
        content: str,
        provider: str,
        model: str,
        tokens_used: int | None = None,
        finish_reason: str | None = None,
        latency_ms: float | None = None,
    ):
        self.content = content
        self.provider = provider
        self.model = model
        self.tokens_used = tokens_used
        self.finish_reason = finish_reason
        self.latency_ms = latency_ms

    def __repr__(self) -> str:
        return f"LLMResponse(provider={self.provider}, model={self.model}, tokens={self.tokens_used})"


class SearchResult:
    """Result from RAG search operation."""

    def __init__(
        self,
        entity: CodeEntity,
        similarity_score: float,
        keyword_score: float = 0.0,
    ):
        self.entity = entity
        self.similarity_score = similarity_score
        self.keyword_score = keyword_score

    @property
    def combined_score(self) -> float:
        """Calculate combined relevance score."""
        # Weight semantic search higher (0.7) than keyword (0.3)
        return 0.7 * self.similarity_score + 0.3 * self.keyword_score

    def __repr__(self) -> str:
        return f"SearchResult({self.entity.qualified_name}, score={self.combined_score:.3f})"
