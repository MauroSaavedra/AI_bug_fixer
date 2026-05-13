"""Ingestion layer domain interfaces.

These interfaces define the contracts for the ingestion pipeline,
enabling loose coupling between domain logic and infrastructure.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.ingestion.domain.entities import CodeChunk, CodeEntity


class IFileSystemLoader(ABC):
    """Interface for loading source code from the file system."""

    @abstractmethod
    def load_entities(self, directory_path: str) -> list[CodeEntity]:
        """Load and parse code entities from the given directory.

        Args:
            directory_path: The root directory to scan.

        Returns:
            A list of parsed code entities.
        """
        raise NotImplementedError


class IChunker(ABC):
    """Interface for chunking code into semantic entities."""

    @abstractmethod
    def chunk_file(self, file_path: Path, content: str) -> list[CodeEntity]:
        """Parse a file and extract semantic code entities.

        Args:
            file_path: Path to the source file.
            content: Raw content of the file.

        Returns:
            A list of code entities extracted from the file.
        """
        raise NotImplementedError


class IVectorStore(ABC):
    """Interface for a vector database that persists code entities."""

    @abstractmethod
    def save_chunks(self, chunks: list[CodeChunk]) -> None:
        """Persist legacy code chunks (backward compatibility).

        Args:
            chunks: List of CodeChunk objects
        """
        raise NotImplementedError

    @abstractmethod
    def save_entities(self, entities: list[CodeEntity]) -> None:
        """Persist AST-extracted entities.

        Args:
            entities: List of CodeEntity objects
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        limit: int = 5,
        filter_dict: dict | None = None,
    ) -> list[CodeEntity]:
        """Search for semantically similar code entities.

        Args:
            query: The search query text
            limit: Maximum number of results
            filter_dict: Optional metadata filters

        Returns:
            List of CodeEntity objects ranked by relevance
        """
        raise NotImplementedError

    @abstractmethod
    def get_collection_stats(self) -> dict:
        """Get statistics about the stored collection.

        Returns:
            Dictionary with collection metadata.
        """
        raise NotImplementedError
