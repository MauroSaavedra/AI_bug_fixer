"""Domain interfaces for the ingestion slice.

These abstract interfaces define contracts that infrastructure implementations
must fulfill, enabling dependency inversion and easy swapping of implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.ingestion.domain.entities import CodeChunk, CodeEntity


class IVectorStore(ABC):
    """Abstract interface for vector database operations.

    This interface abstracts the underlying vector store implementation,
    allowing the application to work with any compatible vector database.
    """

    @abstractmethod
    async def save_chunks(self, chunks: list[CodeChunk]) -> None:
        """Persist chunks to the vector store.

        Args:
            chunks: List of code chunks to store

        Raises:
            VectorStoreError: If storage operation fails
        """
        pass

    @abstractmethod
    async def save_entities(self, entities: list[CodeEntity]) -> None:
        """Persist AST-extracted entities to the vector store.

        This is the preferred method for storing code with AST-based chunking.

        Args:
            entities: List of code entities to store

        Raises:
            VectorStoreError: If storage operation fails
        """
        pass

    @abstractmethod
    async def similarity_search(
        self, query: str, limit: int = 5, filter_dict: dict | None = None
    ) -> list[CodeEntity]:
        """Search for similar code entities.

        Args:
            query: The search query text
            limit: Maximum number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of matching code entities, ranked by relevance

        Raises:
            VectorStoreError: If search operation fails
        """
        pass

    @abstractmethod
    async def get_collection_stats(self) -> dict:
        """Get statistics about the stored collection.

        Returns:
            Dictionary with collection metadata (count, dimensions, etc.)
        """
        pass


class IFileSystemLoader(ABC):
    """Abstract interface for file system operations.

    Abstracts the file reading logic, allowing for testing with mock
    file systems or remote file sources.
    """

    @abstractmethod
    def load_files(self, path: str | Path) -> list[CodeChunk]:
        """Load code files from the given path.

        Args:
            path: Directory or file path to load

        Returns:
            List of code chunks extracted from files

        Raises:
            FileNotFoundError: If path does not exist
            PermissionError: If files cannot be read
        """
        pass

    @abstractmethod
    def load_entities(self, path: str | Path) -> list[CodeEntity]:
        """Load and parse code files using AST-based extraction.

        This is the preferred method for loading code with full semantic
        understanding of functions, classes, and methods.

        Args:
            path: Directory or file path to load

        Returns:
            List of code entities with complete metadata

        Raises:
            FileNotFoundError: If path does not exist
            PermissionError: If files cannot be read
            SyntaxError: If Python files contain syntax errors
        """
        pass


class IChunker(ABC):
    """Abstract interface for code chunking strategies.

    Enables different chunking approaches (AST-based, token-based, etc.)
    to be used interchangeably.
    """

    @abstractmethod
    def chunk_file(self, file_path: Path, content: str) -> list[CodeEntity]:
        """Extract code entities from a single file.

        Args:
            file_path: Path to the source file
            content: File content as string

        Returns:
            List of code entities extracted from the file

        Raises:
            SyntaxError: If file cannot be parsed
            ValueError: If content is invalid
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> set[str]:
        """Get file extensions supported by this chunker.

        Returns:
            Set of supported extensions (e.g., {'.py', '.pyw'})
        """
        pass