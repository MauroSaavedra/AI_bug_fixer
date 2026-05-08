"""File system loader with AST-based chunking support.

This module implements IFileSystemLoader with support for both legacy file-based
chunking and modern AST-based semantic extraction.
"""

import os
from pathlib import Path
from loguru import logger
from src.ingestion.domain.entities import CodeChunk, CodeEntity
from src.ingestion.domain.interfaces import IFileSystemLoader
from src.ingestion.infrastructure.python_ast_chunker import PythonASTChunker


class LocalFileSystemLoader(IFileSystemLoader):
    """Load and parse code files from the local filesystem.

    This loader supports two modes:
    1. **AST-based chunking** (default): Uses PythonASTChunker to extract
       semantic entities (functions, classes, methods) with precise metadata.
    2. **Legacy file-based**: Reads entire files as single chunks.

    The loader automatically selects the appropriate chunker based on file type.
    """

    def __init__(
        self,
        supported_extensions: tuple[str, ...] | None = None,
        use_ast_chunking: bool = True,
    ):
        """Initialize the file system loader.

        Args:
            supported_extensions: Tuple of supported file extensions.
                Defaults to (".py", ".pyw") for Python files.
            use_ast_chunking: Whether to use AST-based semantic chunking.
                If False, falls back to file-level chunks.
        """
        if supported_extensions is None:
            supported_extensions = (".py", ".pyw")

        self.supported_extensions = supported_extensions
        self.use_ast_chunking = use_ast_chunking

        # Initialize AST chunker if needed
        self._ast_chunker = PythonASTChunker() if use_ast_chunking else None

    def load_files(self, path: str | Path) -> list[CodeChunk]:
        """Load code files using legacy file-based chunking.

        This method is kept for backward compatibility. For new code,
        use `load_entities()` instead.

        Args:
            path: Directory or file path to load

        Returns:
            List of code chunks (one per file)

        Raises:
            FileNotFoundError: If path does not exist
        """
        path = Path(path)
        chunks: list[CodeChunk] = []

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        files_to_process = self._get_files_to_process(path)

        for file_path in files_to_process:
            try:
                chunk = self._read_file_as_chunk(file_path)
                if chunk:
                    chunks.append(chunk)
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")

        return chunks

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
        """
        path = Path(path)
        entities: list[CodeEntity] = []

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        files_to_process = self._get_files_to_process(path)

        for file_path in files_to_process:
            try:
                if self.use_ast_chunking and self._ast_chunker:
                    file_entities = self._parse_file_with_ast(file_path)
                    entities.extend(file_entities)
                else:
                    # Fallback to file-level chunk converted to entity
                    chunk = self._read_file_as_chunk(file_path)
                    if chunk:
                        entities.append(chunk.to_entity())
            except SyntaxError as e:
                logger.warning(f"Syntax error in {file_path}: {e}")
                # Fallback to file-level chunking
                chunk = self._read_file_as_chunk(file_path)
                if chunk:
                    entities.append(chunk.to_entity())
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

        return entities

    def _get_files_to_process(self, path: Path) -> list[Path]:
        """Get list of files to process from the given path.

        If path is a directory, recursively finds all matching files.
        If path is a file, returns it directly.

        Args:
            path: Path to process

        Returns:
            List of file paths
        """
        if path.is_file():
            if self._is_supported_file(path):
                return [path]
            return []

        files: list[Path] = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                file_path = Path(root) / filename
                if self._is_supported_file(file_path):
                    files.append(file_path)

        return sorted(files)  # Consistent ordering

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file has a supported extension."""
        return any(str(file_path).endswith(ext) for ext in self.supported_extensions)

    def _read_file_as_chunk(self, file_path: Path) -> CodeChunk | None:
        """Read a file as a single legacy chunk.

        Args:
            file_path: Path to the file

        Returns:
            CodeChunk or None if file cannot be read
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Count lines for end_line
            line_count = content.count("\n") + 1

            return CodeChunk(
                content=content,
                file_path=str(file_path),
                start_line=1,
                metadata={
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "end_line": line_count,
                    "is_file_level": True,
                },
            )
        except UnicodeDecodeError:
            # Skip binary files
            logger.warning(f"Skipping binary file: {file_path}")
            return None
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return None

    def _parse_file_with_ast(self, file_path: Path) -> list[CodeEntity]:
        """Parse a file using AST-based chunking.

        Args:
            file_path: Path to the Python file

        Returns:
            List of extracted code entities
        """
        if not self._ast_chunker:
            raise RuntimeError("AST chunker not initialized")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return self._ast_chunker.chunk_file(file_path, content)

    def get_supported_extensions(self) -> set[str]:
        """Get the set of supported file extensions."""
        return set(self.supported_extensions)
