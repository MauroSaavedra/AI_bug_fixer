"""Domain entities for the ingestion slice.

This module defines the core data structures used throughout the ingestion
pipeline, from AST parsing to vector storage.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class EntityType(Enum):
    """Types of code entities that can be extracted from source files."""

    MODULE = auto()  # File-level module
    CLASS = auto()  # Class definition
    FUNCTION = auto()  # Standalone function
    METHOD = auto()  # Method within a class
    NESTED_FUNCTION = auto()  # Function defined inside another function
    STANDALONE = auto()  # Top-level statements (imports, assignments, etc.)


@dataclass(frozen=True)
class CodeEntity:
    """Represents a code entity extracted via AST parsing.

    This is the primary unit of storage in the vector database. Each entity
    represents a distinct, semantically coherent chunk of code that can be
    independently understood and fixed.

    Attributes:
        entity_type: The type of code entity (function, class, method, etc.)
        name: The identifier name of the entity (function name, class name)
        content: The actual source code text of this entity
        file_path: Absolute path to the source file
        start_line: 1-indexed line number where entity begins
        end_line: 1-indexed line number where entity ends (inclusive)
        parent: For nested entities (methods, inner functions), the parent's name
        docstring: The entity's docstring if present
        signature: For functions/methods, the signature line (def name(args):)
        imports: List of imports visible at this entity's scope
        metadata: Additional metadata for filtering and ranking
    """

    entity_type: EntityType
    name: str
    content: str
    file_path: Path
    start_line: int
    end_line: int
    parent: str | None = None
    docstring: str | None = None
    signature: str | None = None
    imports: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate entity after creation."""
        if self.start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {self.start_line}")
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name including parent if present.

        Examples:
            - standalone function: "my_function"
            - class method: "MyClass.my_method"
            - nested function: "outer.inner"
        """
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    @property
    def location(self) -> str:
        """Get human-readable location for this entity.

        Format: "path/to/file.py:123-145"
        """
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def line_count(self) -> int:
        """Calculate number of lines in this entity."""
        return self.end_line - self.start_line + 1

    @property
    def is_callable(self) -> bool:
        """Check if this entity is callable (function, method, or class)."""
        return self.entity_type in {
            EntityType.FUNCTION,
            EntityType.METHOD,
            EntityType.NESTED_FUNCTION,
            EntityType.CLASS,
        }

    def get_context_header(self) -> str:
        """Generate a header string for RAG context.

        This helps the LLM understand the entity's context when retrieved.
        """
        parts = [f"# {self.entity_type.name}: {self.qualified_name}"]
        if self.signature:
            parts.append(f"# Signature: {self.signature}")
        parts.append(f"# Location: {self.location}")
        if self.docstring:
            doc_preview = self.docstring[:100].replace("\n", " ")
            if len(self.docstring) > 100:
                doc_preview += "..."
            parts.append(f"# Docstring: {doc_preview}")
        return "\n".join(parts)

    def to_chroma_document(self) -> str:
        """Convert to document format optimized for ChromaDB storage.

        Combines context header with content for semantic search.
        """
        header = self.get_context_header()
        return f"{header}\n\n```python\n{self.content}\n```"

    def to_metadata(self) -> dict[str, Any]:
        """Convert to ChromaDB-compatible metadata dict.

        ChromaDB requires non-empty lists in metadata, so we filter those out.
        """
        metadata = {
            "entity_type": self.entity_type.name,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_path": str(self.file_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_count": self.line_count,
            "parent": self.parent if self.parent else "",
            "is_callable": self.is_callable,
        }

        # Add non-empty metadata
        if self.signature:
            metadata["signature"] = self.signature
        if self.docstring:
            metadata["docstring"] = self.docstring[:1000]  # Limit size
        if self.imports:
            metadata["imports"] = self.imports[:10]  # Limit to first 10 imports

        # Add custom metadata, filtering empty lists
        for key, value in self.metadata.items():
            if isinstance(value, list) and not value:
                continue  # Skip empty lists
            if value is not None:
                metadata[key] = value

        return metadata

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.qualified_name} ({self.entity_type.name}) at {self.location}"

    def __hash__(self) -> int:
        """Hash based on unique identifier."""
        return hash((str(self.file_path), self.start_line, self.end_line, self.name))


@dataclass(frozen=True)
class CodeChunk:
    """Legacy chunk entity for backward compatibility.

    Deprecated: Use CodeEntity instead for new code.
    This is kept for compatibility with existing vector stores.
    """

    content: str
    file_path: str
    start_line: int
    metadata: dict[str, Any]

    def to_entity(self) -> CodeEntity:
        """Convert legacy chunk to new CodeEntity format."""
        return CodeEntity(
            entity_type=EntityType.STANDALONE,
            name=self.metadata.get("filename", "unknown"),
            content=self.content,
            file_path=Path(self.file_path),
            start_line=self.start_line,
            end_line=self.metadata.get("end_line", self.start_line),
            metadata=self.metadata,
        )