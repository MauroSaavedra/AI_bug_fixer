"""Unit tests for ingestion domain entities.

Tests CodeEntity and related classes for proper construction and behavior.
"""

from pathlib import Path

import pytest

from src.ingestion.domain.entities import CodeEntity, CodeChunk, EntityType


class TestCodeEntity:
    """Test suite for CodeEntity."""

    def test_basic_construction(self):
        """Test basic CodeEntity construction."""
        entity = CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="test_function",
            content="def test_function(): pass",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=2,
        )

        assert entity.entity_type == EntityType.FUNCTION
        assert entity.name == "test_function"
        assert entity.content == "def test_function(): pass"
        assert entity.start_line == 1
        assert entity.end_line == 2

    def test_line_validation(self):
        """Test that line numbers are validated."""
        # Valid
        CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="test",
            content="pass",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=1,
        )

        # Invalid: start_line < 1
        with pytest.raises(ValueError, match="start_line must be >= 1"):
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="test",
                content="pass",
                file_path=Path("/test.py"),
                start_line=0,
                end_line=1,
            )

        # Invalid: end_line < start_line
        with pytest.raises(ValueError, match="end_line .* must be >= start_line"):
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="test",
                content="pass",
                file_path=Path("/test.py"),
                start_line=5,
                end_line=3,
            )

    def test_qualified_name_with_parent(self):
        """Test qualified name when parent is present."""
        entity = CodeEntity(
            entity_type=EntityType.METHOD,
            name="my_method",
            content="def my_method(self): pass",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=2,
            parent="MyClass",
        )

        assert entity.qualified_name == "MyClass.my_method"

    def test_qualified_name_without_parent(self):
        """Test qualified name when no parent."""
        entity = CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="standalone_func",
            content="def standalone_func(): pass",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=2,
        )

        assert entity.qualified_name == "standalone_func"

    def test_location_property(self):
        """Test location property format."""
        entity = CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="test",
            content="pass",
            file_path=Path("/path/to/file.py"),
            start_line=10,
            end_line=20,
        )

        assert entity.location == "/path/to/file.py:10-20"

    def test_line_count(self):
        """Test line count calculation."""
        entity = CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="test",
            content="line1\nline2\nline3",
            file_path=Path("/test.py"),
            start_line=10,
            end_line=15,
        )

        assert entity.line_count == 6  # 15 - 10 + 1

    def test_is_callable(self):
        """Test is_callable property."""
        callable_types = [
            EntityType.FUNCTION,
            EntityType.METHOD,
            EntityType.NESTED_FUNCTION,
            EntityType.CLASS,
        ]

        for entity_type in callable_types:
            entity = CodeEntity(
                entity_type=entity_type,
                name="test",
                content="pass",
                file_path=Path("/test.py"),
                start_line=1,
                end_line=2,
            )
            assert entity.is_callable

        # Non-callable
        entity = CodeEntity(
            entity_type=EntityType.STANDALONE,
            name="test",
            content="x = 1",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=1,
        )
        assert not entity.is_callable

    def test_to_metadata(self):
        """Test metadata conversion."""
        entity = CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="test_func",
            content="def test_func(): pass",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=2,
            signature="def test_func()",
            docstring="Test docstring",
            metadata={"custom_key": "custom_value"},
        )

        metadata = entity.to_metadata()

        assert metadata["entity_type"] == "FUNCTION"
        assert metadata["name"] == "test_func"
        assert metadata["qualified_name"] == "test_func"
        assert metadata["file_path"] == "/test.py"
        assert metadata["start_line"] == 1
        assert metadata["end_line"] == 2
        assert metadata["line_count"] == 2
        assert metadata["signature"] == "def test_func()"
        assert metadata["docstring"] == "Test docstring"
        assert metadata["custom_key"] == "custom_value"

    def test_to_chroma_document(self):
        """Test ChromaDB document format."""
        entity = CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="test_func",
            content="def test_func():\n    return 42",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=2,
            signature="def test_func()",
        )

        doc = entity.to_chroma_document()

        assert "FUNCTION: test_func" in doc
        assert "Signature: def test_func()" in doc
        assert "```python" in doc
        assert "def test_func():" in doc


class TestCodeChunk:
    """Test suite for CodeChunk (legacy)."""

    def test_construction(self):
        """Test CodeChunk construction."""
        chunk = CodeChunk(
            content="def test(): pass",
            file_path="/test.py",
            start_line=1,
            metadata={"filename": "test.py"},
        )

        assert chunk.content == "def test(): pass"
        assert chunk.file_path == "/test.py"
        assert chunk.start_line == 1
        assert chunk.metadata["filename"] == "test.py"

    def test_to_entity_conversion(self):
        """Test conversion to CodeEntity."""
        chunk = CodeChunk(
            content="def test(): pass",
            file_path="/test.py",
            start_line=1,
            metadata={
                "filename": "test.py",
                "end_line": 2,
            },
        )

        entity = chunk.to_entity()

        assert entity.entity_type == EntityType.STANDALONE
        assert entity.name == "test.py"
        assert entity.content == "def test(): pass"
        assert entity.file_path == Path("/test.py")
        assert entity.start_line == 1
        assert entity.end_line == 2


class TestEntityType:
    """Test suite for EntityType enum."""

    def test_all_types(self):
        """Test all entity types exist."""
        types = [
            EntityType.MODULE,
            EntityType.CLASS,
            EntityType.FUNCTION,
            EntityType.METHOD,
            EntityType.NESTED_FUNCTION,
            EntityType.STANDALONE,
        ]

        for entity_type in types:
            assert isinstance(entity_type, EntityType)

    def test_name_property(self):
        """Test name property."""
        assert EntityType.FUNCTION.name == "FUNCTION"
        assert EntityType.CLASS.name == "CLASS"
