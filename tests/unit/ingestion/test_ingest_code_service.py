"""Unit tests for IngestCodeService.

Tests the application service that orchestrates the ingestion pipeline.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion.application.ingest_code_service import IngestCodeService
from src.ingestion.domain.entities import CodeEntity, EntityType
from src.ingestion.domain.interfaces import IFileSystemLoader, IVectorStore


class TestIngestCodeService:
    """Test suite for IngestCodeService."""

    @pytest.fixture
    def mock_file_source(self):
        """Create a mock file source."""
        source = MagicMock(spec=IFileSystemLoader)
        return source

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock(spec=IVectorStore)
        return store

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="func1",
                content="def func1(): pass",
                file_path=Path("/test/file1.py"),
                start_line=1,
                end_line=2,
            ),
            CodeEntity(
                entity_type=EntityType.CLASS,
                name="MyClass",
                content="class MyClass: pass",
                file_path=Path("/test/file2.py"),
                start_line=1,
                end_line=2,
            ),
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="func2",
                content="def func2(): pass",
                file_path=Path("/test/file2.py"),
                start_line=5,
                end_line=6,
            ),
        ]

    def test_execute_success(self, mock_file_source, mock_vector_store, sample_entities):
        """Test successful ingestion execution."""
        # Setup
        mock_file_source.load_entities.return_value = sample_entities
        service = IngestCodeService(
            file_source=mock_file_source,
            vector_store=mock_vector_store,
        )

        # Execute
        stats = service.execute("/test/directory")

        # Verify
        assert stats["total_files"] == 2
        assert stats["total_entities"] == 3
        assert stats["entity_breakdown"]["FUNCTION"] == 2
        assert stats["entity_breakdown"]["CLASS"] == 1
        assert stats["duration_seconds"] > 0

        mock_file_source.load_entities.assert_called_once_with("/test/directory")
        mock_vector_store.save_entities.assert_called_once_with(sample_entities)

    def test_execute_empty_directory(self, mock_file_source, mock_vector_store):
        """Test execution with no files found."""
        # Setup
        mock_file_source.load_entities.return_value = []
        service = IngestCodeService(
            file_source=mock_file_source,
            vector_store=mock_vector_store,
        )

        # Execute
        stats = service.execute("/empty/directory")

        # Verify
        assert stats["total_files"] == 0
        assert stats["total_entities"] == 0
        assert stats["duration_seconds"] == 0.0

        mock_file_source.load_entities.assert_called_once()
        mock_vector_store.save_entities.assert_not_called()

    def test_execute_file_not_found(self, mock_file_source, mock_vector_store):
        """Test execution when directory doesn't exist."""
        # Setup
        mock_file_source.load_entities.side_effect = FileNotFoundError("Directory not found")
        service = IngestCodeService(
            file_source=mock_file_source,
            vector_store=mock_vector_store,
        )

        # Execute and verify
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            service.execute("/nonexistent")

    def test_execute_vector_store_error(self, mock_file_source, mock_vector_store, sample_entities):
        """Test execution when vector store fails."""
        # Setup
        mock_file_source.load_entities.return_value = sample_entities
        mock_vector_store.save_entities.side_effect = RuntimeError("Database error")
        service = IngestCodeService(
            file_source=mock_file_source,
            vector_store=mock_vector_store,
        )

        # Execute and verify
        with pytest.raises(RuntimeError, match="Database error"):
            service.execute("/test/directory")

    def test_analyze_entities(self, sample_entities):
        """Test entity analysis."""
        service = IngestCodeService(
            file_source=MagicMock(),
            vector_store=MagicMock(),
        )

        breakdown = service._analyze_entities(sample_entities)

        assert breakdown["FUNCTION"] == 2
        assert breakdown["CLASS"] == 1

    def test_analyze_entities_empty(self):
        """Test entity analysis with empty list."""
        service = IngestCodeService(
            file_source=MagicMock(),
            vector_store=MagicMock(),
        )

        breakdown = service._analyze_entities([])

        assert breakdown == {}
