"""Unit tests for LocalFileSystemLoader.

Tests the file system loader infrastructure.
"""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.ingestion.domain.entities import CodeEntity, EntityType
from src.ingestion.infrastructure.local_file_system_loader import LocalFileSystemLoader


class TestLocalFileSystemLoader:
    """Test suite for LocalFileSystemLoader."""

    @pytest.fixture
    def ast_loader(self):
        """Create AST-based loader."""
        return LocalFileSystemLoader(
            supported_extensions=(".py", ".pyw"),
            use_ast_chunking=True,
        )

    @pytest.fixture
    def file_loader(self):
        """Create file-based loader (legacy mode)."""
        return LocalFileSystemLoader(
            supported_extensions=(".py",),
            use_ast_chunking=False,
        )

    def test_init_default_extensions(self):
        """Test initialization with default extensions."""
        loader = LocalFileSystemLoader()
        assert loader.supported_extensions == (".py", ".pyw")
        assert loader.use_ast_chunking is True

    def test_init_custom_extensions(self):
        """Test initialization with custom extensions."""
        loader = LocalFileSystemLoader(
            supported_extensions=(".py",),
            use_ast_chunking=False,
        )
        assert loader.supported_extensions == (".py",)
        assert loader.use_ast_chunking is False

    def test_is_supported_file(self, ast_loader):
        """Test file extension checking."""
        assert ast_loader._is_supported_file(Path("/test/file.py"))
        assert ast_loader._is_supported_file(Path("/test/file.pyw"))
        assert not ast_loader._is_supported_file(Path("/test/file.txt"))
        assert not ast_loader._is_supported_file(Path("/test/file.js"))

    def test_get_supported_extensions(self, ast_loader):
        """Test getting supported extensions."""
        extensions = ast_loader.get_supported_extensions()
        assert extensions == {".py", ".pyw"}

    def test_read_file_as_chunk(self, file_loader):
        """Test reading a file as a legacy chunk."""
        with patch("builtins.open", mock_open(read_data="def test(): pass")):
            chunk = file_loader._read_file_as_chunk(Path("/test.py"))

        assert chunk is not None
        assert chunk.content == "def test(): pass"
        assert chunk.file_path == "/test.py"
        assert chunk.start_line == 1
        assert chunk.metadata["filename"] == "test.py"
        assert chunk.metadata["extension"] == ".py"

    def test_read_file_as_chunk_unicode_error(self, file_loader):
        """Test handling of Unicode decode errors."""
        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")):
            chunk = file_loader._read_file_as_chunk(Path("/test.py"))

        assert chunk is None

    @patch("os.walk")
    def test_get_files_to_process_directory(self, mock_walk, ast_loader):
        """Test getting files from directory."""
        mock_walk.return_value = [
            ("/test", [], ["file1.py", "file2.txt", "file3.py"]),
            ("/test/subdir", [], ["file4.py", "file5.js"]),
        ]

        files = ast_loader._get_files_to_process(Path("/test"))

        assert len(files) == 3
        assert Path("/test/file1.py") in files
        assert Path("/test/file3.py") in files
        assert Path("/test/subdir/file4.py") in files

    def test_get_files_to_process_single_file(self, ast_loader):
        """Test getting files when path is a single file."""
        with patch.object(Path, "is_file", return_value=True):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(ast_loader, '_is_supported_file', return_value=True):
                    files = ast_loader._get_files_to_process(Path("/test/single.py"))
                    assert len(files) == 1
                    assert files[0] == Path("/test/single.py")

    def test_get_files_to_process_unsupported_file(self, ast_loader):
        """Test getting files when path is unsupported file."""
        files = ast_loader._get_files_to_process(Path("/test/file.txt"))
        assert len(files) == 0

    @patch.object(LocalFileSystemLoader, "_get_files_to_process")
    def test_load_entities_file_not_found(self, mock_get_files, ast_loader):
        """Test handling of missing directory."""
        mock_get_files.return_value = []

        with pytest.raises(FileNotFoundError):
            ast_loader.load_entities("/nonexistent")

    def test_load_files_legacy_mode(self, file_loader):
        """Test legacy file loading mode."""
        from src.ingestion.domain.entities import CodeChunk

        with patch.object(Path, "exists", return_value=True):
            with patch.object(file_loader, "_get_files_to_process", return_value=[Path("/test.py")]):
                with patch.object(file_loader, "_read_file_as_chunk") as mock_read:
                    mock_read.return_value = CodeChunk(
                        content="test",
                        file_path="/test.py",
                        start_line=1,
                        metadata={},
                    )

                    chunks = file_loader.load_files("/test")

        assert len(chunks) == 1
        mock_read.assert_called_once()
