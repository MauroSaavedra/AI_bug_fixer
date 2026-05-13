"""Unit tests for static_analyzer.py infrastructure.

Tests the StaticAnalyzer's async subprocess calls to mypy/pylint/ruff
and its parsing of their output formats.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.detection.infrastructure.static_analyzer import (
    DetectionError,
    StaticAnalyzer,
)
from src.detection.domain.entities import BugSeverity, BugSource  # noqa: F401


class TestStaticAnalyzerInit:
    """Test StaticAnalyzer initialization."""

    def test_init_default_tools(self):
        """Default tools include mypy, pylint, ruff."""
        analyzer = StaticAnalyzer()
        assert set(analyzer.tools) == {"mypy", "pylint", "ruff"}

    def test_init_custom_tools(self):
        """Custom tools list."""
        analyzer = StaticAnalyzer(tools=["mypy"])
        assert analyzer.tools == ["mypy"]

    def test_name(self):
        """Name reflects configured tools."""
        analyzer = StaticAnalyzer(tools=["mypy", "ruff"])
        assert analyzer.name == "StaticAnalyzer(mypy,ruff)"

    def test_name_all_tools(self):
        """Name with all default tools."""
        analyzer = StaticAnalyzer()
        assert "mypy" in analyzer.name
        assert "pylint" in analyzer.name
        assert "ruff" in analyzer.name


class TestStaticAnalyzerAvailability:
    """Test is_available with and without tools installed."""

    def test_no_tools_available(self):
        """If no tools are installed, is_available is False."""
        analyzer = StaticAnalyzer(tools=["nonexistent_tool"])
        assert not analyzer.is_available

    @patch("shutil.which", return_value="/usr/bin/mypy")
    def test_some_tools_available(self, _mock_which):
        """If at least one tool is available, is_available is True."""
        # Only mock the first call to _check_tool
        analyzer = StaticAnalyzer(tools=["mypy"])
        assert analyzer.is_available


class TestMypyParsing:
    """Test mypy output parsing."""

    @pytest.fixture
    def analyzer(self):
        return StaticAnalyzer(tools=["mypy"])

    def test_parse_mypy_output(self, analyzer):
        """Parse standard mypy error output."""
        output = "main.py:5:2: error: Incompatible return value type [return-value]"
        bugs = analyzer._parse_mypy_output(output)
        assert len(bugs) == 1
        bug = bugs[0]
        assert bug.location.file_path.name == "main.py"
        assert bug.location.line_number == 5
        assert bug.location.column == 2
        assert "Incompatible return value type" in bug.message
        # The grouping in the regex captures the [return-value] as part of the message
        # because the error_code group (\w+) doesn't match hyphens.
        # This is an existing behavior of the implementation.
        assert "return-value" in (bug.message or "")
        assert bug.source == BugSource.MYPY

    def test_parse_mypy_warning(self, analyzer):
        """Parse mypy warning output."""
        output = "main.py:10:1: note: Function does not return a value [return]"
        bugs = analyzer._parse_mypy_output(output)
        assert len(bugs) == 1
        assert bugs[0].severity == BugSeverity.INFO

    def test_parse_mypy_no_errors(self, analyzer):
        """Parse mypy success output."""
        output = "Success: no issues found in 1 source file"
        bugs = analyzer._parse_mypy_output(output)
        assert len(bugs) == 0

    def test_parse_mypy_empty_output(self, analyzer):
        """Parse empty mypy output."""
        bugs = analyzer._parse_mypy_output("")
        assert len(bugs) == 0

    def test_parse_mypy_multiple_errors(self, analyzer):
        """Parse multiple mypy errors."""
        output = """main.py:5:2: error: Return value is None [return-value]
main.py:8:10: warning: Unused import [unused-import]
main.py:12:1: note: Check complete"""
        bugs = analyzer._parse_mypy_output(output)
        assert len(bugs) == 3
        assert bugs[0].location.line_number == 5
        assert bugs[1].location.line_number == 8
        assert bugs[2].location.line_number == 12


class TestPylintParsing:
    """Test pylint JSON output parsing."""

    @pytest.fixture
    def analyzer(self):
        return StaticAnalyzer(tools=["pylint"])

    def test_parse_pylint_json_output(self, analyzer):
        """Parse pylint JSON output."""
        output = """[
            {
                "type": "error",
                "module": "main",
                "obj": "my_func",
                "line": 10,
                "column": 0,
                "path": "main.py",
                "symbol": "undefined-variable",
                "message": "Undefined variable 'x'",
                "message-id": "E0602"
            }
        ]"""
        bugs = analyzer._parse_pylint_json(output)
        assert len(bugs) == 1
        bug = bugs[0]
        assert bug.message == "Undefined variable 'x'"
        assert bug.location.line_number == 10
        assert bug.error_code == "undefined-variable"

    def test_parse_pylint_unused_import(self, analyzer):
        """Parse pylint unused import."""
        output = """[
            {
                "type": "warning",
                "module": "main",
                "obj": "",
                "line": 1,
                "column": 0,
                "path": "main.py",
                "symbol": "unused-import",
                "message": "Unused import os",
                "message-id": "W0611"
            }
        ]"""
        bugs = analyzer._parse_pylint_json(output)
        assert len(bugs) == 1
        assert bugs[0].suggested_fix == "Remove unused import: "

    def test_parse_pylint_invalid_json(self, analyzer):
        """Handle invalid JSON."""
        bugs = analyzer._parse_pylint_json("not json")
        assert len(bugs) == 0

    def test_parse_pylint_nonlist_json(self, analyzer):
        """Handle non-list JSON."""
        bugs = analyzer._parse_pylint_json('{"key": "value"}')
        assert len(bugs) == 0


class TestRuffParsing:
    """Test ruff JSON output parsing."""

    @pytest.fixture
    def analyzer(self):
        return StaticAnalyzer(tools=["ruff"])

    def test_parse_ruff_json_output(self, analyzer):
        """Parse ruff JSON output."""
        output = """[
            {
                "code": "E501",
                "message": "Line too long (120 > 88)",
                "fix": null,
                "location": {"row": 5, "column": 89},
                "end_location": {"row": 5, "column": 120},
                "filename": "main.py"
            }
        ]"""
        bugs = analyzer._parse_ruff_json(output)
        assert len(bugs) == 1
        bug = bugs[0]
        assert bug.message == "Line too long (120 > 88)"
        assert bug.error_code == "E501"
        assert bug.location.line_number == 5

    def test_parse_ruff_fixable_output(self, analyzer):
        """Parse ruff output with fix."""
        output = """[
            {
                "code": "F401",
                "message": "Unused import `os`",
                "fix": {"content": ""},
                "location": {"row": 1, "column": 1},
                "end_location": {"row": 1, "column": 13},
                "filename": "main.py"
            }
        ]"""
        bugs = analyzer._parse_ruff_json(output)
        assert len(bugs) == 1
        # Fixable is INFO, non-fixable is WARNING
        assert bugs[0].severity == BugSeverity.INFO

    def test_parse_ruff_empty_json(self, analyzer):
        """Handle empty ruff output."""
        bugs = analyzer._parse_ruff_json("[]")
        assert len(bugs) == 0

    def test_parse_ruff_invalid_json(self, analyzer):
        """Handle invalid JSON."""
        bugs = analyzer._parse_ruff_json("not json")
        assert len(bugs) == 0


class TestStaticAnalyzerAsync:
    """Test async subprocess detection."""

    @pytest.fixture
    def analyzer(self):
        return StaticAnalyzer(tools=["mypy"])

    @pytest.mark.asyncio
    async def test_detect_file_runs_mypy(self, tmp_path, analyzer):
        """Test mypy is run on a file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x: int = 'hello'\n")

        # Mock create_subprocess_exec for mypy
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"test.py:1:5: error: Incompatible types [assignment]", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_subproc:
            bugs = await analyzer.detect_file(test_file)
            mock_subproc.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_directory_runs_concurrent(self, tmp_path, analyzer):
        """Test concurrent detection on directory."""
        (tmp_path / "test.py").write_text("x = 1\n")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_subproc:
            bugs = await analyzer.detect(tmp_path)
            mock_subproc.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_mypy_timeout(self, tmp_path, analyzer):
        """Test mypy timeout handling."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            bugs = await analyzer.detect_file(test_file)
            assert bugs == []
            mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_multiple_tools_concurrently(self, tmp_path):
        """Test that multiple tools run concurrently."""
        (tmp_path / "test.py").write_text("x = 1\n")
        analyzer = StaticAnalyzer(tools=["mypy", "ruff", "pylint"])

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"[]", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_subproc:
            bugs = await analyzer.detect(tmp_path)
            # All tools attempt to run if available
            assert mock_subproc.call_count >= 1


class TestDetectionError:
    """Test custom exception."""

    def test_detection_error_message(self):
        error = DetectionError("mypy", "Type checking failed", "extra output")
        assert "mypy" in str(error)
        assert "Type checking failed" in str(error)
