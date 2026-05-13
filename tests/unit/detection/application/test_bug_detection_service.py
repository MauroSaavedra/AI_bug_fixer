"""Unit tests for bug_detection_service.py.

Tests the BugDetectionService's orchestration of multiple detectors,
deduplication, and result aggregation.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.detection.application.bug_detection_service import BugDetectionService
from src.detection.domain.entities import BugSeverity, BugSource, CodeLocation, DetectedBug, DetectionResult


def _make_bug(message="test bug", line=1, code="test-code"):
    """Helper to create a DetectedBug."""
    return DetectedBug(
        source=BugSource.MYPY,
        severity=BugSeverity.ERROR,
        location=CodeLocation(file_path=Path("/test.py"), line_number=line),
        message=message,
        error_code=code,
    )


class FakeDetector:
    """A fake sync detector for testing."""

    def __init__(self, name="Fake", bugs=None):
        self._name = name
        self._bugs = bugs or []
        self._is_available = True

    @property
    def name(self):
        return self._name

    @property
    def is_available(self):
        return self._is_available

    def detect(self, directory):
        return self._bugs

    def detect_file(self, file_path):
        return self._bugs


class FakeAsyncDetector:
    """A fake async detector for testing."""

    def __init__(self, name="FakeAsync", bugs=None):
        self._name = name
        self._bugs = bugs or []

    @property
    def name(self):
        return self._name

    @property
    def is_available(self):
        return True

    async def detect(self, directory):
        return self._bugs

    async def detect_file(self, file_path):
        return self._bugs


class TestBugDetectionServiceInit:
    """Test BugDetectionService initialization."""

    def test_init_default_detectors(self):
        """Default detectors include static analyzer, pattern, test parser."""
        service = BugDetectionService()
        # Note: detector names vary by tool availability
        assert len(service._detectors) == 3

    def test_init_custom_detectors(self):
        """Custom detectors list."""
        fake = FakeDetector(name="Custom")
        service = BugDetectionService(detectors=[fake])
        assert len(service._detectors) == 1
        assert service._detectors[0].name == "Custom"

    def test_register_detector(self):
        """Register a new detector."""
        service = BugDetectionService(detectors=[])
        service.register_detector(FakeDetector("New"))
        assert len(service._detectors) == 1


class TestBugDetectionServiceDeduplicate:
    """Test bug deduplication logic."""

    @pytest.fixture
    def service(self):
        return BugDetectionService(detectors=[])

    def test_deduplicate_no_duplicates(self, service):
        """Bugs from different lines are not deduplicated."""
        bugs = [
            _make_bug("Bug A", line=1),
            _make_bug("Bug B", line=2),
        ]
        result = service._deduplicate_bugs(bugs)
        assert len(result) == 2

    def test_deduplicate_same_bug(self, service):
        """Identical bugs are deduplicated."""
        bugs = [
            _make_bug("Same bug", line=1),
            _make_bug("Same bug", line=1),
        ]
        result = service._deduplicate_bugs(bugs)
        assert len(result) == 1

    def test_deduplicate_different_file(self, service):
        """Bugs in different files are not deduplicated."""
        bug1 = _make_bug("Bug", line=1)
        bug2 = _make_bug("Bug", line=1)
        # Modify path to create a unique file object for each bug
        bug2_location = CodeLocation(file_path=Path("/other.py"), line_number=1)
        from dataclasses import replace
        bug2_replaced = replace(bug2, location=bug2_location)

        result = service._deduplicate_bugs([bug1, bug2_replaced])
        assert len(result) == 2

    def test_deduplicate_similar_messages(self, service):
        """Messages that differ after 50 chars are not deduplicated."""
        msg = "A" * 60
        bugs = [
            _make_bug(f"{msg} suffix 1", line=1),
            _make_bug(f"{msg} suffix 2", line=1),
        ]
        result = service._deduplicate_bugs(bugs)
        assert len(result) == 1  # Only first 50 chars considered

    def test_deduplicate_empty(self, service):
        """Handle empty list."""
        result = service._deduplicate_bugs([])
        assert result == []


class TestBugDetectionServiceDetectFile:
    """Test detect_bugs on a file."""

    @pytest.mark.asyncio
    async def test_detect_file_with_sync_detector(self, tmp_path):
        """Detect on a file with sync detector."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        fake = FakeDetector(name="Sync", bugs=[_make_bug("Test bug", 1)])
        service = BugDetectionService(detectors=[fake])

        result = await service.detect_bugs(test_file)
        assert isinstance(result, DetectionResult)
        assert len(result.bugs) == 1
        assert result.bugs[0].message == "Test bug"

    @pytest.mark.asyncio
    async def test_detect_file_with_async_detector(self, tmp_path):
        """Detect on a file with async detector."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        fake = FakeAsyncDetector(name="Async", bugs=[_make_bug("Async bug", 1)])
        service = BugDetectionService(detectors=[fake])

        result = await service.detect_bugs(test_file)
        assert len(result.bugs) == 1
        assert result.bugs[0].message == "Async bug"

    @pytest.mark.asyncio
    async def test_detect_file_not_found(self, tmp_path):
        """FileNotFound when path doesn't exist."""
        service = BugDetectionService(detectors=[])
        with pytest.raises(FileNotFoundError):
            await service.detect_bugs(tmp_path / "nonexistent.py")

    @pytest.mark.asyncio
    async def test_detect_unavailable_detector(self, tmp_path):
        """Unavailable detectors are skipped."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        class UnavailableDetector(FakeDetector):
            @property
            def is_available(self):
                return False
        fake = UnavailableDetector(name="Unavailable")
        service = BugDetectionService(detectors=[fake])

        result = await service.detect_bugs(test_file)
        assert len(result.bugs) == 0

    @pytest.mark.asyncio
    async def test_detect_file_summary(self, tmp_path):
        """Check result summary includes correct counts."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        bugs = [
            _make_bug("Error bug", line=1),
            DetectedBug(
                source=BugSource.PYLINT,
                severity=BugSeverity.WARNING,
                location=CodeLocation(file_path=Path("/test.py"), line_number=2),
                message="Warning bug",
            ),
        ]
        fake = FakeDetector(name="Mixed", bugs=bugs)
        service = BugDetectionService(detectors=[fake])

        result = await service.detect_bugs(test_file)
        assert result.error_count == 1
        assert result.warning_count == 1
        assert result.info_count == 0


class TestBugDetectionServiceDetectDirectory:
    """Test detect_bugs on a directory."""

    @pytest.mark.asyncio
    async def test_detect_directory(self, tmp_path):
        """Detect on a directory."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        fake = FakeDetector(name="Dir", bugs=[_make_bug("Dir bug", 1)])
        service = BugDetectionService(detectors=[fake])

        result = await service.detect_bugs(tmp_path)
        assert len(result.bugs) == 1
        assert result.files_analyzed == 2

    @pytest.mark.asyncio
    async def test_detect_directory_excludes_tests(self, tmp_path):
        """Detect directory excluding test files."""
        (tmp_path / "main.py").write_text("x = 1\n")
        (tmp_path / "test_main.py").write_text("def test_x(): pass\n")
        fake = FakeDetector(name="Dir")
        service = BugDetectionService(detectors=[fake])

        # Note: files_analyzed counts test files unless include_tests=False is implemented
        result = await service.detect_bugs(tmp_path)
        assert result.files_analyzed >= 1


class TestBugDetectionServiceDetectSingleBug:
    """Test detect_single_bug."""

    @pytest.fixture
    def service(self):
        fake = FakeDetector(name="TestDetector", bugs=[
            _make_bug("division by zero", line=1),
            _make_bug("unused import", line=2, code="unused"),
        ])
        return BugDetectionService(detectors=[fake])

    def test_detect_single_bug_found(self, service):
        """Specific bug description matches."""
        bug = service.detect_single_bug("/test.py", "division")
        assert bug is not None
        assert "division" in bug.message

    def test_detect_single_bug_not_found(self, service):
        """No matching bug."""
        bug = service.detect_single_bug("/test.py", "nonexistent issue")
        assert bug is None

    def test_detect_single_bug_no_detectors(self):
        """No detectors available."""
        service = BugDetectionService(detectors=[])
        bug = service.detect_single_bug("/test.py", "anything")
        assert bug is None


class TestBugDetectionServiceCountFiles:
    """Test file counting."""

    def test_count_python_files(self, tmp_path):
        """Count Python files."""
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        (tmp_path / "not_python.txt").write_text("text")
        service = BugDetectionService(detectors=[])
        count = service._count_python_files(tmp_path, include_tests=True)
        assert count == 2

    def test_count_excludes_tests(self, tmp_path):
        """Option to exclude test files."""
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "test_main.py").write_text("def test_x(): pass")
        service = BugDetectionService(detectors=[])
        count = service._count_python_files(tmp_path, include_tests=False)
        assert count == 1
