"""Unit tests for pattern detector.

Tests the pattern-based bug detection functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.discovery.infrastructure.pattern_detector import PatternDetector, AntiPatternRegistry
from src.detection.domain.entities import BugSeverity, CodeLocation


class TestPatternDetector:
    """Test suite for PatternDetector."""

    @pytest.fixture
    def detector(self):
        """Create PatternDetector instance."""
        return PatternDetector()

    def test_init(self, detector):
        """Test initialization of PatternDetector."""
        assert detector.name == "PatternDetector"
        assert detector.is_available is True

    def test_init(self, detector):
        """Test initialization."""
        assert detector.name == "PatternDetector"
        assert detector.is_available is True

    def test_is_available_always_true(self, detector):
        """Test that pattern detector is always available."""
        assert detector.is_available is True

