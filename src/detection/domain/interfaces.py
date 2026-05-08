"""Domain interfaces for bug detection.

This module defines abstract interfaces that detection implementations
must fulfill, enabling dependency inversion and easy swapping.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.detection.domain.entities import DetectedBug, DetectionResult


class IBugDetector(ABC):
    """Abstract interface for bug detection tools.

    All bug detectors (mypy, pylint, test failure parser) implement
    this interface, providing a consistent API for the detection service.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get detector name for identification."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the detection tool is installed and available."""
        pass

    @abstractmethod
    async def detect(self, directory: str | Path) -> list[DetectedBug]:
        """Run detection on the given directory.

        Args:
            directory: Path to analyze

        Returns:
            List of detected bugs

        Raises:
            DetectionError: If detection fails
        """
        pass

    @abstractmethod
    async def detect_file(self, file_path: str | Path) -> list[DetectedBug]:
        """Run detection on a single file.

        Args:
            file_path: File to analyze

        Returns:
            List of detected bugs
        """
        pass


class IDetectionService(ABC):
    """Abstract interface for the detection service.

    The detection service aggregates results from multiple detectors
    and provides unified access to bug detection capabilities.
    """

    @abstractmethod
    async def detect_bugs(
        self,
        directory: str | Path,
        include_tests: bool = True,
    ) -> DetectionResult:
        """Run comprehensive bug detection.

        Args:
            directory: Directory to analyze
            include_tests: Whether to analyze test files

        Returns:
            Aggregated detection results
        """
        pass

    @abstractmethod
    def register_detector(self, detector: IBugDetector) -> None:
        """Register a new bug detector.

        Args:
            detector: Detector to add
        """
        pass
