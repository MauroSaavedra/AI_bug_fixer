"""Infrastructure layer for bug detection.

Contains concrete implementations of bug detection tools.
"""

from src.detection.infrastructure.static_analyzer import StaticAnalyzer
from src.detection.infrastructure.test_failure_parser import TestFailureParser

__all__ = ["StaticAnalyzer", "TestFailureParser"]
