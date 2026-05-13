"""Test failure detection and parsing.

This module captures and parses test failures from pytest output,
mapping them to code locations for automatic fixing.
"""

import re
from pathlib import Path
from typing import Any
from loguru import logger
import xml.etree.ElementTree as ET

from src.detection.domain.entities import (
    BugSeverity,
    BugSource,
    CodeLocation,
    DetectedBug,
)
from src.detection.domain.interfaces import IBugDetector


class TestFailureParser(IBugDetector):
    """Parser for test failure output.

    Supports multiple formats:
    - pytest terminal output (with --tb=short)
    - JUnit XML output (--junitxml=results.xml)
    - pytest JSON output (with pytest-json-report plugin)
    """

    def __init__(self, test_output_path: str | Path | None = None):
        """Initialize the parser.

        Args:
            test_output_path: Path to existing test output file.
                If None, assumes pytest output will be provided.
        """
        self.test_output_path = Path(test_output_path) if test_output_path else None

    @property
    def name(self) -> str:
        """Get detector name."""
        return "TestFailureParser"

    @property
    def is_available(self) -> bool:
        """Check if test output is available."""
        if self.test_output_path:
            return self.test_output_path.exists()
        return True  # Will try to parse from provided text

    def detect(self, directory: str | Path) -> list[DetectedBug]:
        """Parse test failures from output file.

        Args:
            directory: Directory containing test output

        Returns:
            List of detected bugs from test failures
        """
        if self.test_output_path and self.test_output_path.exists():
            if self.test_output_path.suffix == ".xml":
                return self._parse_junit_xml(self.test_output_path)
            elif self.test_output_path.suffix == ".json":
                return self._parse_pytest_json(self.test_output_path)
            else:
                return self._parse_pytest_text(self.test_output_path)

        directory = Path(directory)
        bugs: list[DetectedBug] = []

        # Check for JUnit XML
        junit_file = directory / "junit.xml"
        if junit_file.exists():
            bugs.extend(self._parse_junit_xml(junit_file))

        # Check for test results
        results_file = directory / "test-results.xml"
        if results_file.exists():
            bugs.extend(self._parse_junit_xml(results_file))

        return bugs

    def detect_file(self, file_path: str | Path) -> list[DetectedBug]:
        """Parse test failures from a specific file.

        Args:
            file_path: Path to test output file

        Returns:
            List of detected bugs
        """
        file_path = Path(file_path)

        if file_path.suffix == ".xml":
            return self._parse_junit_xml(file_path)
        elif file_path.suffix == ".json":
            return self._parse_pytest_json(file_path)
        else:
            return self._parse_pytest_text(file_path)

    def _parse_junit_xml(self, file_path: Path) -> list[DetectedBug]:
        """Parse JUnit XML test results."""
        bugs: list[DetectedBug] = []

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Find all failed test cases
            for testcase in root.iter("testcase"):
                failure = testcase.find("failure")
                if failure is not None:
                    # Extract test name and failure info
                    test_name = testcase.get("name", "unknown")
                    file_path_str = testcase.get("file", "")
                    line_num = int(testcase.get("line", 0))

                    message = failure.get("message", "Test failed")
                    details = failure.text or ""

                    # Try to extract line number from details
                    if not line_num and details:
                        line_match = re.search(r"File \"([^\"]+)\", line (\d+)", details)
                        if line_match:
                            file_path_str = line_match.group(1)
                            line_num = int(line_match.group(2))

                    bug = DetectedBug(
                        source=BugSource.PYTEST,
                        severity=BugSeverity.ERROR,
                        location=CodeLocation(
                            file_path=Path(file_path_str).resolve() if file_path_str else Path("unknown"),
                            line_number=line_num or 1,
                        ),
                        message=f"Test failure: {test_name} - {message}",
                        error_code="test-failure",
                        code_snippet=details[:500] if details else None,
                        tool_output=details[:1000],
                    )
                    bugs.append(bug)

                # Also check for errors (different from failures)
                error = testcase.find("error")
                if error is not None:
                    test_name = testcase.get("name", "unknown")
                    file_path_str = testcase.get("file", "")
                    line_num = int(testcase.get("line", 0))

                    message = error.get("message", "Test error")

                    bug = DetectedBug(
                        source=BugSource.PYTEST,
                        severity=BugSeverity.ERROR,
                        location=CodeLocation(
                            file_path=Path(file_path_str).resolve() if file_path_str else Path("unknown"),
                            line_number=line_num or 1,
                        ),
                        message=f"Test error: {test_name} - {message}",
                        error_code="test-error",
                        tool_output=error.text[:1000] if error.text else None,
                    )
                    bugs.append(bug)

        except ET.ParseError as e:
            logger.error(f"Failed to parse JUnit XML: {e}")

        return bugs

    def _parse_pytest_json(self, file_path: Path) -> list[DetectedBug]:
        """Parse pytest JSON report."""
        import json

        bugs: list[DetectedBug] = []

        try:
            with open(file_path) as f:
                report = json.load(f)

            # Parse test results
            for test in report.get("tests", []):
                if test.get("outcome") == "failed":
                    # Extract failure info
                    call = test.get("call", {})
                    crash = call.get("crash", {})

                    file_path_str = crash.get("path", "")
                    line_num = crash.get("lineno", 1)
                    message = call.get("longrepr", "Test failed")

                    bug = DetectedBug(
                        source=BugSource.PYTEST,
                        severity=BugSeverity.ERROR,
                        location=CodeLocation(
                            file_path=Path(file_path_str).resolve() if file_path_str else Path("unknown"),
                            line_number=line_num,
                        ),
                        message=f"Test failure: {test.get('nodeid', 'unknown')} - {message[:200]}",
                        error_code="test-failure",
                        code_snippet=message[:500],
                        tool_output=json.dumps(test),
                    )
                    bugs.append(bug)

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to parse pytest JSON: {e}")

        return bugs

    def _parse_pytest_text(self, file_path: Path) -> list[DetectedBug]:
        """Parse pytest text output."""
        try:
            with open(file_path) as f:
                text = f.read()
            return self._parse_pytest_output(text)
        except IOError as e:
            logger.error(f"Failed to read test output: {e}")
            return []

    def _parse_pytest_output(self, text: str) -> list[DetectedBug]:
        """Parse pytest terminal output.

        Extracts failure information from pytest's short traceback format.
        """
        bugs: list[DetectedBug] = []

        # Pattern for failed tests
        # Format: FAILED path/to/test_file.py::test_name - message
        failed_pattern = r"FAILED\s+(\S+)::(\S+)\s+-\s+(.+?)(?=\n(?:FAILED|PASSED|ERROR)|\Z)"

        for match in re.finditer(failed_pattern, text, re.DOTALL):
            file_path_str = match.group(1)
            test_name = match.group(2)
            message = match.group(3).strip()

            # Try to find line number in traceback
            line_num = 1
            traceback_section = text[match.end():]
            line_match = re.search(
                rf"File \"[^\"]*{re.escape(file_path_str)}\", line (\d+)",
                traceback_section[:2000],
            )
            if line_match:
                line_num = int(line_match.group(1))

            bug = DetectedBug(
                source=BugSource.PYTEST,
                severity=BugSeverity.ERROR,
                location=CodeLocation(
                    file_path=Path(file_path_str).resolve(),
                    line_number=line_num,
                ),
                message=f"Test failure: {test_name} - {message[:200]}",
                error_code="test-failure",
                tool_output=match.group(0),
            )
            bugs.append(bug)

        # Pattern for errors (setup/teardown failures)
        error_pattern = r"ERROR\s+(\S+)::(\S+)\s+-\s+(.+?)(?=\n(?:FAILED|PASSED|ERROR)|\Z)"

        for match in re.finditer(error_pattern, text, re.DOTALL):
            file_path_str = match.group(1)
            test_name = match.group(2)
            message = match.group(3).strip()

            bug = DetectedBug(
                source=BugSource.PYTEST,
                severity=BugSeverity.ERROR,
                location=CodeLocation(
                    file_path=Path(file_path_str).resolve(),
                    line_number=1,
                ),
                message=f"Test error: {test_name} - {message[:200]}",
                error_code="test-error",
                tool_output=match.group(0),
            )
            bugs.append(bug)

        return bugs

    def extract_stack_frame(self, traceback_text: str, frame_index: int = 0) -> dict[str, Any] | None:
        """Extract a specific stack frame from traceback.

        Args:
            traceback_text: Full traceback text
            frame_index: Which frame to extract (0 = most recent)

        Returns:
            Dictionary with file, line, function info or None
        """
        # Pattern: File "path", line N, in function
        pattern = r'File "([^"]+)", line (\d+), in (\w+)'
        matches = list(re.finditer(pattern, traceback_text))

        if frame_index < len(matches):
            match = matches[frame_index]
            return {
                "file": match.group(1),
                "line": int(match.group(2)),
                "function": match.group(3),
            }

        return None
