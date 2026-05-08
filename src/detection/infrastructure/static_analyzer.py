"""Static analysis infrastructure for bug detection.

This module implements bug detection using static analysis tools
like mypy, pylint, and ruff.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from loguru import logger

from src.detection.domain.entities import (
    BugSeverity,
    BugSource,
    CodeLocation,
    DetectedBug,
)
from src.detection.domain.interfaces import IBugDetector


class StaticAnalyzer(IBugDetector):
    """Static analysis tool wrapper for bug detection.

    Supports multiple tools:
    - mypy: Type checking
    - pylint: Code quality analysis
    - ruff: Fast Python linting
    """

    SEVERITY_MAP = {
        "error": BugSeverity.ERROR,
        "warning": BugSeverity.WARNING,
        "note": BugSeverity.INFO,
    }

    def __init__(self, tools: list[str] | None = None):
        """Initialize the static analyzer.

        Args:
            tools: List of tools to use ("mypy", "pylint", "ruff").
                Defaults to all available tools.
        """
        if tools is None:
            tools = ["mypy", "pylint", "ruff"]

        self.tools = tools
        self._available_tools: dict[str, bool] = {}

    @property
    def name(self) -> str:
        """Get detector name."""
        return f"StaticAnalyzer({','.join(self.tools)})"

    @property
    def is_available(self) -> bool:
        """Check if any analysis tool is available."""
        return any(self._check_tool(tool) for tool in self.tools)

    def _check_tool(self, tool: str) -> bool:
        """Check if a tool is installed."""
        if tool not in self._available_tools:
            self._available_tools[tool] = shutil.which(tool) is not None
        return self._available_tools[tool]

    async def detect(self, directory: str | Path) -> list[DetectedBug]:
        """Run static analysis on a directory.

        Args:
            directory: Directory to analyze

        Returns:
            List of detected bugs from all available tools
        """
        directory = Path(directory)
        all_bugs: list[DetectedBug] = []

        if "mypy" in self.tools and self._check_tool("mypy"):
            bugs = await self._run_mypy(directory)
            all_bugs.extend(bugs)

        if "pylint" in self.tools and self._check_tool("pylint"):
            bugs = await self._run_pylint(directory)
            all_bugs.extend(bugs)

        if "ruff" in self.tools and self._check_tool("ruff"):
            bugs = await self._run_ruff(directory)
            all_bugs.extend(bugs)

        return all_bugs

    async def detect_file(self, file_path: str | Path) -> list[DetectedBug]:
        """Run static analysis on a single file.

        Args:
            file_path: File to analyze

        Returns:
            List of detected bugs
        """
        file_path = Path(file_path)
        all_bugs: list[DetectedBug] = []

        if "mypy" in self.tools and self._check_tool("mypy"):
            bugs = await self._run_mypy_on_file(file_path)
            all_bugs.extend(bugs)

        if "pylint" in self.tools and self._check_tool("pylint"):
            bugs = await self._run_pylint_on_file(file_path)
            all_bugs.extend(bugs)

        if "ruff" in self.tools and self._check_tool("ruff"):
            bugs = await self._run_ruff_on_file(file_path)
            all_bugs.extend(bugs)

        return all_bugs

    async def _run_mypy(self, directory: Path) -> list[DetectedBug]:
        """Run mypy type checking."""
        try:
            result = subprocess.run(
                [
                    "mypy",
                    str(directory),
                    "--show-error-codes",
                    "--show-column-numbers",
                    "--no-error-summary",
                    "--no-color-output",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            return self._parse_mypy_output(result.stdout + result.stderr)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"mypy failed: {e}")
            return []

    async def _run_mypy_on_file(self, file_path: Path) -> list[DetectedBug]:
        """Run mypy on a single file."""
        try:
            result = subprocess.run(
                [
                    "mypy",
                    str(file_path),
                    "--show-error-codes",
                    "--show-column-numbers",
                    "--no-error-summary",
                    "--no-color-output",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            return self._parse_mypy_output(result.stdout + result.stderr)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"mypy failed on {file_path}: {e}")
            return []

    def _parse_mypy_output(self, output: str) -> list[DetectedBug]:
        """Parse mypy output into DetectedBug objects."""
        bugs: list[DetectedBug] = []

        # Pattern: file.py:line:column: severity: message [code]
        pattern = r"^(.*?):(\d+):(\d+):\s*(\w+):\s*(.*?)(?:\s*\[(\w+)\])?$"

        for line in output.strip().split("\n"):
            if not line or line.startswith("Success"):
                continue

            match = re.match(pattern, line, re.MULTILINE)
            if match:
                file_path_str = match.group(1)
                line_num = int(match.group(2))
                col_num = int(match.group(3))
                severity_str = match.group(4).lower()
                message = match.group(5)
                error_code = match.group(6) or "mypy-error"

                severity = self.SEVERITY_MAP.get(severity_str, BugSeverity.WARNING)

                bug = DetectedBug(
                    source=BugSource.MYPY,
                    severity=severity,
                    location=CodeLocation(
                        file_path=Path(file_path_str).resolve(),
                        line_number=line_num,
                        column=col_num,
                    ),
                    message=message.strip(),
                    error_code=error_code,
                    tool_output=line,
                )
                bugs.append(bug)

        return bugs

    async def _run_pylint(self, directory: Path) -> list[DetectedBug]:
        """Run pylint code analysis."""
        try:
            result = subprocess.run(
                [
                    "pylint",
                    str(directory),
                    "--output-format=json",
                    "--disable=R,C",  # Disable refactoring and convention messages
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            return self._parse_pylint_json(result.stdout)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"pylint failed: {e}")
            return []

    async def _run_pylint_on_file(self, file_path: Path) -> list[DetectedBug]:
        """Run pylint on a single file."""
        try:
            result = subprocess.run(
                [
                    "pylint",
                    str(file_path),
                    "--output-format=json",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            return self._parse_pylint_json(result.stdout)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"pylint failed on {file_path}: {e}")
            return []

    def _parse_pylint_json(self, output: str) -> list[DetectedBug]:
        """Parse pylint JSON output into DetectedBug objects."""
        bugs: list[DetectedBug] = []

        try:
            messages = json.loads(output)
            if not isinstance(messages, list):
                return bugs

            for msg in messages:
                msg_type = msg.get("type", "warning").lower()
                severity = self.SEVERITY_MAP.get(msg_type, BugSeverity.WARNING)

                # Check for auto-fixable issues
                suggested_fix = None
                if msg.get("symbol") == "unused-import":
                    suggested_fix = f"Remove unused import: {msg.get('obj', '')}"

                bug = DetectedBug(
                    source=BugSource.PYLINT,
                    severity=severity,
                    location=CodeLocation(
                        file_path=Path(msg["path"]).resolve(),
                        line_number=msg["line"],
                        column=msg.get("column"),
                    ),
                    message=msg["message"],
                    error_code=msg["symbol"],
                    suggested_fix=suggested_fix,
                    tool_output=json.dumps(msg),
                )
                bugs.append(bug)
        except json.JSONDecodeError:
            pass

        return bugs

    async def _run_ruff(self, directory: Path) -> list[DetectedBug]:
        """Run ruff linter."""
        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    str(directory),
                    "--output-format=json",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            return self._parse_ruff_json(result.stdout)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"ruff failed: {e}")
            return []

    async def _run_ruff_on_file(self, file_path: Path) -> list[DetectedBug]:
        """Run ruff on a single file."""
        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    str(file_path),
                    "--output-format=json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            return self._parse_ruff_json(result.stdout)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"ruff failed on {file_path}: {e}")
            return []

    def _parse_ruff_json(self, output: str) -> list[DetectedBug]:
        """Parse ruff JSON output into DetectedBug objects."""
        bugs: list[DetectedBug] = []

        try:
            messages = json.loads(output)
            if not isinstance(messages, list):
                return bugs

            for msg in messages:
                severity = BugSeverity.WARNING
                if msg.get("fix"):
                    severity = BugSeverity.INFO  # Fixable issues are lower priority

                suggested_fix = None
                if msg.get("fix"):
                    fix = msg["fix"]
                    suggested_fix = f"Replace with: {fix.get('content', '')}"

                bug = DetectedBug(
                    source=BugSource.RUFF,
                    severity=severity,
                    location=CodeLocation(
                        file_path=Path(msg["filename"]).resolve(),
                        line_number=msg["location"]["row"],
                        column=msg["location"].get("column"),
                        end_line=msg["end_location"].get("row"),
                    ),
                    message=msg["message"],
                    error_code=msg["code"],
                    suggested_fix=suggested_fix,
                    tool_output=json.dumps(msg),
                )
                bugs.append(bug)
        except json.JSONDecodeError:
            pass

        return bugs


class DetectionError(Exception):
    """Error during bug detection."""

    def __init__(self, tool: str, message: str, output: str | None = None):
        self.tool = tool
        self.output = output
        super().__init__(f"{tool}: {message}")
