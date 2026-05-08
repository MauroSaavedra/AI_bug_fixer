"""Domain entities for bug detection.

This module defines the data structures used to represent detected bugs
from various sources (static analysis, test failures, LLM discovery).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class BugSource(Enum):
    """Source of bug detection."""

    MYPY = auto()
    PYLINT = auto()
    RUFF = auto()
    PYTEST = auto()
    LLM_DISCOVERY = auto()
    PATTERN_MATCH = auto()


class BugSeverity(Enum):
    """Severity level of detected bugs."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass(frozen=True)
class CodeLocation:
    """Location of a bug in code."""

    file_path: Path
    line_number: int
    column: int | None = None
    end_line: int | None = None

    def __str__(self) -> str:
        """Human-readable location string."""
        location = f"{self.file_path}:{self.line_number}"
        if self.column:
            location += f":{self.column}"
        return location


@dataclass
class DetectedBug:
    """Represents a detected bug."""

    source: BugSource
    severity: BugSeverity
    location: CodeLocation
    message: str
    id: str = field(default_factory=lambda: "")
    error_code: str | None = None
    suggested_fix: str | None = None
    code_snippet: str | None = None
    tool_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            import hashlib

            location_str = str(self.location)
            content = f"{location_str}:{self.message}".encode()
            object.__setattr__(
                self, "id", hashlib.md5(content).hexdigest()[:12]
            )

    @property
    def qualified_name(self) -> str:
        """Get qualified identifier for the bug."""
        return f"{self.location.file_path.stem}:{self.location.line_number}:{self.error_code or 'BUG'}"

    @property
    def is_auto_fixable(self) -> bool:
        """Check if bug has an automatic fix available."""
        return self.suggested_fix is not None

    def to_user_goal(self) -> str:
        """Convert to a user goal string for the bug fixer."""
        parts = [
            f"Fix {self.severity.name.lower()}",
            f"in {self.location.file_path.name}",
            f"at line {self.location.line_number}",
        ]

        if self.error_code:
            parts.append(f"({self.error_code})")

        parts.append(f": {self.message}")

        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source": self.source.name,
            "severity": self.severity.name,
            "location": {
                "file_path": str(self.location.file_path),
                "line_number": self.location.line_number,
                "column": self.location.column,
                "end_line": self.location.end_line,
            },
            "message": self.message,
            "error_code": self.error_code,
            "suggested_fix": self.suggested_fix,
            "code_snippet": self.code_snippet,
            "is_auto_fixable": self.is_auto_fixable,
        }

    def __str__(self) -> str:
        """Human-readable bug description."""
        severity_icon = {
            BugSeverity.ERROR: "❌",
            BugSeverity.WARNING: "⚠️",
            BugSeverity.INFO: "ℹ️",
        }.get(self.severity, "❓")

        return f"{severity_icon} [{self.source.name}] {self.location} - {self.message}"

    def __hash__(self) -> int:
        """Hash based on ID for deduplication."""
        return hash(self.id)


@dataclass
class DetectionResult:
    """Result of a bug detection run."""

    bugs: list[DetectedBug] = field(default_factory=list)
    duration_seconds: float = 0.0
    files_analyzed: int = 0
    tools_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Count of error-level bugs."""
        return len([b for b in self.bugs if b.severity == BugSeverity.ERROR])

    @property
    def warning_count(self) -> int:
        """Count of warning-level bugs."""
        return len([b for b in self.bugs if b.severity == BugSeverity.WARNING])

    @property
    def info_count(self) -> int:
        """Count of info-level bugs."""
        return len([b for b in self.bugs if b.severity == BugSeverity.INFO])

    @property
    def auto_fixable_count(self) -> int:
        """Count of bugs with suggested fixes."""
        return len([b for b in self.bugs if b.is_auto_fixable])

    def get_by_severity(self, severity: BugSeverity) -> list[DetectedBug]:
        """Filter bugs by severity."""
        return [b for b in self.bugs if b.severity == severity]

    def get_by_source(self, source: BugSource) -> list[DetectedBug]:
        """Filter bugs by source."""
        return [b for b in self.bugs if b.source == source]

    def sort_by_severity(self) -> list[DetectedBug]:
        """Return bugs sorted by severity (errors first)."""
        severity_order = {
            BugSeverity.ERROR: 0,
            BugSeverity.WARNING: 1,
            BugSeverity.INFO: 2,
        }
        return sorted(self.bugs, key=lambda b: severity_order.get(b.severity, 3))

    def __len__(self) -> int:
        """Total number of bugs."""
        return len(self.bugs)

    def __iter__(self):
        """Iterate over bugs."""
        return iter(self.bugs)
