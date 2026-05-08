"""Pattern-based bug detection for common anti-patterns.

This module detects common Python anti-patterns without using LLM,
providing fast detection of known problematic patterns.
"""

import ast
import re
from pathlib import Path
from typing import Any
from loguru import logger
from src.detection.domain.entities import (
    BugSeverity,
    BugSource,
    CodeLocation,
    DetectedBug,
)


class PatternDetector:
    """Detects common anti-patterns using regex and AST analysis.

    This provides fast detection without LLM calls for known patterns
    like mutable defaults, bare except clauses, etc.
    """

    # Regex patterns for common issues
    PATTERNS = {
        # Dictionary access patterns
        "dict_direct_access": {
            "pattern": r"\[['\"](\w+)['\"]\]",
            "message": "Direct dictionary access without .get() - may raise KeyError",
            "suggestion": "Use .get() method or check if key exists first",
            "severity": BugSeverity.WARNING,
        },
        # List modification while iterating
        "list_modify_while_iterating": {
            "pattern": r"for\s+\w+\s+in\s+(\w+):.*?\1\.(remove|append|insert)",
            "message": "Modifying list while iterating over it",
            "suggestion": "Create a new list instead of modifying during iteration",
            "severity": BugSeverity.ERROR,
        },
        # Division without zero check
        "division_no_check": {
            "pattern": r"return\s+.*/\s*\w+",
            "message": "Division operation without zero check",
            "suggestion": "Add check: if denominator == 0: raise ValueError(...)",
            "severity": BugSeverity.WARNING,
        },
        # Empty list check missing for len()
        "len_no_empty_check": {
            "pattern": r"len\(\w+\)",
            "message": "Using len() without checking if empty first",
            "suggestion": "Check if list is empty before using len() in division",
            "severity": BugSeverity.WARNING,
        },
        # Bare except
        "bare_except": {
            "pattern": r"except\s*:\s*$",
            "message": "Bare 'except:' clause catches SystemExit and KeyboardInterrupt",
            "suggestion": "Use 'except Exception:' or be more specific",
            "severity": BugSeverity.WARNING,
        },
        # Mutable default
        "mutable_default": {
            "pattern": r"def\s+\w+\s*\([^)]*=(?<!,)\s*(\[|{)",
            "message": "Mutable default argument (list or dict)",
            "suggestion": "Use None as default and initialize mutable in function body",
            "severity": BugSeverity.ERROR,
        },
        # Hardcoded secret
        "hardcoded_secret": {
            "pattern": r"(password|secret|api_key|token)\s*[=:]\s*['\"][^'\"]+['\"]",
            "message": "Potential hardcoded secret",
            "suggestion": "Use environment variables or secret management",
            "severity": BugSeverity.WARNING,
        },
        # Debug print
        "debug_print": {
            "pattern": r"print\s*\(",
            "message": "Debug print statement found",
            "suggestion": "Use logging instead of print",
            "severity": BugSeverity.INFO,
        },
        # SQL injection
        "sql_string_format": {
            "pattern": r"(execute|query)\s*\(\s*['\"].*%s",
            "message": "Potential SQL injection via string formatting",
            "suggestion": "Use parameterized queries",
            "severity": BugSeverity.ERROR,
        },
        # Broad exception
        "broad_exception": {
            "pattern": r"except\s+(Exception|BaseException)\s*:",
            "message": "Overly broad exception handling",
            "suggestion": "Catch specific exceptions",
            "severity": BugSeverity.WARNING,
        },
    }

    def __init__(self):
        """Initialize pattern detector."""
        self._compiled_patterns = {
            name: re.compile(info["pattern"], re.MULTILINE | re.IGNORECASE)
            for name, info in self.PATTERNS.items()
        }

    @property
    def name(self) -> str:
        """Get detector name."""
        return "PatternDetector"

    @property
    def is_available(self) -> bool:
        """Pattern detector is always available."""
        return True

    async def detect(self, directory: str | Path) -> list[DetectedBug]:
        """Detect bugs in all Python files in directory.

        Args:
            directory: Directory to analyze

        Returns:
            List of detected bugs
        """
        directory = Path(directory)
        bugs: list[DetectedBug] = []

        for file_path in directory.rglob("*.py"):
            if "test" in file_path.name.lower():
                continue
            file_bugs = self.scan_file(file_path)
            bugs.extend(file_bugs)

        return bugs

    async def detect_file(self, file_path: str | Path) -> list[DetectedBug]:
        """Detect bugs in a single file.

        Args:
            file_path: File to analyze

        Returns:
            List of detected bugs
        """
        return self.scan_file(Path(file_path))

    def scan_file(self, file_path: Path) -> list[DetectedBug]:
        """Scan a file for anti-patterns.

        Args:
            file_path: File to scan

        Returns:
            List of detected pattern violations
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            logger.info(f"Failed to read {file_path}: {e}")
            return []

        bugs: list[DetectedBug] = []

        # Run regex patterns
        bugs.extend(self._scan_regex(content, file_path))

        # Run AST-based patterns
        bugs.extend(self._scan_ast(content, file_path))

        return bugs

    def _scan_regex(self, content: str, file_path: Path) -> list[DetectedBug]:
        """Scan using regex patterns."""
        bugs: list[DetectedBug] = []

        for pattern_name, compiled in self._compiled_patterns.items():
            pattern_info = self.PATTERNS[pattern_name]

            for match in compiled.finditer(content):
                # Calculate line number
                line_num = content[:match.start()].count("\n") + 1

                bug = DetectedBug(
                    source=BugSource.PATTERN_MATCH,
                    severity=pattern_info["severity"],
                    location=CodeLocation(
                        file_path=file_path,
                        line_number=line_num,
                    ),
                    message=pattern_info["message"],
                    error_code=f"pattern-{pattern_name}",
                    suggested_fix=pattern_info.get("suggestion"),
                    code_snippet=content[match.start():match.end()],
                )
                bugs.append(bug)

        return bugs

    def _scan_ast(self, content: str, file_path: Path) -> list[DetectedBug]:
        """Scan using AST-based patterns."""
        bugs: list[DetectedBug] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return bugs

        # Walk the AST
        for node in ast.walk(tree):
            # Check for issues
            if isinstance(node, ast.FunctionDef):
                bugs.extend(self._check_function(node, content, file_path))
            elif isinstance(node, ast.ExceptHandler):
                bugs.extend(self._check_exception_handler(node, content, file_path))

        return bugs

    def _check_function(
        self,
        node: ast.FunctionDef,
        content: str,
        file_path: Path,
    ) -> list[DetectedBug]:
        """Check a function for issues."""
        bugs: list[DetectedBug] = []

        # Check for unused arguments
        if node.args.args:
            arg_names = {arg.arg for arg in node.args.args}

            # Simple check: see if argument names appear in function body
            body_text = ast.unparse(node.body) if hasattr(ast, "unparse") else ""

            for arg in arg_names:
                if arg not in body_text and arg != "self":
                    bug = DetectedBug(
                        source=BugSource.PATTERN_MATCH,
                        severity=BugSeverity.INFO,
                        location=CodeLocation(
                            file_path=file_path,
                            line_number=node.lineno,
                        ),
                        message=f"Potentially unused argument: {arg}",
                        error_code="pattern-unused-arg",
                        code_snippet=f"def {node.name}(...)",
                    )
                    bugs.append(bug)

        return bugs

    def _check_exception_handler(
        self,
        node: ast.ExceptHandler,
        content: str,
        file_path: Path,
    ) -> list[DetectedBug]:
        """Check an exception handler for issues."""
        bugs: list[DetectedBug] = []

        # Check for pass in except block
        if isinstance(node.body, list) and len(node.body) == 1:
            if isinstance(node.body[0], ast.Pass):
                bug = DetectedBug(
                    source=BugSource.PATTERN_MATCH,
                    severity=BugSeverity.WARNING,
                    location=CodeLocation(
                        file_path=file_path,
                        line_number=node.lineno,
                    ),
                    message="Bare 'except: pass' swallows exceptions silently",
                    error_code="pattern-except-pass",
                    suggested_fix="Add logging or proper error handling",
                )
                bugs.append(bug)

        return bugs


class AntiPatternRegistry:
    """Registry of anti-patterns for discovery.

    Allows users to register custom patterns for detection.
    """

    def __init__(self):
        """Initialize registry."""
        self._patterns: dict[str, dict] = {}

    def register(
        self,
        name: str,
        pattern: str,
        message: str,
        severity: BugSeverity = BugSeverity.WARNING,
        suggestion: str | None = None,
    ) -> None:
        """Register a custom anti-pattern.

        Args:
            name: Pattern identifier
            pattern: Regex pattern string
            message: Human-readable description
            severity: Severity level
            suggestion: Fix suggestion
        """
        self._patterns[name] = {
            "pattern": pattern,
            "message": message,
            "severity": severity,
            "suggestion": suggestion,
        }

    def get_detector(self) -> PatternDetector:
        """Get a detector with registered patterns."""
        detector = PatternDetector()
        detector.PATTERNS.update(self._patterns)
        detector._compiled_patterns = {
            name: re.compile(info["pattern"], re.MULTILINE | re.IGNORECASE)
            for name, info in detector.PATTERNS.items()
        }
        return detector
