"""Bug detection service for coordinating multiple detectors.

This module implements the application service that aggregates bugs
from multiple detection sources, deduplicates them, and provides
a unified interface for bug detection.
"""

import asyncio
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from loguru import logger
from src.detection.domain.entities import (
    BugSeverity,
    BugSource,
    DetectedBug,
    DetectionResult,
)
from src.detection.domain.interfaces import IBugDetector
from src.detection.infrastructure.static_analyzer import StaticAnalyzer
from src.detection.infrastructure.test_failure_parser import TestFailureParser
from src.discovery.infrastructure.pattern_detector import PatternDetector
from src.fix_agent_orchestration.domain.interfaces import ILLMClient


# Thread pool for running sync detectors concurrently
_executor = ThreadPoolExecutor(max_workers=4)


class BugDetectionService:
    """Service for comprehensive bug detection.

    This service coordinates multiple bug detectors (static analysis,
    pattern matching, test failures, and optional LLM discovery)
    and aggregates their results with deduplication and prioritization.

    Usage:
        service = BugDetectionService()
        result = await service.detect_bugs("./src")

        for bug in result.bugs:
            print(bug)
    """

    def __init__(self, detectors: list[IBugDetector] | None = None):
        """Initialize the detection service.

        Args:
            detectors: List of bug detectors to use.
                If None, uses default detectors (mypy, pylint, ruff,
                pattern detector, test failures).
        """
        if detectors is None:
            self._detectors: list[IBugDetector] = [
                StaticAnalyzer(tools=["mypy", "pylint", "ruff"]),
                PatternDetector(),  # Always available, no external deps
                TestFailureParser(),
            ]
        else:
            self._detectors = detectors

    def register_detector(self, detector: IBugDetector) -> None:
        """Register a new bug detector.

        Args:
            detector: Detector to add to the pipeline
        """
        self._detectors.append(detector)

    async def detect_bugs(
        self,
        path: str | Path,
        include_tests: bool = True,
        use_llm_discovery: bool = False,
        llm_client: "ILLMClient | None" = None,
    ) -> DetectionResult:
        """Run comprehensive bug detection (async).

        Orchestrates multiple detectors (static analysis, pattern matching,
        test failures, and optional LLM discovery) concurrently.

        This must be called from within an async context (e.g. inside
        asyncio.run() or another async function).

        Args:
            path: File or directory to analyze
            include_tests: Whether to include test file analysis
            use_llm_discovery: Whether to use LLM-based discovery
            llm_client: LLM client for discovery (required if use_llm_discovery=True)

        Returns:
            DetectionResult with all found bugs
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return await self._detect_file(path, use_llm_discovery, llm_client)
        else:
            return await self._detect_directory(path, include_tests, use_llm_discovery, llm_client)

    async def _detect_file(
        self,
        file_path: Path,
        use_llm_discovery: bool,
        llm_client: "ILLMClient | None",
    ) -> DetectionResult:
        """Detect bugs in a single file via async subprocesses."""
        logger.info(f"Starting bug detection for file: {file_path}")
        start_time = time.perf_counter()
        all_bugs: list[DetectedBug] = []
        tools_run: list[str] = []
        errors: list[str] = []

        # Run each detector on the file concurrently
        tasks: list[asyncio.Task] = []
        for detector in self._detectors:
            if not detector.is_available:
                continue

            if hasattr(detector, "detect_file"):
                logger.info(f"Running {detector.name} on file...")
                # Check if detector.detect_file is async or sync
                if inspect.iscoroutinefunction(detector.detect_file):
                    tasks.append(asyncio.create_task(detector.detect_file(file_path)))
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_running_loop()
                    tasks.append(loop.run_in_executor(_executor, detector.detect_file, file_path))
                tools_run.append(detector.name)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Detection failed: {result}")
                    errors.append(str(result))
                else:
                    all_bugs.extend(result)

        # Add LLM discovery if requested
        if use_llm_discovery and llm_client is not None:
            from src.detection.infrastructure.llm_bug_detector import LLMBugDetector
            logger.info("Running LLM-based discovery...")
            try:
                llm_detector = LLMBugDetector(llm_client)
                bugs = await llm_detector.detect_file(file_path)
                logger.info(f"Found {len(bugs)} bugs via LLM")
                all_bugs.extend(bugs)
                tools_run.append("LLM-Discovery")
            except Exception as e:
                error_msg = f"LLM discovery failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Deduplicate bugs
        unique_bugs = self._deduplicate_bugs(all_bugs)

        end_time = time.perf_counter()
        duration = end_time - start_time

        result = DetectionResult(
            bugs=unique_bugs,
            duration_seconds=duration,
            files_analyzed=1,
            tools_run=tools_run,
            errors=errors,
        )

        self._print_summary(result)
        return result

    async def _detect_directory(
        self,
        directory: Path,
        include_tests: bool,
        use_llm_discovery: bool,
        llm_client: "ILLMClient | None",
    ) -> DetectionResult:
        """Detect bugs in a directory via async subprocesses."""
        logger.info(f"Starting bug detection for directory: {directory}")

        start_time = time.perf_counter()
        all_bugs: list[DetectedBug] = []
        tools_run: list[str] = []
        errors: list[str] = []

        # Count files
        files_analyzed = self._count_python_files(directory, include_tests)

        # Run each detector concurrently
        tasks: list[asyncio.Task] = []
        for detector in self._detectors:
            if not detector.is_available:
                logger.info(f"Skipping {detector.name} (not available)")
                continue

            logger.info(f"Running {detector.name}...")
            # Check if detector.detect is async or sync
            if inspect.iscoroutinefunction(detector.detect):
                tasks.append(asyncio.create_task(detector.detect(directory)))
            else:
                # Run sync function in thread pool
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(_executor, detector.detect, directory))
            tools_run.append(detector.name)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Detection failed: {result}")
                    errors.append(str(result))
                else:
                    all_bugs.extend(result)

        # Add LLM discovery if requested
        if use_llm_discovery and llm_client is not None:
            from src.detection.infrastructure.llm_bug_detector import LLMBugDetector
            logger.info("Running LLM-based discovery...")
            try:
                llm_detector = LLMBugDetector(llm_client)
                bugs = await llm_detector.detect(directory)
                logger.info(f"Found {len(bugs)} bugs via LLM")
                all_bugs.extend(bugs)
                tools_run.append("LLM-Discovery")
            except Exception as e:
                error_msg = f"LLM discovery failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Deduplicate bugs
        unique_bugs = self._deduplicate_bugs(all_bugs)

        end_time = time.perf_counter()
        duration = end_time - start_time

        result = DetectionResult(
            bugs=unique_bugs,
            duration_seconds=duration,
            files_analyzed=files_analyzed,
            tools_run=tools_run,
            errors=errors,
        )

        self._print_summary(result)
        return result

    def detect_single_bug(
        self,
        file_path: str | Path,
        bug_description: str,
    ) -> DetectedBug | None:
        """Detect a specific bug in a file.

        Useful when you know approximately where a bug is.

        Args:
            file_path: File to analyze
            bug_description: Description to match

        Returns:
            Best matching DetectedBug or None
        """
        file_path = Path(file_path)
        all_bugs: list[DetectedBug] = []

        # Run ALL detectors that support single file detection
        for detector in self._detectors:
            if not detector.is_available:
                continue

            if hasattr(detector, "detect_file"):
                try:
                    bugs = detector.detect_file(file_path)
                    all_bugs.extend(bugs)
                except Exception:
                    pass  # Continue with other detectors

        # Find best match
        for bug in all_bugs:
            if bug_description.lower() in bug.message.lower():
                return bug

        return None

    def _deduplicate_bugs(self, bugs: list[DetectedBug]) -> list[DetectedBug]:
        """Remove duplicate bugs based on location and message.

        Two bugs are considered duplicates if they:
        1. Are in the same file
        2. Are on the same line
        3. Have similar messages (first 50 chars)

        Args:
            bugs: List of potentially duplicate bugs

        Returns:
            List of unique bugs
        """
        seen = set()
        unique = []

        for bug in bugs:
            # Create deduplication key
            key = (
                str(bug.location.file_path),
                bug.location.line_number,
                bug.message[:50].lower(),
            )

            if key not in seen:
                seen.add(key)
                unique.append(bug)

        return unique

    def _count_python_files(self, directory: Path, include_tests: bool) -> int:
        """Count Python files in directory."""
        count = 0
        for file in directory.rglob("*.py"):
            if not include_tests and "test" in file.name.lower():
                continue
            count += 1
        return count

    def _print_summary(self, result: DetectionResult) -> None:
        """Print detection summary."""
        logger.info(f"Detection complete!")
        logger.info(f"Duration: {result.duration_seconds:.2f}s")
        logger.info(f"Files analyzed: {result.files_analyzed}")
        logger.info(f"Tools run: {', '.join(result.tools_run)}")
        logger.info(f"Total bugs: {len(result.bugs)}")
        logger.info(f"Errors: {result.error_count}")
        logger.info(f"Warnings: {result.warning_count}")
        logger.info(f"Info: {result.info_count}")
        logger.info(f"Auto-fixable: {result.auto_fixable_count}")

        if result.errors:
            logger.info(f"Detection errors:")
            for error in result.errors:
                logger.info(f"{error}")

    def get_by_severity(self, result: DetectionResult, severity: BugSeverity) -> list[DetectedBug]:
        """Filter bugs by severity."""
        return result.get_by_severity(severity)

    def get_by_source(self, result: DetectionResult, source: BugSource) -> list[DetectedBug]:
        """Filter bugs by detection source."""
        return result.get_by_source(source)
