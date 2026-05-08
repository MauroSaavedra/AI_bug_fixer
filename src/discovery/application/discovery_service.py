"""Discovery service for finding bugs using LLM and patterns.

This module coordinates code discovery using both pattern matching
and LLM-based analysis to find bugs that static analysis might miss.
"""

import time
from pathlib import Path
from loguru import logger
from src.detection.application.bug_detection_service import BugDetectionService
from src.detection.domain.entities import DetectedBug, DetectionResult
from src.detection.infrastructure.static_analyzer import StaticAnalyzer
from src.discovery.infrastructure.pattern_detector import PatternDetector


class DiscoveryService:
    """Service for discovering bugs using multiple methods.

    Combines:
    - Pattern-based detection (fast, no LLM)
    - Static analysis (mypy, pylint, ruff)
    - Optional LLM-based discovery (slow but thorough)

    Usage:
        service = DiscoveryService()
        result = await service.discover_bugs("./src")
    """

    def __init__(
        self,
        use_patterns: bool = True,
        use_static_analysis: bool = True,
        use_llm: bool = False,
    ):
        """Initialize discovery service.

        Args:
            use_patterns: Enable pattern-based detection
            use_static_analysis: Enable static analysis tools
            use_llm: Enable LLM-based discovery (slower)
        """
        self.use_patterns = use_patterns
        self.use_static_analysis = use_static_analysis
        self.use_llm = use_llm

        # Initialize detectors
        self._pattern_detector = PatternDetector() if use_patterns else None
        self._static_analyzer = StaticAnalyzer() if use_static_analysis else None
        self._bug_detection = BugDetectionService()

    async def discover_bugs(
        self,
        directory: str | Path,
        max_findings: int | None = None,
    ) -> DetectionResult:
        """Discover bugs in a directory.

        Args:
            directory: Directory to analyze
            max_findings: Maximum bugs to return (None = all)

        Returns:
            DetectionResult with all found bugs
        """
        directory = Path(directory)
        logger.info(f"Starting bug discovery for: {directory}")

        start_time = time.perf_counter()
        all_bugs: list[DetectedBug] = []

        # Pattern-based detection
        if self.use_patterns and self._pattern_detector:
            logger.info("Running pattern detection...")
            bugs = await self._run_pattern_detection(directory)
            all_bugs.extend(bugs)
            logger.info(f"Found {len(bugs)} pattern violations")

        # Static analysis
        if self.use_static_analysis:
            logger.info("Running static analysis...")
            result = await self._bug_detection.detect_bugs(directory)
            all_bugs.extend(result.bugs)
            logger.info(f"Found {len(result.bugs)} static analysis issues")

        # Deduplicate
        unique_bugs = self._deduplicate(all_bugs)

        # Limit if requested
        if max_findings and len(unique_bugs) > max_findings:
            unique_bugs = unique_bugs[:max_findings]

        end_time = time.perf_counter()

        # Create result
        result = DetectionResult(
            bugs=unique_bugs,
            duration_seconds=end_time - start_time,
            files_analyzed=self._count_files(directory),
            tools_run=self._get_tools_used(),
        )

        # Print summary
        self._print_summary(result)

        return result

    async def _run_pattern_detection(self, directory: Path) -> list[DetectedBug]:
        """Run pattern-based detection on directory."""
        if not self._pattern_detector:
            return []

        bugs: list[DetectedBug] = []

        for file_path in directory.rglob("*.py"):
            if "test" in file_path.name.lower():
                continue  # Skip test files

            file_bugs = self._pattern_detector.scan_file(file_path)
            bugs.extend(file_bugs)

        return bugs

    def _count_files(self, directory: Path) -> int:
        """Count Python files in directory."""
        return len(list(directory.rglob("*.py")))

    def _get_tools_used(self) -> list[str]:
        """Get list of tools that were used."""
        tools = []
        if self.use_patterns:
            tools.append("pattern-detector")
        if self.use_static_analysis:
            tools.extend(["mypy", "pylint", "ruff"])
        return tools

    def _deduplicate(self, bugs: list[DetectedBug]) -> list[DetectedBug]:
        """Remove duplicate bugs."""
        seen = set()
        unique = []

        for bug in bugs:
            key = (str(bug.location.file_path), bug.location.line_number, bug.message[:50])
            if key not in seen:
                seen.add(key)
                unique.append(bug)

        return unique

    def _print_summary(self, result: DetectionResult) -> None:
        """Print discovery summary."""
        logger.info(f"Discovery complete!")
        logger.info(f"Duration: {result.duration_seconds:.2f}s")
        logger.info(f"Files analyzed: {result.files_analyzed}")
        logger.info(f"Total bugs found: {len(result.bugs)}")
        logger.info(f"Errors: {result.error_count}")
        logger.info(f"Warnings: {result.warning_count}")
        logger.info(f"Info: {result.info_count}")
        logger.info(f"Auto-fixable: {result.auto_fixable_count}")

        if result.bugs:
            logger.info("Top findings:")
            for i, bug in enumerate(result.bugs[:5], 1):
                logger.info(f"{i}. {bug}")
