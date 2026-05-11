"""LLM-based bug detector for semantic code analysis.

This module implements bug detection using LLM analysis, examining
functions and methods individually to find logical bugs, edge cases,
and semantic issues that static analysis tools miss.
"""

import asyncio
import time
import json
from pathlib import Path
from loguru import logger
try:
    from langfuse import observe
except ImportError:
    observe = None  # type: ignore[assignment]
from src.detection.domain.entities import (
    BugSeverity,
    BugSource,
    CodeLocation,
    DetectedBug,
)
from src.detection.domain.interfaces import IBugDetector
from src.ingestion.domain.entities import EntityType
from src.ingestion.infrastructure.python_ast_chunker import PythonASTChunker
from src.observability.langfuse_utils import update_current_generation


class LLMBugDetector(IBugDetector):
    """Bug detector using LLM analysis of code.

    Analyzes functions and methods individually to find:
    - Logic errors and edge cases
    - Missing input validation
    - Security vulnerabilities
    - Performance issues
    - Common anti-patterns

    Features:
    - Function-by-function analysis for granularity
    - Smart filtering (skips trivial functions)
    - Parallel async execution
    - Context-aware bug detection
    """

    # Functions smaller than this are considered trivial
    MIN_LINES_FOR_ANALYSIS = 3
    # Maximum lines to analyze per function
    MAX_LINES_FOR_ANALYSIS = 100

    def __init__(self, llm_client):
        """Initialize LLM bug detector.

        Args:
            llm_client: LLM client for analysis
        """
        self.llm_client = llm_client
        self._ast_chunker = PythonASTChunker()

    @property
    def name(self) -> str:
        """Get detector name."""
        return "LLMBugDetector"

    @property
    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self.llm_client is not None and self.llm_client.is_available()

    async def detect(self, directory: str | Path) -> list[DetectedBug]:
        """Detect bugs in all Python files in directory.

        Args:
            directory: Directory to analyze

        Returns:
            List of detected bugs
        """
        directory = Path(directory)
        bugs: list[DetectedBug] = []

        # Process each Python file
        for file_path in directory.rglob("*.py"):
            if "test" in file_path.name.lower():
                continue  # Skip test files

            file_bugs = await self.detect_file(file_path)
            bugs.extend(file_bugs)

        return bugs

    async def detect_file(self, file_path: Path) -> list[DetectedBug]:
        """Detect bugs in a single file.

        Analyzes file function-by-function using AST extraction
        and LLM-based semantic analysis.

        Args:
            file_path: File to analyze

        Returns:
            List of detected bugs
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        # Extract functions and methods using AST
        try:
            entities = self._ast_chunker.chunk_file(file_path, content)
        except SyntaxError:
            return []

        # Filter to only callable entities
        callable_entities = [
            e for e in entities
            if e.entity_type in (EntityType.FUNCTION, EntityType.METHOD)
        ]

        if not callable_entities:
            return []

        logger.info(f"Analyzing {len(callable_entities)} functions in {file_path.name}...")

        # Filter entities worth analyzing
        entities_to_analyze = self._filter_entities(callable_entities)

        if not entities_to_analyze:
            return []

        # Analyze each entity in parallel
        tasks = [
            self._analyze_entity(entity, file_path, content)
            for entity in entities_to_analyze
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect bugs, handling any exceptions
        bugs: list[DetectedBug] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Analysis error: {result}")
                continue
            if result:
                bugs.extend(result)

        return bugs

    def _filter_entities(self, entities: list) -> list:
        """Filter entities to only analyze non-trivial functions.

        Skips:
        - Very short functions (< MIN_LINES_FOR_ANALYSIS lines)
        - Very long functions (> MAX_LINES_FOR_ANALYSIS lines)
        - Simple getters/setters

        Args:
            entities: List of code entities

        Returns:
            Filtered list of entities to analyze
        """
        filtered = []

        for entity in entities:
            line_count = entity.line_count

            # Skip trivial functions
            if line_count < self.MIN_LINES_FOR_ANALYSIS:
                continue

            # Skip very long functions (likely data, not logic)
            if line_count > self.MAX_LINES_FOR_ANALYSIS:
                continue

            # Skip simple getters (lines = 2-3, just return statement)
            if self._is_simple_getter(entity):
                continue

            filtered.append(entity)

        return filtered

    def _is_simple_getter(self, entity) -> bool:
        """Check if entity is a simple getter/setter.

        Args:
            entity: Code entity to check

        Returns:
            True if simple getter/setter
        """
        content = entity.content.lower()

        # Simple patterns for getters/setters
        getter_patterns = ["return self.", "return self._"]
        setter_patterns = ["= value", "= new_value"]

        lines = content.split("\n")
        non_empty_lines = [l.strip() for l in lines if l.strip()]

        # Very short functions that just return
        if len(non_empty_lines) <= 3:
            if any(p in content for p in getter_patterns):
                return True
            if entity.name.startswith(("get_", "set_")):
                return True

        return False

    @observe(name="llm_bug_detection", as_type="span")
    async def _analyze_entity(
        self,
        entity,
        file_path: Path,
        full_content: str,
    ) -> list[DetectedBug] | None:
        """Analyze a single code entity with LLM.

        Args:
            entity: Code entity to analyze
            file_path: Source file path
            full_content: Full file content for context

        Returns:
            List of detected bugs or None
        """
        prompt = self._build_analysis_prompt(entity, full_content)

        # Call LLM with Langfuse observability (v4 API)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.llm_client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            # Update the current Langfuse generation
            update_current_generation(
                model=getattr(self.llm_client, "model_name", "unknown"),
                model_parameters={
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    "provider": getattr(self.llm_client, "provider_name", "unknown"),
                    "agent_name": "LLMBugDetector",
                },
            )
        except Exception as e:
            logger.warning(f"LLM call failed for {entity.name}: {e}")
            return None

        # Parse response
        return self._parse_llm_response(response.content, entity, file_path)

    def _system_prompt(self) -> str:
        """System prompt for LLM bug detection."""
        return """You are an expert code reviewer and security analyst.
Your task is to analyze Python code and identify potential bugs, issues, and improvements.

Focus on finding:
1. Logic errors and edge cases
2. Missing input validation
3. Security vulnerabilities (SQL injection, XSS, etc.)
4. Performance issues
5. Error handling problems
6. Resource leaks
7. Concurrency issues

For each issue found, provide:
- line: Line number where issue occurs
- severity: ERROR, WARNING, or INFO
- message: Clear description of the issue
- suggestion: How to fix it

Respond with JSON in this format:
{
    "issues": [
        {
            "line": 15,
            "severity": "WARNING",
            "message": "Division by zero not handled",
            "suggestion": "Add check: if b == 0: raise ValueError(...)"
        }
    ]
}

If no issues found, return: {"issues": []}"""

    def _build_analysis_prompt(self, entity, full_content: str) -> str:
        """Build analysis prompt for a specific entity.

        Args:
            entity: Code entity
            full_content: Full file content

        Returns:
            Prompt string
        """
        return f"""Analyze this Python function and identify any bugs or issues:

Function: {entity.qualified_name}
Location: {entity.location}

```python
{entity.content}
```

Context (imports from file):
{self._extract_imports(full_content)}

Identify potential bugs, edge cases, security issues, or improvements.
Be specific about line numbers and severity levels."""

    def _extract_imports(self, content: str) -> str:
        """Extract import statements from file content.

        Args:
            content: File content

        Returns:
            Import statements as string
        """
        lines = content.split("\n")
        imports = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                imports.append(stripped)

        return "\n".join(imports[:10])  # Limit to first 10 imports

    def _parse_llm_response(
        self,
        response: str,
        entity,
        file_path: Path,
    ) -> list[DetectedBug]:
        """Parse LLM response into DetectedBug objects.

        Args:
            response: LLM response text
            entity: Code entity that was analyzed
            file_path: Source file path

        Returns:
            List of detected bugs
        """
        bugs: list[DetectedBug] = []

        # Try to extract JSON from response
        try:
            # Look for JSON in response
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                data = json.loads(json_str)
            else:
                data = json.loads(response)

            issues = data.get("issues", [])

            for issue in issues:
                line_num = issue.get("line", entity.start_line)
                severity_str = issue.get("severity", "WARNING").upper()
                severity = BugSeverity.WARNING
                if severity_str == "ERROR":
                    severity = BugSeverity.ERROR
                elif severity_str == "INFO":
                    severity = BugSeverity.INFO

                bug = DetectedBug(
                    source=BugSource.LLM_DISCOVERY,
                    severity=severity,
                    location=CodeLocation(
                        file_path=file_path,
                        line_number=line_num,
                    ),
                    message=issue.get("message", "Issue found"),
                    error_code="llm-discovery",
                    suggested_fix=issue.get("suggestion"),
                    code_snippet=entity.content[:200],
                    metadata={
                        "function": entity.qualified_name,
                        "analyzed_by": "LLM",
                    },
                )
                bugs.append(bug)

        except json.JSONDecodeError:
            # LLM didn't return valid JSON
            pass
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return bugs
