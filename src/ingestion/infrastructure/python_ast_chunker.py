"""Python AST-based code chunker for semantic code extraction.

This module implements IChunker using Python's built-in ast module to parse
source code and extract semantic entities: functions, classes, methods, and
standalone code blocks with accurate line tracking.
"""

import ast
import re
from pathlib import Path
from typing import Any

from src.ingestion.domain.entities import CodeEntity, EntityType
from src.ingestion.domain.interfaces import IChunker


class PythonASTChunker(IChunker):
    """Extract semantic code entities from Python source files using AST.

    This chunker parses Python files and extracts:
    - Functions (standalone)
    - Classes (with their methods)
    - Methods (with class context)
    - Nested functions (with parent context)
    - Standalone code (imports, assignments, etc. grouped)

    Each entity includes precise line numbers, signatures, and docstrings.
    """

    SUPPORTED_EXTENSIONS = {".py", ".pyw"}

    def __init__(self, include_standalone: bool = True):
        """Initialize the chunker.

        Args:
            include_standalone: Whether to include standalone code blocks
                (imports, assignments, etc.) as chunks
        """
        self.include_standalone = include_standalone

    def get_supported_extensions(self) -> set[str]:
        """Get supported file extensions."""
        return self.SUPPORTED_EXTENSIONS.copy()

    def chunk_file(self, file_path: Path, content: str) -> list[CodeEntity]:
        """Parse a Python file and extract semantic entities.

        Args:
            file_path: Path to the Python file
            content: File content as string

        Returns:
            List of code entities extracted from the file

        Raises:
            SyntaxError: If file contains syntax errors
        """
        # Parse the AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse {file_path}: {e}") from e

        # Split content into lines for extraction
        lines = content.split("\n")

        # Collect all entities
        entities: list[CodeEntity] = []
        covered_lines: set[int] = set()

        # Extract module-level docstring and imports
        module_docstring = self._extract_module_docstring(tree)
        module_imports = self._extract_imports(tree, lines)

        # Walk the AST and extract entities
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                entity = self._extract_function(
                    node, file_path, lines, module_imports
                )
                entities.append(entity)
                covered_lines.update(range(entity.start_line, entity.end_line + 1))

            elif isinstance(node, ast.ClassDef):
                class_entity, class_methods = self._extract_class(
                    node, file_path, lines, module_imports
                )
                entities.append(class_entity)
                entities.extend(class_methods)
                covered_lines.update(range(class_entity.start_line, class_entity.end_line + 1))
                for method in class_methods:
                    covered_lines.update(range(method.start_line, method.end_line + 1))

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # Imports are handled separately, mark as covered
                if hasattr(node, "lineno"):
                    covered_lines.add(node.lineno)

        # Extract standalone code if enabled
        if self.include_standalone:
            standalone = self._extract_standalone(
                tree, file_path, lines, covered_lines, module_imports
            )
            entities.extend(standalone)

        # Sort entities by line number for consistent ordering
        entities.sort(key=lambda e: (e.start_line, e.end_line))

        return entities

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        lines: list[str],
        imports: list[str],
        parent: str | None = None,
    ) -> CodeEntity:
        """Extract a function entity from AST node.

        Args:
            node: Function definition node
            file_path: Path to source file
            lines: File content split by lines
            imports: List of imports in scope
            parent: Parent entity name if nested

        Returns:
            CodeEntity representing the function
        """
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Extract the raw source
        content = self._extract_source(lines, start_line, end_line)

        # Build signature
        signature = self._build_signature(node)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Determine entity type
        if parent:
            entity_type = EntityType.NESTED_FUNCTION
        elif isinstance(node, ast.AsyncFunctionDef):
            entity_type = EntityType.FUNCTION  # Async functions are still functions
        else:
            entity_type = EntityType.FUNCTION

        # Check for nested functions and extract them
        nested_entities = self._extract_nested_functions(node, file_path, lines, imports)

        return CodeEntity(
            entity_type=entity_type,
            name=node.name,
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            parent=parent,
            docstring=docstring,
            signature=signature,
            imports=imports,
            metadata={
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "has_nested": len(nested_entities) > 0,
                "decorator_count": len(node.decorator_list),
            },
        )

    def _extract_nested_functions(
        self,
        parent_node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        file_path: Path,
        lines: list[str],
        imports: list[str],
    ) -> list[CodeEntity]:
        """Extract nested functions from a parent node."""
        nested: list[CodeEntity] = []
        parent_name = parent_node.name

        for child in ast.walk(parent_node):
            if child is parent_node:
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Determine if direct child (not nested within another function)
                if self._is_direct_child(child, parent_node):
                    entity = self._extract_function(
                        child, file_path, lines, imports, parent=parent_name
                    )
                    nested.append(entity)

        return nested

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        lines: list[str],
        imports: list[str],
    ) -> tuple[CodeEntity, list[CodeEntity]]:
        """Extract a class and its methods.

        Args:
            node: Class definition node
            file_path: Path to source file
            lines: File content split by lines
            imports: List of imports in scope

        Returns:
            Tuple of (class entity, list of method entities)
        """
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Extract class source
        content = self._extract_source(lines, start_line, end_line)

        # Build class signature
        bases = [self._ast_to_string(base) for base in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Create class entity
        class_entity = CodeEntity(
            entity_type=EntityType.CLASS,
            name=node.name,
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            parent=None,
            docstring=docstring,
            signature=signature,
            imports=imports,
            metadata={
                "base_classes": bases,
                "decorator_count": len(node.decorator_list),
                "method_count": len(
                    [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                ),
            },
        )

        # Extract methods
        methods: list[CodeEntity] = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._extract_method(
                    child, file_path, lines, imports, class_name=node.name
                )
                methods.append(method)

        return class_entity, methods

    def _extract_method(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        lines: list[str],
        imports: list[str],
        class_name: str,
    ) -> CodeEntity:
        """Extract a method from a class.

        Args:
            node: Method definition node
            file_path: Path to source file
            lines: File content split by lines
            imports: List of imports in scope
            class_name: Name of the containing class

        Returns:
            CodeEntity representing the method
        """
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        content = self._extract_source(lines, start_line, end_line)
        signature = self._build_signature(node)
        docstring = ast.get_docstring(node)

        # Determine special method types
        method_type = "instance"
        if node.name == "__init__":
            method_type = "constructor"
        elif node.args.args and node.args.args[0].arg == "cls":
            method_type = "class_method"
        elif any(
            isinstance(d, ast.Name) and d.id == "staticmethod" for d in node.decorator_list
        ):
            method_type = "static_method"

        return CodeEntity(
            entity_type=EntityType.METHOD,
            name=node.name,
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            parent=class_name,
            docstring=docstring,
            signature=signature,
            imports=imports,
            metadata={
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "method_type": method_type,
                "decorator_count": len(node.decorator_list),
            },
        )

    def _extract_standalone(
        self,
        tree: ast.Module,
        file_path: Path,
        lines: list[str],
        covered_lines: set[int],
        imports: list[str],
    ) -> list[CodeEntity]:
        """Extract standalone code blocks (imports, assignments, etc.).

        Groups consecutive uncovered statements into logical blocks.
        """
        entities: list[CodeEntity] = []
        current_block_start: int | None = None
        current_block_end: int | None = None

        for node in ast.iter_child_nodes(tree):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                line_range = range(node.lineno, (node.end_lineno or node.lineno) + 1)

                # Check if any line in this node is uncovered
                if any(l not in covered_lines for l in line_range):
                    if current_block_start is None:
                        current_block_start = node.lineno
                        current_block_end = node.end_lineno or node.lineno
                    else:
                        # Extend current block if adjacent
                        if node.lineno <= current_block_end + 1:
                            current_block_end = max(current_block_end, node.end_lineno or node.lineno)
                        else:
                            # Finalize previous block
                            entities.append(
                                self._create_standalone_entity(
                                    file_path, lines, current_block_start, current_block_end, imports
                                )
                            )
                            current_block_start = node.lineno
                            current_block_end = node.end_lineno or node.lineno

        # Don't forget the last block
        if current_block_start is not None and current_block_end is not None:
            entities.append(
                self._create_standalone_entity(
                    file_path, lines, current_block_start, current_block_end, imports
                )
            )

        return entities

    def _create_standalone_entity(
        self,
        file_path: Path,
        lines: list[str],
        start_line: int,
        end_line: int,
        imports: list[str],
    ) -> CodeEntity:
        """Create a CodeEntity for standalone code."""
        content = self._extract_source(lines, start_line, end_line)

        # Generate a name based on content
        name = self._generate_standalone_name(content)

        return CodeEntity(
            entity_type=EntityType.STANDALONE,
            name=name,
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            parent=None,
            docstring=None,
            signature=None,
            imports=imports,
            metadata={"is_standalone": True},
        )

    def _extract_source(self, lines: list[str], start_line: int, end_line: int) -> str:
        """Extract source code given line range (1-indexed)."""
        # Adjust for 0-indexed list
        start_idx = start_line - 1
        end_idx = end_line
        return "\n".join(lines[start_idx:end_idx])

    def _build_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str:
        """Build function signature string from AST node."""
        parts = []

        # Add async keyword
        if isinstance(node, ast.AsyncFunctionDef):
            parts.append("async")

        parts.append("def")
        parts.append(node.name)

        # Build argument string
        args = self._format_arguments(node.args)
        parts.append(f"({args})")

        # Add return annotation if present
        if node.returns:
            parts.append(f"-> {self._ast_to_string(node.returns)}")

        return " ".join(parts)

    def _format_arguments(self, args: ast.arguments) -> str:
        """Format function arguments as string."""
        parts: list[str] = []

        # Positional-only args
        for arg in args.posonlyargs:
            parts.append(self._format_arg(arg))

        if args.posonlyargs:
            parts.append("/")

        # Regular args
        for arg in args.args:
            parts.append(self._format_arg(arg))

        # Vararg
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")

        # Keyword-only args
        if args.kwonlyargs:
            if not args.vararg:
                parts.append("*")
            for arg in args.kwonlyargs:
                parts.append(self._format_arg(arg))

        # Kwarg
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        return ", ".join(parts)

    def _format_arg(self, arg: ast.arg) -> str:
        """Format a single argument."""
        result = arg.arg
        if arg.annotation:
            result += f": {self._ast_to_string(arg.annotation)}"
        return result

    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node back to source string.

        Uses ast.unparse for Python 3.9+, falls back to repr for older versions.
        """
        try:
            return ast.unparse(node)
        except AttributeError:
            # Fallback for older Python versions
            return repr(node)

    def _extract_module_docstring(self, tree: ast.Module) -> str | None:
        """Extract module-level docstring."""
        return ast.get_docstring(tree)

    def _extract_imports(self, tree: ast.Module, lines: list[str]) -> list[str]:
        """Extract all import statements from the module."""
        imports: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                imports.append(f"import {', '.join(names)}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                level = "." * node.level
                imports.append(f"from {level}{module} import {', '.join(names)}")

        return imports

    def _is_direct_child(
        self, child: ast.AST, parent: ast.AST
    ) -> bool:
        """Check if child is a direct child of parent (not nested deeper)."""
        # Find the immediate parent of child
        for node in ast.walk(parent):
            if node is child:
                continue
            if hasattr(node, "body"):
                for subnode in node.body if isinstance(node.body, list) else [node.body]:
                    if subnode is child:
                        return node is parent
        return True

    def _generate_standalone_name(self, content: str) -> str:
        """Generate a descriptive name for standalone code block."""
        lines = content.strip().split("\n")
        if not lines:
            return "standalone_block"

        # Check for import statements
        import_pattern = re.compile(r"^(import|from)\s+(.+)$")
        assignments = []
        other_lines = []

        for line in lines:
            stripped = line.strip()
            if import_pattern.match(stripped):
                return "imports"
            elif "=" in stripped and not stripped.startswith("#"):
                # Extract variable name
                var_match = re.match(r"^(\w+)\s*=", stripped)
                if var_match:
                    assignments.append(var_match.group(1))
            elif stripped:
                other_lines.append(stripped)

        if assignments:
            return f"assignments_{'_'.join(assignments[:3])}"
        elif other_lines:
            first_word = re.match(r"^(\w+)", other_lines[0])
            if first_word:
                return f"code_{first_word.group(1)}"

        return "standalone_block"
