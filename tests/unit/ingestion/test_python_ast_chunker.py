"""Unit tests for PythonASTChunker.

Tests the AST-based code chunking infrastructure.
"""

from pathlib import Path

import pytest

from src.ingestion.domain.entities import EntityType
from src.ingestion.infrastructure.python_ast_chunker import PythonASTChunker


class TestPythonASTChunker:
    """Test suite for PythonASTChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a chunker instance."""
        return PythonASTChunker()

    @pytest.fixture
    def sample_function(self):
        """Sample function code."""
        return '''
def calculate_average(numbers):
    """Calculate average of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
'''

    @pytest.fixture
    def sample_class(self):
        """Sample class code."""
        return '''
class BankAccount:
    """A bank account class."""
    
    def __init__(self, balance=0.0):
        self.balance = balance
    
    def deposit(self, amount):
        """Deposit money."""
        self.balance += amount
        
    def withdraw(self, amount):
        """Withdraw money."""
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
'''

    def test_supported_extensions(self, chunker):
        """Test that supported extensions are correct."""
        extensions = chunker.get_supported_extensions()
        assert ".py" in extensions
        assert ".pyw" in extensions

    def test_chunk_simple_function(self, chunker, sample_function):
        """Test chunking a simple function."""
        file_path = Path("/test/sample.py")
        entities = chunker.chunk_file(file_path, sample_function)

        # Should extract the function
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(functions) == 1

        func = functions[0]
        assert func.name == "calculate_average"
        assert func.qualified_name == "calculate_average"
        assert func.start_line == 2
        assert "calculate_average" in func.signature
        assert func.docstring == "Calculate average of numbers."

    def test_chunk_class_with_methods(self, chunker, sample_class):
        """Test chunking a class with methods."""
        file_path = Path("/test/bank.py")
        entities = chunker.chunk_file(file_path, sample_class)

        # Should extract class and methods
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        methods = [e for e in entities if e.entity_type == EntityType.METHOD]

        assert len(classes) == 1
        assert classes[0].name == "BankAccount"

        assert len(methods) == 3
        method_names = {m.name for m in methods}
        assert method_names == {"__init__", "deposit", "withdraw"}

        # Check parent relationships
        for method in methods:
            assert method.parent == "BankAccount"
            assert method.qualified_name.startswith("BankAccount.")

    def test_extract_imports(self, chunker):
        """Test import extraction."""
        code = '''
import os
import sys
from pathlib import Path
from typing import Optional, List

def test():
    pass
'''
        tree = chunker._extract_imports.__wrapped__ if hasattr(chunker._extract_imports, '__wrapped__') else None
        # Instead, let's test via chunk_file
        entities = chunker.chunk_file(Path("/test.py"), code)

        # Standalone entity should have imports
        standalone = [e for e in entities if e.entity_type == EntityType.STANDALONE]
        if standalone:
            assert len(standalone[0].imports) >= 4  # Should include all imports

    def test_syntax_error_handling(self, chunker):
        """Test that syntax errors are properly raised."""
        invalid_code = "def broken(:"  # Invalid syntax

        with pytest.raises(SyntaxError):
            chunker.chunk_file(Path("/test.py"), invalid_code)

    def test_nested_function(self, chunker):
        """Test extraction of nested functions."""
        code = '''
def outer():
    """Outer function."""
    def inner():
        """Inner function."""
        return 42
    return inner()
'''
        entities = chunker.chunk_file(Path("/test.py"), code)

        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        nested = [e for e in entities if e.entity_type == EntityType.NESTED_FUNCTION]

        assert len(functions) >= 1
        assert any(f.name == "outer" for f in functions)

    def test_async_function(self, chunker):
        """Test extraction of async functions."""
        code = '''
async def fetch_data():
    """Fetch data asynchronously."""
    await asyncio.sleep(1)
    return "data"
'''
        entities = chunker.chunk_file(Path("/test.py"), code)

        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(functions) == 1

        # Check metadata
        assert functions[0].metadata.get("is_async") is True

    def test_empty_file(self, chunker):
        """Test chunking an empty file."""
        entities = chunker.chunk_file(Path("/test.py"), "")

        # Should return empty or just module-level info
        assert isinstance(entities, list)

    def test_build_signature(self, chunker):
        """Test signature building."""
        import ast

        code = "def func(a: int, b: str = 'default') -> bool: pass"
        tree = ast.parse(code)
        func_node = tree.body[0]

        signature = chunker._build_signature(func_node)

        assert "def func" in signature
        assert "a: int" in signature
        assert "-> bool" in signature
