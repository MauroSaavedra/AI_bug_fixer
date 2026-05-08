"""Pytest configuration and fixtures.

This module provides shared fixtures for all tests.
"""

import pytest
from pathlib import Path

from src.ingestion.domain.entities import CodeEntity, EntityType


@pytest.fixture
def sample_function_entity():
    """Create a sample function entity."""
    return CodeEntity(
        entity_type=EntityType.FUNCTION,
        name="divide_numbers",
        content="""def divide_numbers(a: float, b: float) -> float:
    \"\"\"Divide two numbers.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result of division
    \"\"\"
    return a / b""",
        file_path=Path("/test/math_utils.py"),
        start_line=1,
        end_line=12,
        signature="def divide_numbers(a: float, b: float) -> float",
        docstring="Divide two numbers.",
    )


@pytest.fixture
def sample_class_entity():
    """Create a sample class entity."""
    return CodeEntity(
        entity_type=EntityType.CLASS,
        name="BankAccount",
        content="""class BankAccount:
    \"\"\"A bank account class.\"\"\"
    
    def __init__(self, balance=0.0):
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
    
    def withdraw(self, amount):
        self.balance -= amount""",
        file_path=Path("/test/bank.py"),
        start_line=1,
        end_line=13,
        signature="class BankAccount",
        docstring="A bank account class.",
    )


@pytest.fixture
def sample_method_entity():
    """Create a sample method entity."""
    return CodeEntity(
        entity_type=EntityType.METHOD,
        name="deposit",
        content="""def deposit(self, amount):
    \"\"\"Deposit money.\"\"\"
    self.balance += amount""",
        file_path=Path("/test/bank.py"),
        start_line=7,
        end_line=9,
        parent="BankAccount",
        signature="def deposit(self, amount)",
        docstring="Deposit money.",
    )


@pytest.fixture
def sample_entities(sample_function_entity, sample_class_entity, sample_method_entity):
    """Create a list of sample entities."""
    return [
        sample_function_entity,
        sample_class_entity,
        sample_method_entity,
    ]


@pytest.fixture
def temp_chroma_db(tmp_path):
    """Create a temporary ChromaDB path."""
    return tmp_path / "chroma_db"
