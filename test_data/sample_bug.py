"""Sample file with intentional bugs for testing.

This module contains various bugs that can be used to test the
bug fixing capabilities of the AgenticSource system.
"""

from typing import Any


def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers.

    BUG: Does not handle division by zero!

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result of a / b
    """
    # Bug: No check for zero division
    return a / b


def calculate_average(numbers: list[float]) -> float:
    """Calculate average of a list of numbers.

    BUG: Does not handle empty list!

    Args:
        numbers: List of numbers

    Returns:
        Average value
    """
    # Bug: Should check if list is empty
    return sum(numbers) / len(numbers)


def find_maximum(values: list[Any]) -> Any | None:
    """Find the maximum value in a list.

    BUG: Does not handle None values properly!

    Args:
        values: List of comparable values

    Returns:
        Maximum value or None if list is empty
    """
    if not values:
        return None

    # Bug: Does not handle None in the list
    max_val = values[0]
    for val in values[1:]:
        if val > max_val:
            max_val = val
    return max_val


def safe_get(dictionary: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary.

    This function has a subtle bug where it doesn't properly handle
    nested key access.

    Args:
        dictionary: Dictionary to search
        key: Key to look up
        default: Default value if key not found

    Returns:
        Value or default
    """
    # Bug: Should use dictionary.get(key, default) instead
    if key in dictionary:
        return dictionary[key]
    return default


class BankAccount:
    """Simple bank account with intentional bug.

    BUG: Withdraw allows negative amounts!
    """

    def __init__(self, balance: float = 0.0) -> None:
        """Initialize account.

        BUG: Should validate balance is non-negative!
        """
        self.balance = balance
        self.transactions: list[dict] = []

    def deposit(self, amount: float) -> None:
        """Deposit money.

        BUG: Should validate amount is positive!
        """
        self.balance += amount
        self.transactions.append({"type": "deposit", "amount": amount})

    def withdraw(self, amount: float) -> bool:
        """Withdraw money.

        BUGS:
        1. Does not check if amount is positive
        2. Does not check if sufficient funds

        Returns:
            True if withdrawal succeeded
        """
        self.balance -= amount
        self.transactions.append({"type": "withdrawal", "amount": amount})
        return True

    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance


def format_user_data(user: dict) -> str:
    """Format user data as a string.

    BUG: Potential KeyError if fields missing!

    Args:
        user: Dictionary with user data

    Returns:
        Formatted string
    """
    # Bug: Should use .get() to handle missing keys
    return f"Name: {user['name']}, Email: {user['email']}, Age: {user['age']}"


def process_items(items: list[str]) -> list[str]:
    """Process a list of items.

    BUG: Modifies list while iterating!

    Args:
        items: List of items

    Returns:
        Processed items
    """
    # Bug: Should create a new list instead of modifying
    for i, item in enumerate(items):
        if item.startswith("temp_"):
            items.remove(item)
    return items
