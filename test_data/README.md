# Test Data

This directory contains sample files with intentional bugs for testing the bug fixing system.

## Files

- `sample_bug.py` - Contains multiple Python bugs including:
  - Division by zero
  - Empty list handling
  - None value handling
  - Dictionary access bugs
  - Class method validation issues
  - List modification during iteration

## Usage

1. First, ingest this test data:
   ```bash
   python main.py --ingest --directory ./test_data
   ```

2. Then test bug fixing:
   ```bash
   python main.py --fix "Fix the divide_numbers function to handle division by zero"
   ```

3. Or test in interactive mode:
   ```bash
   python main.py --directory ./test_data
   ```
