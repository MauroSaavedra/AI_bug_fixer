"""Unit tests for test_failure_parser.py infrastructure.

Tests the TestFailureParser's ability to parse pytest and JUnit XML output.
"""

from pathlib import Path

import pytest

from src.detection.infrastructure.test_failure_parser import TestFailureParser
from src.detection.domain.entities import BugSeverity, BugSource


class TestTestFailureParserInit:
    """Test TestFailureParser initialization."""

    def test_init_no_path(self):
        """Default init with no path."""
        parser = TestFailureParser()
        assert parser.test_output_path is None
        assert parser.is_available is True

    def test_init_with_path(self, tmp_path):
        """Init with a valid path."""
        output_file = tmp_path / "output.txt"
        output_file.write_text("test output")
        parser = TestFailureParser(output_file)
        assert parser.test_output_path == output_file
        assert parser.is_available is True

    def test_init_with_missing_path(self, tmp_path):
        """Init with a non-existent path."""
        output_file = tmp_path / "nonexistent.txt"
        parser = TestFailureParser(output_file)
        assert parser.is_available is False


class TestParseJUnitXML:
    """Test JUnit XML parsing."""

    JUNIT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="pytest" tests="3" failures="1" errors="1">
        <testcase name="test_pass" file="test_sample.py" line="5" time="0.01"/>
        <testcase name="test_fail" file="test_sample.py" line="10" time="0.02">
            <failure message="assertion failed">Traceback data</failure>
        </testcase>
        <testcase name="test_error" file="test_sample.py" line="15" time="0.01">
            <error message="division by zero">Error details</error>
        </testcase>
    </testsuite>
</testsuites>"""

    def test_parse_junit_xml(self, tmp_path):
        """Parse JUnit XML with failures."""
        xml_file = tmp_path / "junit.xml"
        xml_file.write_text(self.JUNIT_XML)
        parser = TestFailureParser(xml_file)

        bugs = parser.detect(tmp_path)
        assert len(bugs) == 2

        failure_bug = [b for b in bugs if "failure" in b.message.lower()][0]
        assert failure_bug.source == BugSource.PYTEST
        assert failure_bug.severity == BugSeverity.ERROR
        assert failure_bug.location.line_number == 10
        assert "test_fail" in failure_bug.message

        error_bug = [b for b in bugs if "error" in b.message.lower()][0]
        assert "test_error" in error_bug.message
        assert error_bug.location.line_number == 15

    def test_parse_junit_xml_no_failures(self, tmp_path):
        """Parse JUnit XML with no failures."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="pytest" tests="2" failures="0">
        <testcase name="test_pass" file="test.py" line="1" time="0.01"/>
    </testsuite>
</testsuites>"""
        xml_file = tmp_path / "junit.xml"
        xml_file.write_text(xml)
        parser = TestFailureParser(xml_file)

        bugs = parser.detect(tmp_path)
        assert len(bugs) == 0

    def test_parse_invalid_xml(self, tmp_path):
        """Gracefully handle invalid XML."""
        xml_file = tmp_path / "junit.xml"
        xml_file.write_text("not xml at all")
        parser = TestFailureParser(xml_file)

        bugs = parser._parse_junit_xml(xml_file)
        assert len(bugs) == 0


class TestParsePytestJSON:
    """Test pytest JSON report parsing."""

    def test_parse_pytest_json(self, tmp_path):
        """Parse pytest JSON report."""
        json_content = """
{
    "tests": [
        {
            "nodeid": "test_sample.py::test_add",
            "outcome": "failed",
            "call": {
                "crash": {"path": "test_sample.py", "lineno": 42}
            }
        }
    ]
}
"""
        json_file = tmp_path / "report.json"
        json_file.write_text(json_content)
        parser = TestFailureParser(json_file)

        bugs = parser.detect_file(json_file)
        assert len(bugs) == 1
        assert "test_sample.py::test_add" in bugs[0].message
        assert "Test failed" in bugs[0].message
        assert bugs[0].location.line_number == 42

    def test_parse_pytest_json_no_failures(self, tmp_path):
        """Parse JSON with no failures."""
        json_file = tmp_path / "report.json"
        json_file.write_text('{"tests": [{"outcome": "passed", "nodeid": "test.py::test_x"}]}')
        parser = TestFailureParser(json_file)

        bugs = parser._parse_pytest_json(json_file)
        assert len(bugs) == 0

    def test_parse_invalid_json(self, tmp_path):
        """Handle invalid JSON."""
        json_file = tmp_path / "report.json"
        json_file.write_text("not json")
        parser = TestFailureParser(json_file)

        bugs = parser._parse_pytest_json(json_file)
        assert len(bugs) == 0


class TestParsePytestText:
    """Test pytest terminal output parsing."""

    def test_parse_pytest_text(self, tmp_path):
        """Parse pytest text output."""
        text = "FAILED test_sample.py::test_add - assert 1 == 2\n\nself = <test_sample.TestCalc object at 0x7f8a>\n\n    def test_add(self):\n>       assert 1 == 2\nE       assert 1 == 2\n\nFile \"test_sample.py\", line 10, in test_add\n    assert 1 == 2\n"
        text_file = tmp_path / "pytest_output.txt"
        text_file = tmp_path / "pytest_output.txt"
        text_file.write_text(text)
        parser = TestFailureParser(text_file)

        bugs = parser._parse_pytest_text(text_file)
        assert len(bugs) == 1
        assert bugs[0].location.file_path.name == "test_sample.py"
        assert "test_add" in bugs[0].message

    def test_parse_pytest_text_errors(self, tmp_path):
        """Parse ERROR entries in pytest output."""
        text = "ERROR test_sample.py::test_setup - fixture not found"
        text_file = tmp_path / "pytest_output.txt"
        text_file.write_text(text)
        parser = TestFailureParser(text_file)

        bugs = parser._parse_pytest_text(text_file)
        assert len(bugs) == 1
        assert "error" in bugs[0].message.lower()

    def test_no_failures(self, tmp_path):
        """Text output with no failures."""
        text = "test_sample.py::test_pass PASSED"
        text_file = tmp_path / "pytest_output.txt"
        text_file.write_text(text)
        parser = TestFailureParser(text_file)

        bugs = parser._parse_pytest_text(text_file)
        assert len(bugs) == 0


class TestExtractStackFrame:
    """Test stack frame extraction."""

    @pytest.fixture
    def parser(self):
        return TestFailureParser()

    def test_extract_first_frame(self, parser):
        """Extract first (most recent) stack frame."""
        traceback = """Traceback (most recent call last):
  File "/home/user/project/main.py", line 42, in process_data
    result = fetch_data()
  File "/home/user/project/utils.py", line 10, in fetch_data
    raise ConnectionError()"""
        frame = parser.extract_stack_frame(traceback, frame_index=0)
        assert frame is not None
        assert frame["file"] == "/home/user/project/main.py"
        assert frame["line"] == 42
        assert frame["function"] == "process_data"

    def test_extract_second_frame(self, parser):
        """Extract second stack frame."""
        traceback = """Traceback (most recent call last):
  File "/home/user/main.py", line 10, in main
    run()
  File "/home/user/lib.py", line 20, in run
    pass"""
        # frame_index=0 is the first match in the list, which is
        # the first line that matches the pattern
        # In the traceback, frame 0 is "main", frame 1 is "run"
        frame = parser.extract_stack_frame(traceback, frame_index=1)
        assert frame is not None
        assert frame["file"] == "/home/user/lib.py"
        assert frame["line"] == 20
        assert frame["function"] == "run"

    def test_empty_traceback(self, parser):
        """Handle empty traceback."""
        frame = parser.extract_stack_frame("", frame_index=0)
        assert frame is None

    def test_frame_index_out_of_range(self, parser):
        """Handle out of range index."""
        traceback = "File \"main.py\", line 10, in main"
        frame = parser.extract_stack_frame(traceback, frame_index=10)
        assert frame is None


class TestDetectDirectory:
    """Test detect() on directory."""

    def test_detect_finds_junit_xml(self, tmp_path):
        """Detect finds JUnit XML in directory."""
        xml_content = """<?xml version="1.0"?>
<testsuites>
    <testcase name="test_fail" file="test.py" line="1">
        <failure message="failed">trace</failure>
    </testcase>
</testsuites>"""
        (tmp_path / "junit.xml").write_text(xml_content)
        parser = TestFailureParser()

        bugs = parser.detect(tmp_path)
        assert len(bugs) == 1

    def test_detect_finds_test_results_xml(self, tmp_path):
        """Detect finds test-results.xml in directory."""
        xml_content = """<?xml version="1.0"?>
<testsuites>
    <testcase name="test_error" file="test.py" line="5">
        <error message="error">trace</error>
    </testcase>
</testsuites>"""
        (tmp_path / "test-results.xml").write_text(xml_content)
        parser = TestFailureParser()

        bugs = parser.detect(tmp_path)
        assert len(bugs) == 1
        assert bugs[0].location.line_number == 5

    def test_detect_no_files(self, tmp_path):
        """Detect on empty directory."""
        parser = TestFailureParser()
        bugs = parser.detect(tmp_path)
        assert len(bugs) == 0
