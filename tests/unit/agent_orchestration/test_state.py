"""Unit tests for AgentState.

Tests the Pydantic state model for agent orchestration.
"""

import pytest
from pydantic import ValidationError
from src.fix_agent_orchestration.domain.state import AgentState
from src.ingestion.domain.entities import CodeEntity, EntityType


class TestAgentState:
    """Test suite for AgentState."""

    @pytest.fixture
    def basic_state(self):
        """Create a basic agent state."""
        return AgentState(user_goal="Fix the division by zero bug")

    @pytest.fixture
    def sample_entities(self):
        """Create sample code entities."""
        return [
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="divide_numbers",
                content="def divide_numbers(a, b): return a / b",
                file_path=__file__,
                start_line=1,
                end_line=2,
                docstring="Divide two numbers",
            ),
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="calculate_average",
                content="def calculate_average(nums): return sum(nums) / len(nums)",
                file_path=__file__,
                start_line=10,
                end_line=12,
            ),
        ]

    def test_basic_construction(self):
        """Test basic state construction."""
        state = AgentState(user_goal="Fix the bug")

        assert state.user_goal == "Fix the bug"
        assert state.max_retries == 3
        assert state.retry_count == 0
        assert state.status == "planning"
        assert not state.is_approved

    def test_user_goal_validation(self):
        """Test that short user goals are rejected."""
        with pytest.raises(ValidationError, match="at least 10 characters"):
            AgentState(user_goal="Fix")

    def test_max_retries_validation(self):
        """Test max_retries validation."""
        AgentState(user_goal="Fix the bug", max_retries=0)
        AgentState(user_goal="Fix the bug", max_retries=10)

        with pytest.raises(ValueError):
            AgentState(user_goal="Fix the bug", max_retries=-1)

        with pytest.raises(ValueError):
            AgentState(user_goal="Fix the bug", max_retries=11)

    def test_can_retry(self, basic_state):
        """Test retry capability checking."""
        assert basic_state.can_retry()  # retry_count < max_retries

        basic_state.retry_count = 3
        assert not basic_state.can_retry()  # retry_count == max_retries

        basic_state.retry_count = 4
        assert not basic_state.can_retry()  # retry_count > max_retries

    def test_get_context_summary_no_entities(self, basic_state):
        """Test context summary with no entities."""
        summary = basic_state.get_context_summary()
        assert "No context retrieved" in summary

    def test_get_context_summary_with_entities(self, basic_state, sample_entities):
        """Test context summary with entities."""
        basic_state.retrieved_context = sample_entities
        summary = basic_state.get_context_summary()

        assert "Retrieved 2 code entities" in summary
        assert "divide_numbers" in summary
        assert "calculate_average" in summary
        assert "Divide two numbers" in summary  # Docstring

    def test_get_fix_context_no_fix(self, basic_state):
        """Test fix context with no fix."""
        context = basic_state.get_fix_context()
        assert context == ""

    def test_get_fix_context_with_fix(self, basic_state, sample_entities):
        """Test fix context with fix and feedback."""
        basic_state.retrieved_context = sample_entities
        basic_state.proposed_fix = "def divide_numbers(a, b): return a / b if b != 0 else 0"
        basic_state.review_feedback = "Handle the edge case"

        context = basic_state.get_fix_context()

        assert "Relevant Code" in context
        assert "divide_numbers" in context
        assert "Proposed Fix" in context
        assert "def divide_numbers" in context
        assert "Reviewer Feedback" in context
        assert "Handle the edge case" in context

    def test_log_transition(self, basic_state):
        """Test state transition logging."""
        basic_state.log_transition("planning", "coding", "Query generated")

        assert len(basic_state.history) == 1
        assert basic_state.history[0]["from"] == "planning"
        assert basic_state.history[0]["to"] == "coding"
        assert basic_state.history[0]["note"] == "Query generated"
        assert "timestamp" in basic_state.history[0]

    def test_to_retry_state(self, basic_state, sample_entities):
        """Test creating retry state."""
        basic_state.search_query = "find division functions"
        basic_state.extracted_keywords = ["divide", "zero"]
        basic_state.retrieved_context = sample_entities
        basic_state.review_feedback = "Add type hints"
        basic_state.review_issues = ["Missing type hints"]

        retry_state = basic_state.to_retry_state()

        assert retry_state.retry_count == 1
        assert retry_state.status == "coding"
        assert retry_state.search_query == basic_state.search_query
        assert retry_state.retrieved_context == basic_state.retrieved_context
        assert retry_state.review_feedback == basic_state.review_feedback
        assert len(retry_state.history) > 0

    def test_model_dump_for_prompt(self, basic_state, sample_entities):
        """Test model dump for prompt context."""
        basic_state.search_query = "test query"
        basic_state.proposed_fix = "test fix"
        basic_state.retrieved_context = sample_entities

        data = basic_state.model_dump_for_prompt()

        assert "user_goal" in data
        assert "retrieved_count" in data
        assert data["retrieved_count"] == 2
        assert "proposed_fix" in data
        assert "is_approved" in data
        # Should not include full context
        assert "retrieved_context" not in data
