"""Unit tests for agent implementations.

Tests Planner, Coder, and Reviewer agents.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.fix_agent_orchestration.domain.interfaces import ILLMClient, LLMResponse
from src.fix_agent_orchestration.domain.state import AgentState
from src.fix_agent_orchestration.infrastructure.coder_agent import CoderAgent
from src.fix_agent_orchestration.infrastructure.planner_agent import PlannerAgent
from src.fix_agent_orchestration.infrastructure.reviewer_agent import ReviewerAgent
from src.ingestion.domain.entities import CodeEntity, EntityType


class TestPlannerAgent:
    """Test suite for PlannerAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        client = MagicMock(spec=ILLMClient)
        response = LLMResponse(
            content=json.dumps({
                "search_query": "division by zero error handling",
                "keywords": ["divide", "zero", "error"],
                "target_file": None,
                "reasoning": "Looking for division functions",
            }),
            provider="openai",
            model="gpt-4o",
        )
        client.chat = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def planner(self, mock_llm):
        """Create planner agent."""
        return PlannerAgent(mock_llm)

    @pytest.mark.asyncio
    async def test_execute_success(self, planner, mock_llm):
        """Test successful planning."""
        state = AgentState(user_goal="Fix division by zero bug")
        result = await planner.execute(state)

        assert result.search_query == "division by zero error handling"
        assert result.extracted_keywords == ["divide", "zero", "error"]
        assert result.target_file_hint is None
        assert result.status == "retrieving"

        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_handles_parse_error(self, planner, mock_llm):
        """Test handling of JSON parse error."""
        mock_llm.chat.return_value = LLMResponse(
            content="not valid json",
            provider="openai",
            model="gpt-4o",
        )

        state = AgentState(user_goal="Fix the bug")
        result = await planner.execute(state)

        # Should fall back to using user_goal
        assert result.search_query == state.user_goal


class TestCoderAgent:
    """Test suite for CoderAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        client = MagicMock(spec=ILLMClient)
        response = LLMResponse(
            content=json.dumps({
                "proposed_fix": "def divide(a, b): return a / b if b != 0 else float('inf')",
                "confidence_score": 0.95,
                "reasoning": "Added zero check",
                "files_modified": ["test.py"],
                "testing_suggestions": "Test with b=0",
            }),
            provider="openai",
            model="gpt-4o",
        )
        client.chat = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def coder(self, mock_llm):
        """Create coder agent."""
        return CoderAgent(mock_llm)

    @pytest.fixture
    def sample_entity(self):
        """Create sample code entity."""
        return CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="divide",
            content="def divide(a, b): return a / b",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=2,
        )

    @pytest.mark.asyncio
    async def test_execute_success(self, coder, mock_llm, sample_entity):
        """Test successful coding."""
        state = AgentState(user_goal="Fix division by zero")
        state.retrieved_context = [sample_entity]

        result = await coder.execute(state)

        assert result.proposed_fix == "def divide(a, b): return a / b if b != 0 else float('inf')"
        assert result.confidence_score == 0.95
        assert result.reasoning == "Added zero check"
        assert result.status == "reviewing"

        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_retry(self, coder, mock_llm, sample_entity):
        """Test execution on retry."""
        state = AgentState(user_goal="Fix division by zero")
        state.retry_count = 1
        state.review_feedback = "Add proper error handling"
        state.retrieved_context = [sample_entity]

        result = await coder.execute(state)

        assert result.proposed_fix is not None
        assert result.status == "reviewing"

    @pytest.mark.asyncio
    async def test_execute_handles_parse_error(self, coder, mock_llm, sample_entity):
        """Test handling of JSON parse error."""
        mock_llm.chat.return_value = LLMResponse(
            content="Just raw text without JSON",
            provider="openai",
            model="gpt-4o",
        )

        state = AgentState(user_goal="Fix the bug")
        state.retrieved_context = [sample_entity]
        result = await coder.execute(state)

        # Should use raw content as fix
        assert result.proposed_fix == "Just raw text without JSON"


class TestReviewerAgent:
    """Test suite for ReviewerAgent."""

    @pytest.fixture
    def mock_llm_approve(self):
        """Create mock LLM that approves."""
        client = MagicMock(spec=ILLMClient)
        response = LLMResponse(
            content=json.dumps({
                "is_approved": True,
                "feedback": "Fix looks correct",
                "issues": [],
                "suggestions": [],
                "severity": "none",
            }),
            provider="openai",
            model="gpt-4o",
        )
        client.chat = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def mock_llm_reject(self):
        """Create mock LLM that rejects."""
        client = MagicMock(spec=ILLMClient)
        response = LLMResponse(
            content=json.dumps({
                "is_approved": False,
                "feedback": "The fix doesn't handle all edge cases",
                "issues": ["Missing edge case"],
                "suggestions": ["Add type checking"],
                "severity": "moderate",
            }),
            provider="openai",
            model="gpt-4o",
        )
        client.chat = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def reviewer_approve(self, mock_llm_approve):
        """Create approving reviewer."""
        return ReviewerAgent(mock_llm_approve)

    @pytest.fixture
    def reviewer_reject(self, mock_llm_reject):
        """Create rejecting reviewer."""
        return ReviewerAgent(mock_llm_reject)

    @pytest.mark.asyncio
    async def test_execute_approval(self, reviewer_approve, mock_llm_approve):
        """Test approval flow."""
        state = AgentState(user_goal="Fix the bug")
        state.proposed_fix = "def fix(): pass"
        state.reasoning = "Simple fix"

        result = await reviewer_approve.execute(state)

        assert result.is_approved is True
        assert result.status == "approved"
        assert result.review_feedback == "Fix looks correct"

        mock_llm_approve.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_rejection_with_retry(self, reviewer_reject, mock_llm_reject):
        """Test rejection with retry available."""
        state = AgentState(user_goal="Fix the bug")
        state.proposed_fix = "def fix(): pass"
        state.retry_count = 0
        state.max_retries = 3

        result = await reviewer_reject.execute(state)

        assert result.is_approved is False
        assert result.status == "coding"  # Should trigger retry

    @pytest.mark.asyncio
    async def test_execute_rejection_max_retries(self, reviewer_reject, mock_llm_reject):
        """Test rejection when max retries exceeded."""
        state = AgentState(user_goal="Fix the bug")
        state.proposed_fix = "def fix(): pass"
        state.retry_count = 3
        state.max_retries = 3

        result = await reviewer_reject.execute(state)

        assert result.is_approved is False
        assert result.status == "rejected"

    @pytest.mark.asyncio
    async def test_execute_no_fix(self, reviewer_approve):
        """Test when no fix is provided."""
        state = AgentState(user_goal="Fix the bug")
        state.proposed_fix = None

        result = await reviewer_approve.execute(state)

        assert result.is_approved is False
        assert "No fix" in result.review_feedback
        assert result.status == "rejected"
