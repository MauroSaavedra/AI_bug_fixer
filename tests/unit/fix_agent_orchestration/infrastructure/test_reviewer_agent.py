"""Unit tests for reviewer_agent.py.

Tests the ReviewerAgent's fix validation, approval/rejection logic,
and retry state management.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.fix_agent_orchestration.infrastructure.reviewer_agent import ReviewerAgent
from src.fix_agent_orchestration.domain.interfaces import ILLMClient, LLMResponse
from src.fix_agent_orchestration.domain.state import AgentState


class MockLLMClient(ILLMClient):
    """Mock LLM client for testing."""

    _DEFAULT_RESPONSE = json.dumps({
        "is_approved": True,
        "feedback": "Looks good",
        "issues": [],
        "severity": "low",
    })

    def __init__(self, response_content=None):
        if response_content is None:
            self._response = self._DEFAULT_RESPONSE
        else:
            self._response = response_content

    @property
    def provider_name(self):
        return "mock"

    @property
    def model_name(self):
        return "mock-model"

    async def chat(self, messages, temperature=None, max_tokens=None, stream=False):
        return LLMResponse(
            content=self._response,
            provider="mock",
            model="mock-model",
            latency_ms=50.0,
            finish_reason="stop",
        )

    async def chat_stream(self, messages, temperature=None, max_tokens=None):
        yield "streamed"
        return

    def is_available(self):
        return True


class TestReviewerAgentInit:
    """Test ReviewerAgent initialization."""

    def test_name(self):
        """Name is 'Reviewer'."""
        client = MockLLMClient()
        agent = ReviewerAgent(client)
        assert agent.name == "Reviewer"

    def test_description(self):
        """Description mentions Reviewer."""
        client = MockLLMClient()
        agent = ReviewerAgent(client)
        assert "Reviewer" in agent.description


class TestReviewerAgentExecute:
    """Test ReviewerAgent execution."""

    @pytest.mark.asyncio
    async def test_execute_approves_fix(self):
        """Approve a valid fix."""
        client = MockLLMClient()
        agent = ReviewerAgent(client)
        state = AgentState(user_goal="Fix the bug")
        state.proposed_fix = "def safe_div(a, b): return a / b if b != 0 else 0"
        state.status = "reviewing"

        result = await agent._execute_core(state)
        assert result.status == "approved"
        assert result.is_approved is True

    @pytest.mark.asyncio
    async def test_execute_rejects_fix_with_retry(self):
        """Reject fix and trigger retry."""
        response = json.dumps({
            "is_approved": False,
            "feedback": "Missing type hints",
            "issues": ["Add type annotations"],
            "severity": "medium",
        })
        client = MockLLMClient(response_content=response)
        agent = ReviewerAgent(client)
        state = AgentState(user_goal="Fix the bug", max_retries=3)
        state.proposed_fix = "def add(a, b): return a + b"
        state.status = "reviewing"

        result = await agent._execute_core(state)
        assert result.status == "coding"  # triggers retry
        assert result.is_approved is False
        assert result.review_feedback == "Missing type hints"
        assert result.review_issues == ["Add type annotations"]

    @pytest.mark.asyncio
    async def test_execute_rejects_fix_no_retry_left(self):
        """Reject fix and fail when max retries exceeded."""
        response = json.dumps({
            "is_approved": False,
            "feedback": "Still wrong",
            "issues": ["Fix logic"],
            "severity": "high",
        })
        client = MockLLMClient(response_content=response)
        agent = ReviewerAgent(client)
        state = AgentState(user_goal="Fix the bug", max_retries=1)
        state.retry_count = 1  # Already at max
        state.proposed_fix = "def add(a, b): return a + b"
        state.status = "reviewing"

        result = await agent._execute_core(state)
        assert result.status == "rejected"

    @pytest.mark.asyncio
    async def test_execute_empty_fix(self):
        """Reject immediately if no fix provided."""
        client = MockLLMClient()
        agent = ReviewerAgent(client)
        state = AgentState(user_goal="Fix the bug described")
        state.proposed_fix = None
        state.status = "reviewing"

        result = await agent._execute_core(state)
        assert result.status == "rejected"
        assert result.is_approved is False
        assert "No fix provided" in (result.review_feedback or "")

    @pytest.mark.asyncio
    async def test_execute_fallback_on_parse_error(self):
        """Auto-approve when JSON parse fails."""
        client = MockLLMClient(response_content="not json")
        agent = ReviewerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.proposed_fix = "def add(a, b): return a + b"
        state.status = "reviewing"

        result = await agent._execute_core(state)
        assert result.is_approved is True
        assert "Auto-approved" in (result.review_feedback or "")

    @pytest.mark.asyncio
    async def test_execute_no_proposed_fix_empty_string(self):
        """Reject if proposed_fix is empty string."""
        client = MockLLMClient()
        agent = ReviewerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.proposed_fix = ""
        state.status = "reviewing"

        result = await agent._execute_core(state)
        assert result.status == "rejected"
        assert result.is_approved is False


class TestReviewerAgentFormatting:
    """Test retrieved code formatting."""

    def test_format_no_context(self):
        """Handle empty context."""
        client = MockLLMClient()
        agent = ReviewerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        result = agent._format_retrieved_code(state)
        assert "No code context retrieved" in result

    def test_format_with_context(self):
        """Format code with entity context."""
        from src.ingestion.domain.entities import CodeEntity, EntityType
        from pathlib import Path

        client = MockLLMClient()
        agent = ReviewerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.retrieved_context = [
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="original",
                content="def original(): pass",
                file_path=Path("/test.py"),
                start_line=1,
                end_line=2,
            )
        ]

        result = agent._format_retrieved_code(state)
        assert "Original Code Entity 1" in result
        assert "original" in result
