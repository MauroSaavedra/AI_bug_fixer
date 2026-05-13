"""Unit tests for coder_agent.py.

Tests the CoderAgent's fix generation, retry context, and prompt formatting.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from src.fix_agent_orchestration.infrastructure.coder_agent import CoderAgent
from src.fix_agent_orchestration.domain.interfaces import ILLMClient, LLMResponse
from src.fix_agent_orchestration.domain.state import AgentState


class MockLLMClient(ILLMClient):
    """Mock LLM client for testing."""

    _DEFAULT_RESPONSE = json.dumps({
        "proposed_fix": "def safe_div(a, b): return a / b if b != 0 else 0",
        "confidence_score": 0.95,
        "reasoning": "Added zero check",
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


class TestCoderAgentInit:
    """Test CoderAgent initialization."""

    def test_name(self):
        """Name is 'Coder'."""
        client = MockLLMClient()
        agent = CoderAgent(client)
        assert agent.name == "Coder"

    def test_description(self):
        """Description mentions Coder."""
        client = MockLLMClient()
        agent = CoderAgent(client)
        assert "Coder" in agent.description


class TestCoderAgentExecute:
    """Test CoderAgent execution."""

    @pytest.mark.asyncio
    async def test_execute_generates_fix(self):
        """Coder generates a fix with confidence and reasoning."""
        client = MockLLMClient()
        agent = CoderAgent(client)
        state = AgentState(user_goal="Fix division by zero")

        result = await agent._execute_core(state)

        assert result.status == "reviewing"
        assert result.proposed_fix == "def safe_div(a, b): return a / b if b != 0 else 0"
        assert result.confidence_score == 0.95
        assert result.reasoning == "Added zero check"

    @pytest.mark.asyncio
    async def test_execute_fallback_on_parse_error(self):
        """Use raw content when JSON parse fails."""
        client = MockLLMClient(response_content="raw fix code here")
        agent = CoderAgent(client)
        state = AgentState(user_goal="fix the bug described")

        result = await agent._execute_core(state)

        assert result.proposed_fix == "raw fix code here"
        assert result.confidence_score == 0.5
        assert result.status == "reviewing"

    @pytest.mark.asyncio
    async def test_execute_retry_mode(self):
        """On retry, incorporates reviewer feedback."""
        client = MockLLMClient()
        agent = CoderAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.retry_count = 1
        state.review_feedback = "Missing edge case for negative numbers"
        state.review_issues = ["Handle negatives"]

        result = await agent._execute_core(state)
        assert result.status == "reviewing"

    @pytest.mark.asyncio
    async def test_execute_empty_response(self):
        """Empty response uses defaults."""
        client = MockLLMClient(response_content="{}")
        agent = CoderAgent(client)
        state = AgentState(user_goal="fix the bug described")

        result = await agent._execute_core(state)
        assert result.proposed_fix is None  # default from empty dict
        assert result.confidence_score == 0.5
        assert result.status == "reviewing"


class TestFormatRetrievedCode:
    """Test _format_retrieved_code helper."""

    def test_no_context(self):
        """Handle empty context."""
        client = MockLLMClient()
        agent = CoderAgent(client)
        state = AgentState(user_goal="fix the bug described")
        assert state.retrieved_context == []

        result = agent._format_retrieved_code(state)
        assert result == "No code context retrieved."

    def test_with_context(self):
        """Format retrieved code for prompt."""
        from src.ingestion.domain.entities import CodeEntity, EntityType

        client = MockLLMClient()
        agent = CoderAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.retrieved_context = [
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="divide",
                content="def divide(a, b): return a / b",
                file_path=Path("/test.py"),
                start_line=1,
                end_line=2,
            )
        ]

        result = agent._format_retrieved_code(state)
        assert "divide" in result
        assert "Code Entity 1" in result

    def test_multiple_entities(self):
        """Format multiple entities."""
        from src.ingestion.domain.entities import CodeEntity, EntityType

        client = MockLLMClient()
        agent = CoderAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.retrieved_context = [
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="add",
                content="def add(a, b): return a + b",
                file_path=Path("/test.py"),
                start_line=1,
                end_line=2,
            ),
            CodeEntity(
                entity_type=EntityType.FUNCTION,
                name="sub",
                content="def sub(a, b): return a - b",
                file_path=Path("/test.py"),
                start_line=4,
                end_line=5,
            ),
        ]

        result = agent._format_retrieved_code(state)
        assert "Code Entity 1" in result
        assert "Code Entity 2" in result
        assert "add" in result
        assert "sub" in result


class TestBuildRetryContext:
    """Test _build_retry_context helper."""

    def test_no_retry(self):
        """No retry context on first attempt."""
        client = MockLLMClient()
        agent = CoderAgent(client)
        state = AgentState(user_goal="fix the bug described")
        assert state.retry_count == 0

        result = agent._build_retry_context(state)
        assert result == ""

    def test_with_retry_feedback(self):
        """Format retry context with feedback."""
        client = MockLLMClient()
        agent = CoderAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.retry_count = 1
        state.review_feedback = "Add type hints"
        state.review_issues = ["Missing types", "No docstring"]

        result = agent._build_retry_context(state)
        assert "Add type hints" in result
        assert "Add type hints" in result
        assert "Specific issues" in result
