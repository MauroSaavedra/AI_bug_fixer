"""Unit tests for base_agent.py infrastructure.

Tests the BaseAgent's prompt building, JSON parsing, and error handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.fix_agent_orchestration.infrastructure.base_agent import (
    AgentExecutionError,
    BaseAgent,
)
from src.fix_agent_orchestration.domain.state import AgentState


class MockAgent(BaseAgent):
    """Concrete implementation for testing."""

    @property
    def name(self):
        return "Planner"

    async def _execute_core(self, state: AgentState) -> AgentState:
        state.status = "approved"
        return state


class TestBaseAgentInit:
    """Test BaseAgent initialization."""

    def test_init(self):
        """Initialize with LLM client."""
        mock_client = MagicMock()
        agent = MockAgent(mock_client, temperature=0.5)
        assert agent._temperature == 0.5

    def test_init_default_temperature(self):
        """Default temperature is 0.1."""
        mock_client = MagicMock()
        agent = MockAgent(mock_client)
        assert agent._temperature == 0.1

    def test_property_name(self):
        """Subclasses must implement name."""
        mock_client = MagicMock()
        agent = MockAgent(mock_client)
        assert agent.name == "Planner"

    def test_description_from_persona(self):
        """Description is derived from persona (Planner)."""
        mock_client = MagicMock()
        agent = MockAgent(mock_client)
        # Planner's description mapping
        assert "Planner" in agent.description or len(agent.description) > 0


class TestBaseAgentExecute:
    """Test the execute framework method."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Successfully executes core logic."""
        mock_client = MagicMock()
        agent = MockAgent(mock_client)
        state = AgentState(user_goal="test execute the flow successfully")
        state.status = "planning"

        result = await agent.execute(state)
        assert result.status == "approved"

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        """Catch exceptions and set status to failed."""
        class BrokenAgent(BaseAgent):
            @property
            def name(self):
                return "Planner"

            async def _execute_core(self, state: AgentState) -> AgentState:
                raise ValueError("boom")

        mock_client = MagicMock()
        agent = BrokenAgent(mock_client)
        state = AgentState(user_goal="test execute error handling path")

        result = await agent.execute(state)
        assert result.status == "failed"


class TestCallLLM:
    """Test _call_llm helper."""

    @pytest.mark.asyncio
    async def test_call_llm(self):
        """Call LLM with structured messages."""
        from src.fix_agent_orchestration.domain.interfaces import LLMResponse

        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.provider_name = "test-provider"
        mock_client.chat = AsyncMock(return_value=LLMResponse(
            content="response",
            provider="test",
            model="model",
            latency_ms=100.0,
            finish_reason="stop",
        ))

        agent = MockAgent(mock_client)
        response = await agent._call_llm("system prompt", "task prompt")
        assert response.content == "response"


class TestParseJSONResponse:
    """Test JSON response parsing."""

    @pytest.fixture
    def agent(self):
        mock_client = MagicMock()
        return MockAgent(mock_client)

    def test_parse_json_in_code_block(self, agent):
        """Extract JSON from markdown code block."""
        content = "```json\n{\"key\": \"value\"}\n```"
        result = agent._parse_json_response(content)
        assert result == {"key": "value"}

    def test_parse_json_inline(self, agent):
        """Extract JSON inline."""
        content = '{"key": "value"}'
        result = agent._parse_json_response(content)
        assert result == {"key": "value"}

    def test_parse_json_surrounded_by_text(self, agent):
        """Extract JSON in middle of text."""
        content = 'Here is the result:\n```json\n{"key": "value"}\n```\nEnd.'
        result = agent._parse_json_response(content)
        assert result == {"key": "value"}

    def test_parse_json_invalid(self, agent):
        """Raise ValueError for invalid JSON."""
        with pytest.raises(ValueError):
            agent._parse_json_response("not json")

    def test_parse_json_nested_braces(self, agent):
        """Handle nested JSON objects."""
        content = '{"outer": {"inner": "value"}}'
        result = agent._parse_json_response(content)
        assert result == {"outer": {"inner": "value"}}


class TestSafeGet:
    """Test _safe_get method."""

    @pytest.fixture
    def agent(self):
        mock_client = MagicMock()
        return MockAgent(mock_client)

    def test_get_existing_key(self, agent):
        """Get existing key."""
        data = {"key": "value"}
        assert agent._safe_get(data, "key") == "value"

    def test_get_missing_with_default(self, agent):
        """Get missing key with default."""
        data = {"key": "value"}
        assert agent._safe_get(data, "missing", "default") == "default"

    def test_get_missing_no_default(self, agent):
        """Get missing key without default."""
        data = {"key": "value"}
        assert agent._safe_get(data, "missing") is None


class TestBuildPrompts:
    """Test prompt building methods."""

    @pytest.fixture
    def agent(self):
        mock_client = MagicMock()
        return MockAgent(mock_client)

    def test_build_system_prompt(self, agent):
        """System prompt from persona."""
        prompt = agent._build_system_prompt()
        assert len(prompt) > 0

    def test_build_task_prompt(self, agent):
        """Task prompt with substitution."""
        prompt = agent._build_task_prompt(user_goal="fix the bug in code")
        assert "fix the bug in code" in prompt


class TestAgentExecutionError:
    """Test custom exception."""

    def test_error_message(self):
        error = AgentExecutionError("TestAgent", "something went wrong")
        assert "TestAgent" in str(error)
        assert "something went wrong" in str(error)

    def test_error_with_state(self):
        state = AgentState(user_goal="test error handling state")
        error = AgentExecutionError("TestAgent", "error", state=state)
        assert error.state.user_goal == "test error handling state"
