"""Unit tests for BaseAgent.

Tests the abstract base class for agents.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent_orchestration.domain.interfaces import ILLMClient, LLMResponse
from src.agent_orchestration.domain.state import AgentState
from src.agent_orchestration.infrastructure.base_agent import (
    AgentExecutionError,
    BaseAgent,
)


class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""

    @property
    def name(self) -> str:
        return "Planner"

    async def _execute_core(self, state: AgentState) -> AgentState:
        state.status = "approved"
        return state


class TestBaseAgent:
    """Test suite for BaseAgent."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock(spec=ILLMClient)
        client.chat = AsyncMock()
        return client

    @pytest.fixture
    def agent(self, mock_llm_client):
        """Create a test agent."""
        return MockAgent(mock_llm_client, temperature=0.1)

    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful execution."""
        state = AgentState(user_goal="Test goal with a test")
        result = await agent.execute(state)

        assert result.status == "approved"

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self, agent):
        """Test that exceptions are handled gracefully."""
        # Make _execute_core raise an exception
        async def failing_execute(state):
            raise ValueError("Test error")

        agent._execute_core = failing_execute

        state = AgentState(user_goal="Test goal with a test")
        result = await agent.execute(state)

        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_call_llm(self, agent, mock_llm_client):
        """Test LLM calling."""
        mock_response = LLMResponse(
            content='{"test": "response"}',
            provider="openai",
            model="gpt-4o",
        )
        mock_llm_client.chat.return_value = mock_response

        response = await agent._call_llm(
            system_prompt="You are a test",
            task_prompt="Test task",
        )

        assert response.content == '{"test": "response"}'
        mock_llm_client.chat.assert_called_once()

        # Check the messages format
        call_args = mock_llm_client.chat.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a test"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test task"

    def test_parse_json_response_clean(self, agent):
        """Test parsing clean JSON response."""
        content = '{"key": "value", "number": 123}'
        result = agent._parse_json_response(content)

        assert result["key"] == "value"
        assert result["number"] == 123

    def test_parse_json_response_with_markdown(self, agent):
        """Test parsing JSON wrapped in markdown."""
        content = '```json\n{"key": "value"}\n```'
        result = agent._parse_json_response(content)

        assert result["key"] == "value"

    def test_parse_json_response_with_backticks(self, agent):
        """Test parsing JSON with backticks."""
        content = '```\n{"key": "value"}\n```'
        result = agent._parse_json_response(content)

        assert result["key"] == "value"

    def test_parse_json_response_invalid(self, agent):
        """Test handling invalid JSON."""
        content = "not valid json"

        with pytest.raises(ValueError):
            agent._parse_json_response(content)

    def test_safe_get_existing_key(self, agent):
        """Test safe_get with existing key."""
        data = {"key": "value", "number": 42}
        result = agent._safe_get(data, "key")

        assert result == "value"

    def test_safe_get_missing_key_with_default(self, agent):
        """Test safe_get with missing key and default."""
        data = {"key": "value"}
        result = agent._safe_get(data, "missing", "default_value")

        assert result == "default_value"

    def test_safe_get_missing_key_no_default(self, agent):
        """Test safe_get with missing key and no default."""
        data = {"key": "value"}
        result = agent._safe_get(data, "missing")

        assert result is None

    def test_build_system_prompt(self, agent):
        """Test building system prompt from persona."""
        prompt = agent._build_system_prompt()

        # Should be from TestAgent persona
        assert "TestAgent" in prompt or len(prompt) > 0

    def test_build_task_prompt(self, agent):
        """Test building task prompt from template."""
        prompt = agent._build_task_prompt(user_goal="Test")
        assert "Test" in prompt


class TestAgentExecutionError:
    """Test suite for AgentExecutionError."""

    def test_error_message(self):
        """Test error message format."""
        state = AgentState(user_goal="Test goal with a test")
        error = AgentExecutionError(
            agent_name="TestAgent",
            message="Something failed",
            state=state,
        )

        assert "TestAgent agent failed" in str(error)
        assert error.agent_name == "TestAgent"
        assert error.state == state
