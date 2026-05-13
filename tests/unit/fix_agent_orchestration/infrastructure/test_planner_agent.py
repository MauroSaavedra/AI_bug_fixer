"""Unit tests for planner_agent.py.

Tests the PlannerAgent's query generation and keyword extraction logic.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.fix_agent_orchestration.infrastructure.planner_agent import PlannerAgent
from src.fix_agent_orchestration.domain.interfaces import ILLMClient, LLMResponse
from src.fix_agent_orchestration.domain.state import AgentState


class MockLLMClient(ILLMClient):
    """Mock LLM client for testing."""

    _DEFAULT_RESPONSE = json.dumps({
        "search_query": "fix division by zero",
        "keywords": ["division", "zero", "check"],
    })

    def __init__(self, response_content=None):
        # Use sentinel to distinguish None from "" empty string
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
            tokens_used=10,
            latency_ms=50.0,
            finish_reason="stop",
        )

    async def chat_stream(self, messages, temperature=None, max_tokens=None):
        yield "streamed"
        return

    def is_available(self):
        return True


class TestPlannerAgentInit:
    """Test PlannerAgent initialization."""

    def test_name(self):
        """Name is 'Planner'."""
        client = MockLLMClient()
        agent = PlannerAgent(client)
        assert agent.name == "Planner"

    def test_description(self):
        """Description mentions planner role."""
        client = MockLLMClient()
        agent = PlannerAgent(client)
        assert "Planner" in agent.description


class TestPlannerAgentExecute:
    """Test PlannerAgent execution."""

    @pytest.mark.asyncio
    async def test_execute_generates_query(self):
        """Planner generates search query and keywords."""
        client = MockLLMClient()
        agent = PlannerAgent(client)
        state = AgentState(user_goal="Fix division by zero in calculator")
        state.status = "planning"

        result = await agent._execute_core(state)

        assert result.status == "retrieving"
        assert result.search_query == "fix division by zero"
        assert result.extracted_keywords == ["division", "zero", "check"]

    @pytest.mark.asyncio
    async def test_execute_fallback_on_parse_error(self):
        """Use user goal as fallback when parsing fails."""
        client = MockLLMClient(response_content="not json")
        agent = PlannerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.status = "planning"

        result = await agent._execute_core(state)

        assert result.search_query == "fix the bug described"
        assert result.extracted_keywords == []
        # On fallback, status stays as "planning" because return happens before transition

    @pytest.mark.asyncio
    async def test_execute_with_target_file(self):
        """Extract target file hint if present."""
        response = json.dumps({
            "search_query": "fix division",
            "keywords": ["division"],
            "target_file": "calculator.py",
        })
        client = MockLLMClient(response_content=response)
        agent = PlannerAgent(client)
        state = AgentState(user_goal="division calculator bug")
        state.status = "planning"

        result = await agent._execute_core(state)
        assert result.target_file_hint == "calculator.py"

    @pytest.mark.asyncio
    async def test_execute_empty_response(self):
        """Handle empty response."""
        client = MockLLMClient(response_content="")
        agent = PlannerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.status = "planning"

        result = await agent._execute_core(state)
        assert result.search_query == "fix the bug described"  # fallback

    @pytest.mark.asyncio
    async def test_execute_missing_keys(self):
        """Handle response with missing keys."""
        response = json.dumps({"search_query": "fix bug"})
        client = MockLLMClient(response_content=response)
        agent = PlannerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.status = "planning"

        result = await agent._execute_core(state)
        assert result.search_query == "fix bug"
        assert result.extracted_keywords == []

    @pytest.mark.asyncio
    async def test_execute_empty_json(self):
        """Empty JSON uses defaults."""
        client = MockLLMClient(response_content="{}")
        agent = PlannerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.status = "planning"

        result = await agent._execute_core(state)
        assert result.search_query == "fix the bug described"
        assert result.extracted_keywords == []
        assert result.status == "retrieving"

    @pytest.mark.asyncio
    async def test_execute_with_target_file(self):
        """Extract target file hint if present."""
        response = json.dumps({
            "search_query": "fix division",
            "keywords": ["division"],
            "target_file": "calculator.py",
        })
        client = MockLLMClient(response_content=response)
        agent = PlannerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.status = "planning"

        result = await agent._execute_core(state)
        assert result.target_file_hint == "calculator.py"

    @pytest.mark.asyncio
    async def test_execute_empty_response(self):
        """Handle empty response from LLM."""
        # Empty string should fail JSON parse and fallback
        client = MockLLMClient(response_content="")
        agent = PlannerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.status = "planning"

        result = await agent._execute_core(state)
        # On empty response (parse failure), fallback uses user_goal
        assert result.search_query == "fix the bug described"

    @pytest.mark.asyncio
    async def test_execute_missing_keys(self):
        """Handle response with missing keys."""
        response = json.dumps({"search_query": "fix bug"})
        client = MockLLMClient(response_content=response)
        agent = PlannerAgent(client)
        state = AgentState(user_goal="fix the bug described")
        state.status = "planning"

        result = await agent._execute_core(state)
        assert result.search_query == "fix bug"
        assert result.extracted_keywords == []

    @pytest.mark.asyncio
    async def test_execute_empty_json(self):
        """Empty JSON uses defaults."""
        client = MockLLMClient(response_content="{}")
        agent = PlannerAgent(client)
        state = AgentState(user_goal="my goal is described")
        state.status = "planning"

        result = await agent._execute_core(state)
        # Empty JSON {} means missing keys, fallback to user_goal
        assert result.search_query == "my goal is described"
        assert result.extracted_keywords == []
