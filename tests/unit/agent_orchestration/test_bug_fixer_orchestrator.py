"""Unit tests for BugFixerOrchestrator.

Tests the main orchestration logic including state machine and retry loops.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.fix_agent_orchestration.application.bug_fixer_orchestrator import (
    BugFixerOrchestrator,
    FixResult,
)
from src.fix_agent_orchestration.domain.interfaces import IAgent, ILLMClient
from src.fix_agent_orchestration.domain.state import AgentState
from src.ingestion.domain.entities import CodeEntity, EntityType
from src.ingestion.infrastructure.chroma_store import ChromaStore


class TestBugFixerOrchestrator:
    """Test suite for BugFixerOrchestrator."""

    @pytest.fixture
    def mock_planner(self):
        """Create mock planner agent."""
        planner = MagicMock(spec=IAgent)
        planner.name = "Planner"
        planner.description = "Generates search queries"
        return planner

    @pytest.fixture
    def mock_coder(self):
        """Create mock coder agent."""
        coder = MagicMock(spec=IAgent)
        coder.name = "Coder"
        coder.description = "Generates fixes"
        return coder

    @pytest.fixture
    def mock_reviewer(self):
        """Create mock reviewer agent."""
        reviewer = MagicMock(spec=IAgent)
        reviewer.name = "Reviewer"
        reviewer.description = "Validates fixes"
        return reviewer

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = MagicMock(spec=ChromaStore)
        store.similarity_search = AsyncMock(return_value=[])
        store.get_collection_stats = AsyncMock(return_value={
            "total_entities": 10,
            "collection_name": "test",
            "entity_type_counts": {},
        })
        return store

    @pytest.fixture
    def orchestrator(self, mock_planner, mock_coder, mock_reviewer, mock_vector_store):
        """Create orchestrator with mock agents."""
        return BugFixerOrchestrator(
            planner=mock_planner,
            coder=mock_coder,
            reviewer=mock_reviewer,
            vector_store=mock_vector_store,
        )

    @pytest.fixture
    def sample_entity(self):
        """Create sample entity for search results."""
        return CodeEntity(
            entity_type=EntityType.FUNCTION,
            name="divide",
            content="def divide(a, b): return a / b",
            file_path=Path("/test.py"),
            start_line=1,
            end_line=2,
        )

    @pytest.mark.asyncio
    async def test_fix_bug_success(self, orchestrator, mock_planner, mock_coder, mock_reviewer, sample_entity):
        """Test successful bug fixing workflow."""
        # Setup state transitions
        async def planner_execute(state):
            state.status = "retrieving"
            state.search_query = "division by zero"
            return state

        async def coder_execute(state):
            state.status = "reviewing"
            state.proposed_fix = "def divide(a, b): return a / b if b != 0 else 0"
            state.confidence_score = 0.95
            state.reasoning = "Added zero check"
            return state

        async def reviewer_execute(state):
            state.status = "approved"
            state.is_approved = True
            state.review_feedback = "Fix looks good"
            return state

        mock_planner.execute = planner_execute
        mock_coder.execute = coder_execute
        mock_reviewer.execute = reviewer_execute

        # Execute
        result = await orchestrator.fix_bug("Fix division by zero", max_retries=3)

        # Verify
        assert result.success is True
        assert result.is_approved is True
        assert result.fix == "def divide(a, b): return a / b if b != 0 else 0"
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_fix_bug_with_retry(self, orchestrator, mock_planner, mock_coder, mock_reviewer):
        """Test bug fixing with retry loop."""
        call_count = 0

        async def reviewer_execute(state):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First review: reject
                state.is_approved = False
                state.review_feedback = "Missing type hints"
                state.review_issues = ["Add type annotations"]
                state.status = "coding"  # Trigger retry
            else:
                # Second review: approve
                state.is_approved = True
                state.status = "approved"

            return state

        async def planner_execute(state):
            state.status = "retrieving"
            return state

        async def coder_execute(state):
            state.status = "reviewing"
            state.proposed_fix = "def divide(a: int, b: int): ..."
            state.confidence_score = 0.9
            return state

        mock_planner.execute = planner_execute
        mock_coder.execute = coder_execute
        mock_reviewer.execute = reviewer_execute

        # Execute
        result = await orchestrator.fix_bug("Fix the bug", max_retries=3)

        # Verify retry happened
        assert result.success is True
        assert result.retry_count == 1
        assert call_count == 2  # Two review calls

    @pytest.mark.asyncio
    async def test_fix_bug_max_retries_exceeded(self, orchestrator, mock_planner, mock_coder, mock_reviewer):
        """Test when max retries are exceeded."""
        async def reviewer_execute(state):
            state.is_approved = False
            state.review_feedback = "Still not correct"
            state.status = "coding"  # Would trigger retry, but max_retries exceeded
            return state

        async def planner_execute(state):
            state.status = "retrieving"
            return state

        async def coder_execute(state):
            state.status = "reviewing"
            return state

        mock_planner.execute = planner_execute
        mock_coder.execute = coder_execute
        mock_reviewer.execute = reviewer_execute

        # Execute with only 1 retry allowed
        result = await orchestrator.fix_bug("Fix the bug", max_retries=1)

        # Should end in rejected state
        assert result.success is False

    @pytest.mark.asyncio
    async def test_state_transitions(self, orchestrator):
        """Test that all state transitions are valid."""
        transitions = [
            ("planning", orchestrator._handle_planning),
            ("retrieving", orchestrator._handle_retrieval),
            ("coding", orchestrator._handle_coding),
            ("reviewing", orchestrator._handle_reviewing),
        ]

        for status, handler in transitions:
            assert callable(handler)

    @pytest.mark.asyncio
    async def test_handle_retrieval(self, orchestrator, mock_vector_store, sample_entity):
        """Test retrieval phase."""
        mock_vector_store.similarity_search.return_value = [sample_entity]

        state = AgentState(user_goal="Fix bug for this test")
        state.search_query = "find division"
        state.status = "retrieving"

        result = await orchestrator._handle_retrieval(state)

        assert result.status == "coding"
        assert len(result.retrieved_context) == 1
        assert result.retrieved_context[0].name == "divide"

    @pytest.mark.asyncio
    async def test_callback_invoked(self, mock_vector_store):
        callback_calls = []

        def on_state_change(state):
            callback_calls.append(state.status)

        orchestrator = BugFixerOrchestrator(
            planner=MagicMock(),
            coder=MagicMock(),
            reviewer=MagicMock(),
            vector_store=mock_vector_store,
            on_state_change=on_state_change,
        )

        async def planner_execute(state):
            state.status = "retrieving"
            return state

        async def coder_execute(state):
            state.status = "reviewing"
            return state

        async def reviewer_execute(state):
            state.status = "approved"
            return state

        orchestrator._planner.execute = planner_execute
        orchestrator._coder.execute = coder_execute
        orchestrator._reviewer.execute = reviewer_execute

        await orchestrator.fix_bug("Fix bug for this test", max_retries=1)

        assert len(callback_calls) > 0


class TestFixResult:
    """Test suite for FixResult."""

    def test_successful_result(self):
        """Test successful fix result."""
        state = AgentState(user_goal="Fix bug for this test")
        state.is_approved = True

        result = FixResult(
            success=True,
            state=state,
            fix="def fixed(): pass",
            feedback="Approved",
            issues=[],
            retry_count=0,
        )

        assert result.is_approved is True
        assert "APPROVED" in str(result)

    def test_rejected_result(self):
        """Test rejected fix result."""
        state = AgentState(user_goal="Fix bug for this test")
        state.is_approved = False

        result = FixResult(
            success=False,
            state=state,
            fix="def broken(): pass",
            feedback="Rejected",
            issues=["Issue 1"],
            retry_count=3,
        )

        assert result.is_approved is False
        assert "REJECTED" in str(result)
        assert "retries: 3" in str(result)

    def test_to_dict(self):
        """Test dictionary conversion."""
        state = AgentState(user_goal="Fix bug for this test")
        state.retrieved_context = [MagicMock(CodeEntity)]

        result = FixResult(
            success=True,
            state=state,
            fix="def fixed(): pass",
            feedback="Approved",
            issues=[],
            retry_count=0,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["fix"] == "def fixed(): pass"
        assert data["feedback"] == "Approved"
        assert data["retry_count"] == 0
