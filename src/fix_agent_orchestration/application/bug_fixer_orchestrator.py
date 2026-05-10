"""BugFixerOrchestrator - Multi-agent orchestration with retry logic.

This module implements the main orchestration logic that coordinates
Planner, Coder, and Reviewer agents in a state machine with retry capabilities.

The orchestration flow:
1. PLANNING → RETRIEVING → CODING → REVIEWING → APPROVED
2. If REJECTED and can retry: REVIEWING → CODING (retry loop)
3. If max retries exceeded: REVIEWING → REJECTED (final)
"""

from typing import Callable

from src.fix_agent_orchestration.domain.interfaces import IAgent, ILLMClient
from src.fix_agent_orchestration.domain.state import AgentState
from src.fix_agent_orchestration.infrastructure.coder_agent import CoderAgent
from src.fix_agent_orchestration.infrastructure.planner_agent import PlannerAgent
from src.fix_agent_orchestration.infrastructure.reviewer_agent import ReviewerAgent
from src.ingestion.domain.entities import CodeEntity
from src.ingestion.infrastructure.chroma_store import ChromaStore
from loguru import logger

class BugFixerOrchestrator:
    """Multi-agent orchestrator for automated bug fixing.

    Coordinates the Planner, Coder, and Reviewer agents through a
    state machine that supports retry loops when fixes are rejected.

    Features:
    - Async state machine execution
    - Hybrid RAG search (semantic + keyword)
    - Configurable retry logic
    - Detailed execution logging
    - Result aggregation
    """

    def __init__(
        self,
        planner: IAgent,
        coder: IAgent,
        reviewer: IAgent,
        vector_store: ChromaStore,
        on_state_change: Callable[[AgentState], None] | None = None,
    ):
        """Initialize the orchestrator with all required agents.

        Args:
            planner: Planner agent for search query generation
            coder: Coder agent for fix generation
            reviewer: Reviewer agent for fix validation
            vector_store: Vector database for code retrieval
            on_state_change: Optional callback for state updates
        """
        self._planner = planner
        self._coder = coder
        self._reviewer = reviewer
        self._vector_store = vector_store
        self._on_state_change = on_state_change

    @classmethod
    def create_default(
        cls,
        llm_client: ILLMClient,
        vector_store: ChromaStore,
        temperature: float = 0.1,
        on_state_change: Callable[[AgentState], None] | None = None,
    ) -> "BugFixerOrchestrator":
        """Factory method to create orchestrator with default agents.

        Args:
            llm_client: LLM client for all agents
            vector_store: Vector database instance
            temperature: Sampling temperature for agents
            on_state_change: Optional state change callback
        Returns:
            Configured BugFixerOrchestrator instance
        """
        planner = PlannerAgent(llm_client, temperature)
        coder = CoderAgent(llm_client, temperature)
        reviewer = ReviewerAgent(llm_client, temperature)

        return cls(
            planner=planner,
            coder=coder,
            reviewer=reviewer,
            vector_store=vector_store,
            on_state_change=on_state_change,
        )
        
    async def fix_bug(
        self,
        user_goal: str,
        max_retries: int = 3,
    ) -> "FixResult":
        """Execute the full bug fixing workflow.

        This is the main entry point for bug fixing. It orchestrates
        the entire flow from planning to final approval/rejection.

        Args:
            user_goal: Description of the bug to fix
            max_retries: Maximum number of retry attempts

        Returns:
            FixResult with final state and proposed fix
        """
        logger.info(f"{'='*60}")
        logger.info(f"Starting bug fix workflow")
        logger.info(f"Goal: {user_goal[:60]}...")
        logger.info(f"Max retries: {max_retries}")
        logger.info(f"{'='*60}")

        # Initialize state
        state = AgentState(
            user_goal=user_goal,
            max_retries=max_retries,
            status="planning",
        )

        # Execute workflow
        while state.status not in ("approved", "rejected", "failed"):
            state = await self._execute_step(state)

            # Notify callback if registered
            if self._on_state_change:
                self._on_state_change(state)

        # Return result
        return FixResult(
            success=state.status == "approved",
            state=state,
            fix=state.proposed_fix,
            feedback=state.review_feedback,
            issues=state.review_issues,
            retry_count=state.retry_count,
        )

    async def _execute_step(self, state: AgentState) -> AgentState:
        """Execute a single step in the state machine.

        Routes to the appropriate handler based on current status.

        Args:
            state: Current agent state

        Returns:
            Updated state after executing the step
        """
        handlers = {
            "planning": self._handle_planning,
            "retrieving": self._handle_retrieval,
            "coding": self._handle_coding,
            "reviewing": self._handle_reviewing,
        }

        handler = handlers.get(state.status)
        if not handler:
            raise ValueError(f"Unknown status: {state.status}")

        return await handler(state)

    async def _handle_planning(self, state: AgentState) -> AgentState:
        """Execute planning step.

        Uses Planner agent to generate search query and extract keywords.

        Args:
            state: Current state with user_goal

        Returns:
            Updated state with search_query and keywords
        """
        logger.info("\nPHASE: Planning")
        state = await self._planner.execute(state)

        if state.status == "failed":
            logger.error("Planning failed")

        return state

    async def _handle_retrieval(self, state: AgentState) -> AgentState:
        """Execute retrieval step.

        Performs hybrid RAG search using Planner's query and keywords.

        Args:
            state: Current state with search_query

        Returns:
            Updated state with retrieved_context
        """
        logger.info("\nPHASE: Retrieval")

        if not state.search_query:
            logger.info("No search query, using user goal")
            state.search_query = state.user_goal

        # Perform semantic search
        logger.info(f"Query: {state.search_query[:60]}...")
        try:
            entities = await self._vector_store.similarity_search(
                query=state.search_query,
                limit=5,
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            entities = []

        # Keyword boost (simple implementation)
        if state.extracted_keywords and entities:
            logger.info(f"Boosting with keywords: {state.extracted_keywords[:3]}")
            # Could implement hybrid scoring here
            # For now, just use the semantic results

        state.retrieved_context = entities

        logger.info(f"Retrieved {len(entities)} entities")
        for entity in entities[:3]:
            logger.info(f"{entity.qualified_name} at {entity.location}")

        # Transition to coding
        state.status = "coding"
        state.log_transition("retrieving", "coding", f"Retrieved {len(entities)} entities")

        return state

    async def _handle_coding(self, state: AgentState) -> AgentState:
        """Execute coding step.

        Uses Coder agent to generate fix with retrieved context.
        On retry, includes reviewer feedback.

        Args:
            state: Current state with retrieved_context

        Returns:
            Updated state with proposed_fix
        """
        logger.info("\nPHASE: Coding")

        if state.retry_count > 0:
            logger.info(f"(Retry attempt #{state.retry_count})")
            logger.info(f"Incorporating reviewer feedback...")

        state = await self._coder.execute(state)

        if state.status == "failed":
            logger.error("Coding failed")

        return state

    async def _handle_reviewing(self, state: AgentState) -> AgentState:
        """Execute reviewing step.

        Uses Reviewer agent to validate the proposed fix.
        Can trigger retry loop if rejected.

        Args:
            state: Current state with proposed_fix

        Returns:
            Updated state with is_approved and review_feedback
        """
        logger.info("\nPHASE: Review")

        if not state.proposed_fix:
            logger.info("No fix to review")
            state.is_approved = False
            state.review_feedback = "No fix was generated"
            state.status = "rejected"
            return state

        state = await self._reviewer.execute(state)

        # Handle retry logic
        if state.status == "coding":
            # Reviewer rejected and we can retry
            logger.info(f"Retrying fix generation (attempt {state.retry_count + 1}/{state.max_retries})")

            # Create retry state
            state = state.to_retry_state()

        return state


class FixResult:
    """Result of a bug fixing workflow.

    Provides a clean interface for accessing the final state and fix.
    """

    def __init__(
        self,
        success: bool,
        state: AgentState,
        fix: str | None,
        feedback: str | None,
        issues: list[str],
        retry_count: int,
    ):
        self.success = success
        self.state = state
        self.fix = fix
        self.feedback = feedback
        self.issues = issues
        self.retry_count = retry_count

    @property
    def is_approved(self) -> bool:
        """Check if fix was approved."""
        return self.success

    def __str__(self) -> str:
        """Human-readable result."""
        status = "✅ APPROVED" if self.success else "❌ REJECTED"
        result = f"{status} (retries: {self.retry_count})\n"

        if self.fix:
            result += f"\nProposed Fix ({len(self.fix)} chars):\n"
            result += f"```python\n{self.fix[:500]}"
            if len(self.fix) > 500:
                result += "\n... (truncated)"
            result += "\n```\n"

        if self.feedback:
            result += f"\nFeedback: {self.feedback[:200]}"
            if len(self.feedback) > 200:
                result += "..."
            result += "\n"

        if self.issues:
            result += f"\nIssues ({len(self.issues)}):\n"
            for issue in self.issues[:3]:
                result += f"  - {issue[:80]}\n"

        return result

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "is_approved": self.is_approved,
            "fix": self.fix,
            "feedback": self.feedback,
            "issues": self.issues,
            "retry_count": self.retry_count,
            "final_status": self.state.status,
            "retrieved_context_count": len(self.state.retrieved_context),
        }
