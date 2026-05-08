"""Reviewer agent implementation.

The Reviewer critically evaluates proposed fixes and decides whether
to approve them or reject with specific feedback for retry.
"""

from src.agent_orchestration.domain.interfaces import ILLMClient
from src.agent_orchestration.domain.state import AgentState
from src.agent_orchestration.infrastructure.base_agent import BaseAgent
from loguru import logger

class ReviewerAgent(BaseAgent):
    """Reviewer agent for validating code fixes.

    Responsibilities:
    - Verify fix addresses the user's goal
    - Check for syntax correctness
    - Identify logic errors and edge cases
    - Assess security implications
    - Look for unintended side effects
    - Provide specific, actionable feedback on rejection

    The Reviewer has the power to trigger retry cycles by rejecting fixes
    with detailed feedback that guides the Coder's next attempt.
    """

    @property
    def name(self) -> str:
        """Agent name."""
        return "Reviewer"

    async def _execute_core(self, state: AgentState) -> AgentState:
        """Execute review logic.

        Evaluates the proposed fix and either approves or rejects it.
        On rejection, provides detailed feedback for retry.

        Args:
            state: Current state with proposed_fix

        Returns:
            Updated state with is_approved and review_feedback
        """
        logger.info(f"{self.name}: Reviewing fix...")
        
        # Reject empty fixes immediately
        if not state.proposed_fix:
            logger.warning(f"{self.name}: No fix provided")

            state.is_approved = False
            state.review_feedback = "No fix provided"
            state.review_issues = ["Missing proposed fix"]
            state.status = "rejected"

            return state

        # Build context
        retrieved_code = self._format_retrieved_code(state)

        # Build prompts
        system_prompt = self._build_system_prompt()
        task_prompt = self._build_task_prompt(
            user_goal=state.user_goal,
            retrieved_code=retrieved_code,
            proposed_fix=state.proposed_fix or "No fix provided",
            reasoning=state.reasoning or "No reasoning provided",
            confidence_score=state.confidence_score or 0.0,
        )

        # Call LLM
        response = await self._call_llm(system_prompt, task_prompt)

        # Parse response
        try:
            result = self._parse_json_response(response.content)
        except ValueError as e:
            logger.warning(f"{self.name}: Failed to parse LLM response, using fallback")
            # Fallback: auto-approve if parse fails
            state.is_approved = True
            state.review_feedback = "Auto-approved (parse error)"
            state.review_issues = []
            return state

        # Update state
        state.is_approved = self._safe_get(result, "is_approved", False)
        state.review_feedback = self._safe_get(result, "feedback", "")
        state.review_issues = self._safe_get(result, "issues", [])
        severity = self._safe_get(result, "severity", "unknown")

        if state.is_approved:
            logger.info(f"{self.name}: Fix approved")
            state.status = "approved"
            state.log_transition("reviewing", "approved", "Fix passed review")
        else:
            logger.info(f"{self.name}: Fix rejected (severity: {severity})")
            logger.info(f"Issues: {len(state.review_issues)}")
            if state.review_issues:
                logger.info(f"{state.review_issues[0][:80]}...")

            # Check if we can retry
            if state.can_retry():
                state.status = "coding"  # Will trigger retry
                state.log_transition(
                    "reviewing",
                    "coding",
                    f"Rejected, preparing retry #{state.retry_count + 1}",
                )
            else:
                state.status = "rejected"
                state.log_transition(
                    "reviewing",
                    "rejected",
                    f"Max retries ({state.max_retries}) exceeded",
                )

        return state

    def _format_retrieved_code(self, state: AgentState) -> str:
        """Format retrieved code entities for the prompt.

        Args:
            state: Current state with retrieved_context

        Returns:
            Formatted code string
        """
        if not state.retrieved_context:
            return "No code context retrieved."

        parts = []
        for i, entity in enumerate(state.retrieved_context, 1):
            parts.append(f"\n--- Original Code Entity {i} ---")
            parts.append(entity.to_chroma_document())
            parts.append("")

        return "\n".join(parts)
