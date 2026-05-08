from src.agent_orchestration.domain.personas import build_retry_context
from src.agent_orchestration.domain.state import AgentState
from src.agent_orchestration.infrastructure.base_agent import BaseAgent
from loguru import logger

class CoderAgent(BaseAgent):
    """Coder agent for generating code fixes.

    Responsibilities:
    - Analyze bug description and code context
    - Generate complete, working fixes
    - Explain reasoning and provide confidence score
    - Incorporate reviewer feedback on retry

    Supports retry logic - on subsequent calls, includes reviewer feedback
    to guide the next fix attempt.
    """

    @property
    def name(self) -> str:
        """Agent name."""
        return "Coder"

    async def _execute_core(self, state: AgentState) -> AgentState:
        """Execute coding logic.

        Generates a fix based on retrieved context.
        On retry, incorporates reviewer feedback.

        Args:
            state: Current state with retrieved_context

        Returns:
            Updated state with proposed_fix
        """
        if state.retry_count > 0:
            logger.info(f"{self.name}: Generating fix (retry #{state.retry_count})...")
        else:
            logger.info(f"{self.name}: Generating fix...")

        # Build context
        context_summary = state.get_context_summary()
        retrieved_code = self._format_retrieved_code(state)
        retry_context = self._build_retry_context(state)

        # Build prompts
        system_prompt = self._build_system_prompt()
        task_prompt = self._build_task_prompt(
            user_goal=state.user_goal,
            context_summary=context_summary,
            retrieved_code=retrieved_code,
            retry_context=retry_context,
        )

        # Call LLM
        response = await self._call_llm(system_prompt, task_prompt)

        # Parse response
        try:
            result = self._parse_json_response(response.content)
        except ValueError as e:
            logger.warning(f"{self.name}: Failed to parse LLM response, using raw content")
            state.proposed_fix = response.content
            state.confidence_score = 0.5
            state.reasoning = "Parse error - using raw content"
            state.status = "reviewing"
            return state

        # Update state
        state.proposed_fix = self._safe_get(result, "proposed_fix")
        state.confidence_score = self._safe_get(result, "confidence_score", 0.5)
        state.reasoning = self._safe_get(result, "reasoning", "")

        logger.info(f"{self.name}: Generated fix ({len(state.proposed_fix or '')} chars)")
        logger.info(f"Confidence: {state.confidence_score:.2f}")

        # Transition state
        state.status = "reviewing"
        state.log_transition(
            "coding",
            "reviewing",
            f"Fix generated, confidence: {state.confidence_score:.2f}",
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
            parts.append(f"\n--- Code Entity {i} ---")
            parts.append(entity.to_chroma_document())
            parts.append("")

        return "\n".join(parts)

    def _build_retry_context(self, state: AgentState) -> str:
        """Build retry context if this is a retry attempt.

        Args:
            state: Current state

        Returns:
            Retry context string or empty
        """
        if state.retry_count == 0:
            return ""

        return build_retry_context(state.review_feedback, state.review_issues)
