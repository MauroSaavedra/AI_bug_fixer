from src.fix_agent_orchestration.domain.interfaces import ILLMClient
from src.fix_agent_orchestration.domain.personas import build_retry_context
from src.fix_agent_orchestration.domain.state import AgentState
from src.fix_agent_orchestration.infrastructure.base_agent import BaseAgent
from loguru import logger

class PlannerAgent(BaseAgent):
    """Planner agent for generating RAG search queries.

    Responsibilities:
    - Analyze user goal for intent and requirements
    - Generate semantic search queries
    - Extract relevant keywords
    - Identify specific files mentioned
    """

    @property
    def name(self) -> str:
        """Agent name."""
        return "Planner"

    async def _execute_core(self, state: AgentState) -> AgentState:
        """Execute planning logic.

        Analyzes user goal and generates search strategy.

        Args:
            state: Current state with user_goal

        Returns:
            Updated state with search_query and keywords
        """
        logger.info(f"{self.name}: Analyzing user goal...")

        # Build prompts
        system_prompt = self._build_system_prompt()
        task_prompt = self._build_task_prompt(user_goal=state.user_goal)

        # Call LLM
        response = await self._call_llm(system_prompt, task_prompt)

        # Parse response
        try:
            result = self._parse_json_response(response.content)
        except ValueError as e:
            logger.warning(f"{self.name}: Failed to parse LLM response, using fallback")
            # Fallback: use user goal as search query
            state.search_query = state.user_goal
            state.extracted_keywords = []
            state.target_file_hint = None
            return state

        # Update state
        state.search_query = self._safe_get(result, "search_query", state.user_goal)
        state.extracted_keywords = self._safe_get(result, "keywords", [])
        state.target_file_hint = self._safe_get(result, "target_file")

        logger.info(f"{self.name}: Generated query: {state.search_query[:60]}...")
        logger.info(f"Keywords: {', '.join(state.extracted_keywords[:3])}")

        # Transition state
        state.status = "retrieving"
        state.log_transition("planning", "retrieving", f"Query: {state.search_query[:50]}")

        return state
