import json
import re
from abc import abstractmethod
from typing import Any, Optional
import time
from loguru import logger
from src.fix_agent_orchestration.domain.interfaces import IAgent, ILLMClient, LLMResponse
from src.fix_agent_orchestration.domain.personas import AgentPersona, AgentPersonas
from src.fix_agent_orchestration.domain.state import AgentState
from src.observability.langfuse_utils import (
    update_current_generation,
    update_current_span,
)
from src.observability.base_agent_observer import (  # type: ignore[attr-defined]
    trace_agent_execution,
)

try:
    from langfuse import observe
except ImportError:
    observe = None  # type: ignore[assignment]


def _make_observe_wrapper(agent_instance):
    """Create a Langfuse observe wrapper for an agent instance.
    
    This is a factory that returns a decorator binding the agent's name.
    """
    def decorator(func):
        if observe is None:
            return func
        return observe(name=f"{agent_instance.name}_agent", as_type="span")(func)
    return decorator


def _wrap_method(obj, method_name, wrapper_factory):
    """Wrap an existing method with a dynamic decorator.
    
    This mutates the object in-place.
    """
    original = getattr(obj, method_name)
    wrapper = wrapper_factory(original)
    setattr(obj, method_name, wrapper)


class BaseAgent(IAgent):
    """Abstract base class for all agents.

    Provides common functionality:
    - LLM client management
    - Prompt building from personas
    - Structured JSON output parsing
    - Error handling with fallbacks
    - Langfuse observability for agent execution and LLM calls

    Subclasses implement the specific agent logic in `_execute_core()`.
    """

    def __init__(self, llm_client: ILLMClient, temperature: float = 0.1):
        """Initialize the base agent.

        Args:
            llm_client: LLM client for making API calls
            temperature: Sampling temperature for this agent
        """
        self._llm_client = llm_client
        self._temperature = temperature
        self._persona = AgentPersonas.get_persona(self.name)

        # Dynamically wrap execute() with Langfuse observe for this instance
        if observe is not None:
            original_execute = self.execute
            # We need to look up the (already bound) method on the instance
            @observe(name=f"{self.name}_agent", as_type="span")
            async def _wrapped(state: AgentState) -> AgentState:
                return await original_execute(state)
            self.execute = _wrapped  # type: ignore[method-assign]

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name - must be implemented by subclasses."""
        pass

    @property
    def description(self) -> str:
        """Get agent description from persona."""
        return self._persona.system_prompt[:100] + "..."
                                      
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's logic.

        This method provides the framework:
        1. Build the prompt from template
        2. Call the LLM
        3. Parse the response
        4. Update state with results

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        try:
            return await self._execute_core(state)
        except Exception as e:
            # Log error and return state with error info
            logger.error(f"{self.name} agent failed: {e}")
            state.status = "failed"
            return state
        
    @abstractmethod
    async def _execute_core(self, state: AgentState) -> AgentState:
        """Core execution logic - must be implemented by subclasses.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        pass

    def _build_system_prompt(self) -> str:
        """Build the system prompt from persona."""
        return self._persona.system_prompt

    def _build_task_prompt(self, **kwargs: Any) -> str:
        """Build the task prompt from template with substitutions.

        Args:
            **kwargs: Template variables to substitute

        Returns:
            Formatted task prompt
        """
        return self._persona.task_prompt_template.format(**kwargs)

    async def _call_llm(
        self,
        system_prompt: str,
        task_prompt: str,
    ) -> LLMResponse:
        """Call the LLM with structured messages.

        Args:
            system_prompt: System role message
            task_prompt: User task message

        Returns:
            LLMResponse with parsed content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        response = await self._llm_client.chat(
            messages=messages,
            temperature=self._temperature,
        )

        # Update Langfuse with generation metadata (v4 API)
        update_current_generation(
            model=getattr(self._llm_client, "model_name", "unknown"),
            model_parameters={"temperature": self._temperature,
                              "max_tokens": None, "stream": False},
            metadata={
                "agent_name": self.name,
                "provider": getattr(self._llm_client, "provider_name", "unknown"),
                "latency_ms": response.latency_ms,
                "finish_reason": response.finish_reason,
            },
        )

        return response

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response with robust extraction.

        Handles cases where LLM wraps JSON in markdown or adds extra text.

        Args:
            content: Raw LLM response content

        Returns:
            Parsed JSON as dictionary

        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Try to extract JSON from markdown code blocks
        # Pattern 1: ```json {...} ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # Pattern 2: Just { ... } at the start
        if not json_match:
            json_match = re.search(r"^(\{.*\})$", content.strip(), re.DOTALL)
            if json_match:
                content = json_match.group(1)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Last resort: try to find any JSON-like structure
            try:
                # Try extracting from middle of text
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content[:200]}")

    def _safe_get(self, data: dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get a value from parsed JSON.

        Args:
            data: Parsed JSON dictionary
            key: Key to retrieve
            default: Default value if key missing

        Returns:
            Value or default
        """
        return data.get(key, default)


class AgentExecutionError(Exception):
    """Error raised when an agent fails to execute."""

    def __init__(self, agent_name: str, message: str, state: AgentState | None = None):
        self.agent_name = agent_name
        self.state = state
        super().__init__(f"{agent_name} agent failed: {message}")
