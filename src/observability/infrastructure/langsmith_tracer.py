import os
from contextlib import contextmanager
from typing import Any, Generator
from loguru import logger
from langsmith import Client, traceable
from langsmith.run_trees import RunTree

from src.agent_orchestration.domain.state import AgentState
from src.config import get_settings


class LangSmithTracer:
    """LangSmith tracer for bug fixing sessions.

    Provides tracing for:
    - Complete bug fixing sessions
    - Individual agent executions
    - LLM calls with prompts and responses
    - State transitions
    - Metrics collection

    Usage:
        tracer = LangSmithTracer()

        with tracer.trace_session("session_id", "Fix division by zero"):
            # Run bug fixing workflow
            pass
    """

    def __init__(self, project_name: str | None = None, config=None, client=None):
        """Initialize LangSmith tracer.

        Args:
            project_name: LangSmith project name. If None, uses config.
        """
        self.config = config or get_settings()

        self.project_name = project_name or self.config.langchain_project
        self.enabled = bool(
        self.config.langchain_tracing_v2 and self.config.langchain_api_key
        )

        self.client = client if self.enabled else None
        
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.enabled

    @contextmanager
    def trace_session(
        self,
        session_id: str,
        bug_description: str,
    ) -> Generator[RunTree | None, None, None]:
        """Trace a complete bug fixing session.

        Args:
            session_id: Unique session identifier
            bug_description: Description of the bug being fixed

        Yields:
            RunTree if tracing enabled, None otherwise
        """
        if not self.enabled or self.client is None:
            yield None
            return

        run = self.client.create_run(
            name="bug_fix_session",
            run_type="chain",
            inputs={"bug_description": bug_description},
            project_name=self.project_name,
            id=session_id,
        )

        try:
            yield run

            self.client.update_run(
                run_id=run.id,
                outputs={"status": "completed"},
            )

        except Exception as e:
            self.client.update_run(
                run_id=run.id,
                error=str(e),
                outputs={"status": "failed"},
            )
            raise

    @contextmanager
    def trace_agent(
        self,
        agent_name: str,
        session_id: str,
        state: AgentState | None = None,
    ) -> Generator[RunTree | None, None, None]:
        """Trace an agent execution.

        Args:
            agent_name: Name of the agent (Planner, Coder, Reviewer)
            session_id: Parent session ID
            state: Current agent state (optional)

        Yields:
            RunTree if tracing enabled, None otherwise
        """
        if not self.enabled:
            yield None
            return

        inputs = {"agent": agent_name}

        if state:
            inputs["status"] = state.status
            inputs["retry_count"] = state.retry_count

        run = self.client.create_run(
            name=f"{agent_name}_agent",
            run_type="tool",
            inputs=inputs,
            parent_run_id=session_id,
            project_name=self.project_name,
        )

        try:
            yield run

            self.client.update_run(
                run_id=run.id,
                outputs={"status": "completed"},
            )

        except Exception as e:
            self.client.update_run(
                run_id=run.id,
                error=str(e),
            )
            raise

    def trace_llm_call(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        model: str = "unknown",
    ) -> None:
        """Trace an LLM call.

        Args:
            session_id: Parent session ID
            messages: Input messages
            response: LLM response
            tokens_in: Input token count
            tokens_out: Output token count
            model: Model name used
        """
        if not self.enabled or self.client is None:
            return

        self.client.create_run(
            name="llm_call",
            run_type="llm",
            inputs={"messages": messages},
            outputs={"response": response},
            parent_run_id=session_id,
            project_name=self.project_name,
            extra={
                "model": model,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "tokens_total": tokens_in + tokens_out,
            },
        )

    def trace_state_transition(
        self,
        session_id: str,
        from_status: str,
        to_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Trace a state transition.

        Args:
            session_id: Parent session ID
            from_status: Previous state
            to_status: New state
            metadata: Additional metadata
        """
        if not self.enabled:
            return

        self.client.create_run(
            name="state_transition",
            run_type="tool",
            inputs={"from": from_status},
            outputs={"to": to_status},
            parent_run_id=session_id,
            project_name=self.project_name,
            extra=metadata or {},
        )

    def export_session(self, session_id: str, filepath: str) -> None:
        """Export session traces to JSON.

        Args:
            session_id: Session to export
            filepath: Output JSON file path
        """
        if not self.enabled or self.client is None:
            logger.info("Tracing not enabled, nothing to export")
            return

        try:
            # Get run from LangSmith
            run = self.client.read_run(session_id)

            data = {
                "session_id": session_id,
                "name": run.name,
                "inputs": run.inputs,
                "outputs": run.outputs,
                "start_time": (
                    run.start_time.isoformat()
                    if run.start_time
                    else None
                ),
                "end_time": (
                    run.end_time.isoformat()
                    if run.end_time
                    else None
                ),
            }

            import json

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported session to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export session: {e}")


# Decorator for easy tracing
def trace_agent_execution(agent_name: str):
    """Decorator to trace agent execution.

    Usage:
        @trace_agent_execution("Planner")
        async def execute(
            self,
            state: AgentState,
        ) -> AgentState:
            # Agent logic
            pass
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get tracer from context or create new
            tracer = LangSmithTracer()

            # Get session ID from state if available
            session_id = None

            for arg in args:
                if isinstance(arg, AgentState):
                    session_id = arg.user_goal  # Use goal as session ID
                    break

            if session_id is None:
                return await func(*args, **kwargs)

            with tracer.trace_agent(agent_name, session_id) as run:
                result = await func(*args, **kwargs)

                # Update with result
                if run and result:
                    tracer.client.update_run(
                        run_id=run.id,
                        outputs={"status": result.status},
                    )

                return result

        return wrapper

    return decorator


# Global tracer instance
_tracer: LangSmithTracer | None = None


def get_tracer() -> LangSmithTracer:
    """Get or create global LangSmith tracer."""
    global _tracer

    if _tracer is None:
        _tracer = LangSmithTracer()

    return _tracer
