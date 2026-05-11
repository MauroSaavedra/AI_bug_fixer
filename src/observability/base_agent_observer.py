"""Decorators and context managers for Langfuse observability.

Provides reusable tracing utilities for agent executions, LLM calls,
and high-level bug detection workflows using manual Langfuse API integration.

Usage:
    from observability import trace_agent_execution, trace_llm_call

    with trace_agent_execution("planner_agent"):
        # planner logic

    with trace_llm_call(provider="openai", model="gpt-4o", temperature=0.1):
        # LLM call
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger

from src.observability.langfuse_tracer import get_tracer


@contextmanager
def trace_agent_execution(agent_name: str) -> Generator[None, None, None]:
    """Context manager to create a span for a single agent execution.

    Usage:
        with trace_agent_execution("planner_agent"):
            state = await planner.execute(state)
    """
    tracer = get_tracer()
    if not tracer.is_enabled:
        yield
        return

    try:
        yield
    finally:
        try:
            tracer.langfuse.update_current_span(
                metadata={"agent_name": agent_name}
            )
        except Exception:
            pass
