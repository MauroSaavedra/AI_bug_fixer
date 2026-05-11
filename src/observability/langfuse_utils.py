"""Langfuse observability helpers.

Provides thin wrappers around the Langfuse v4 API to create/update spans,
traces, and observations safely.  All helpers below **gracefully degrade**
when Langfuse is disabled or not configured.

Usage:
    from src.observability import (trace_llm_call, trace_agent,
                                     trace_bug_workflow, trace_bug_detection)

    with trace_llm_call(model="gpt-4o", temperature=0.1):
        response = llm.chat(...)

    with trace_agent(name="planner_agent"):
        state = await planner.execute(state)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None  # type: ignore[assignment]

from src.observability.langfuse_tracer import get_tracer


@contextmanager
def trace_llm_call(
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    agent_name: str | None = None,
    provider: str | None = None,
) -> Generator[Any, None, None]:
    """Context manager that creates a *generation* observation and updates it.

    Usage inside an @observe()-decorated function:
        with trace_llm_call(model="gpt-4o", temperature=0.1):
            response = await openai.chat.completions.create(...)
            # model / params metadata are attached to the generation
    """
    lf = _get_lf_client()
    if lf is None:
        yield None
        return

    try:
        meta: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider": provider,
            "agent_name": agent_name,
        }
        start = time.time()
        yield None
    finally:
        try:
            meta["__latency_ms"] = (time.time() - start) * 1000
            lf.update_current_generation(
                model=model,
                model_parameters={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                metadata=meta,
            )
        except Exception as exc:
            _log_ignore(exc)


@contextmanager
def trace_agent(
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager creating a span observation for an agent.

    Usage:
        with trace_agent("planner_agent", metadata={"user_goal": ...}):
            state = await planner.execute(state)
    """
    lf = _get_lf_client()
    if lf is None:
        yield None
        return

    try:
        _meta: dict[str, Any] = dict(metadata) if metadata else {}
        start = time.time()
        yield None
    finally:
        try:
            _meta["__latency_ms"] = (time.time() - start) * 1000
            lf.update_current_span(metadata=_meta)
        except Exception as exc:
            _log_ignore(exc)


@contextmanager
def trace_bug_workflow() -> Generator[Any, None, None]:
    """Context manager wrapping an entire bug-fix workflow."""
    lf = _get_lf_client()
    if lf is None:
        yield None
        return

    start = time.time()
    try:
        yield None
    finally:
        try:
            lf.update_current_span(
                metadata={"__latency_ms": (time.time() - start) * 1000}
            )
        except Exception as exc:
            _log_ignore(exc)


@contextmanager
def trace_bug_detection() -> Generator[Any, None, None]:
    """Context manager wrapping a bug-detection run."""
    # Alias — exact same behaviour as trace_bug_workflow
    lf = _get_lf_client()
    if lf is None:
        yield None
        return

    start = time.time()
    try:
        yield None
    finally:
        try:
            lf.update_current_span(
                metadata={"__latency_ms": (time.time() - start) * 1000}
            )
        except Exception as exc:
            _log_ignore(exc)


def update_current_generation(
    *,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    usage_details: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
    prompt: Any | None = None,
) -> None:
    """Update the current generation observation with LLM-specific metadata.

    This should be called from inside a function decorated with
    @observe(as_type='generation') to enrich the generation trace.
    """
    lf = _get_lf_client()
    if lf is None:
        return

    try:
        lf.update_current_generation(
            model=model,
            model_parameters=model_parameters,
            usage_details=usage_details,
            metadata=metadata,
            prompt=prompt,
        )
    except Exception as exc:
        _log_ignore(exc)


def update_current_span(
    *,
    metadata: dict[str, Any] | None = None,
    output: Any | None = None,
    level: str | None = None,
) -> None:
    """Update the current span observation with arbitrary metadata.

    This should be called from inside a function decorated with
    @observe(as_type='span') to enrich the span trace.
    """
    lf = _get_lf_client()
    if lf is None:
        return

    try:
        lf.update_current_span(
            metadata=metadata,
            output=output,
            level=level,
        )
    except Exception as exc:
        _log_ignore(exc)


# ── internal helpers ────────────────────────────────────────────────────────


def _get_lf_client() -> Any | None:
    """Return the raw Langfuse client if tracing is enabled."""
    tracer = get_tracer()
    if not tracer.is_enabled:
        return None
    return tracer.langfuse


def _log_ignore(exc: Exception) -> None:
    """Log an observability warning but do not raise."""
    logger.debug(f"Langfuse update skipped: {exc}")
