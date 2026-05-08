import pytest
from unittest.mock import MagicMock

from src.observability.infrastructure.langsmith_tracer import LangSmithTracer

@pytest.fixture
def disabled_tracer():
    config = MagicMock(
        langchain_tracing_v2=False,
        langchain_api_key="",
        langchain_project="test-project",
    )
    return LangSmithTracer(config=config, client=None)


@pytest.fixture
def enabled_tracer():
    config = MagicMock(
        langchain_tracing_v2=True,
        langchain_api_key="test-key",
        langchain_project="test-project",
    )

    client = MagicMock()

    return LangSmithTracer(config=config, client=client)

def test_init_enabled(enabled_tracer):
    assert enabled_tracer.project_name == "test-project"
    assert enabled_tracer.enabled is True
    assert enabled_tracer.client is not None


def test_init_disabled(disabled_tracer):
    assert disabled_tracer.project_name == "test-project"
    assert disabled_tracer.enabled is False
    assert disabled_tracer.client is None

def test_trace_session_disabled(disabled_tracer):
    with disabled_tracer.trace_session("session-1", "bug") as ctx:
        assert ctx is None


def test_trace_session_enabled(enabled_tracer):
    mock_run = MagicMock()
    mock_run.id = "run-123"

    enabled_tracer.client.create_run.return_value = mock_run

    with enabled_tracer.trace_session("session-1", "bug description") as ctx:
        assert ctx is not None
        assert ctx.id == "run-123"

    enabled_tracer.client.update_run.assert_called()

def test_trace_agent_enabled(enabled_tracer):
    mock_run = MagicMock()
    mock_run.id = "agent-run-1"

    enabled_tracer.client.create_run.return_value = mock_run

    state = MagicMock()
    state.status = "coding"
    state.retry_count = 2

    with enabled_tracer.trace_agent("Planner", "session-1", state) as ctx:
        assert ctx is not None
        assert ctx.id == "agent-run-1"

    enabled_tracer.client.update_run.assert_called_with(
        run_id="agent-run-1",
        outputs={"status": "completed"},
    )


def test_trace_agent_disabled(disabled_tracer):
    state = MagicMock()
    state.status = "coding"
    state.retry_count = 1

    with disabled_tracer.trace_agent("Planner", "session-1", state) as ctx:
        assert ctx is None

def test_trace_llm_disabled(disabled_tracer):
    # should not crash
    disabled_tracer.trace_llm_call(
        "session-1",
        [{"role": "user", "content": "hello"}],
        "response"
    )


def test_trace_llm_enabled(enabled_tracer):
    enabled_tracer.trace_llm_call(
        "session-1",
        [{"role": "user", "content": "hello"}],
        "response",
        model="gpt-4o",
        tokens_in=10,
        tokens_out=20,
    )

    enabled_tracer.client.create_run.assert_called_once()

def test_trace_state_transition_enabled(enabled_tracer):
    enabled_tracer.trace_state_transition(
        "session-1",
        "planning",
        "coding",
        {"extra": "metadata"}
    )

    enabled_tracer.client.create_run.assert_called_once()


def test_trace_state_transition_disabled(disabled_tracer):
    # should not crash
    enabled_tracer = disabled_tracer  # alias clarity

    enabled_tracer.trace_state_transition(
        "session-1",
        "planning",
        "coding"
    )

def test_is_enabled(enabled_tracer, disabled_tracer):
    assert enabled_tracer.is_enabled() is True
    assert disabled_tracer.is_enabled() is False
