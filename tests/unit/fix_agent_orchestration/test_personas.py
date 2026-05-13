"""Unit tests for agent personas.

Tests the persona configurations and prompt templates.
"""

import pytest

from src.fix_agent_orchestration.domain.personas import (
    AgentPersona,
    AgentPersonas,
    build_retry_context,
)


class TestAgentPersonas:
    """Test suite for AgentPersonas."""

    def test_planner_persona(self):
        """Test Planner persona configuration."""
        persona = AgentPersonas.get_persona("Planner")

        assert persona.name == "Planner"
        assert "search query" in persona.system_prompt.lower()
        assert "RAG" in persona.system_prompt
        assert "{user_goal}" in persona.task_prompt_template
        assert "search_query" in persona.output_format_description

    def test_coder_persona(self):
        """Test Coder persona configuration."""
        persona = AgentPersonas.get_persona("Coder")

        assert persona.name == "Coder"
        assert "fix" in persona.system_prompt.lower()
        assert "code" in persona.system_prompt.lower()
        assert "{user_goal}" in persona.task_prompt_template
        assert "{retrieved_code}" in persona.task_prompt_template
        assert "proposed_fix" in persona.output_format_description

    def test_reviewer_persona(self):
        """Test Reviewer persona configuration."""
        persona = AgentPersonas.get_persona("Reviewer")

        assert persona.name == "Reviewer"
        assert "review" in persona.system_prompt.lower()
        assert "evaluate" in persona.system_prompt.lower()
        assert "is_approved" in persona.output_format_description

    def test_get_persona_case_sensitive(self):
        """Test that persona lookup is case sensitive."""
        planner = AgentPersonas.get_persona("Planner")
        assert planner.name == "Planner"

        # Should raise error for lowercase
        with pytest.raises(ValueError, match="Unknown agent"):
            AgentPersonas.get_persona("planner")

        # Should raise error for uppercase
        with pytest.raises(ValueError, match="Unknown agent"):
            AgentPersonas.get_persona("PLANNER")

    def test_get_persona_invalid(self):
        """Test that invalid persona raises error."""
        with pytest.raises(ValueError, match="Unknown agent"):
            AgentPersonas.get_persona("InvalidAgent")

    def test_persona_structure(self):
        """Test that all personas have required fields."""
        for name in ["Planner", "Coder", "Reviewer"]:
            persona = AgentPersonas.get_persona(name)
            assert isinstance(persona, AgentPersona)
            assert persona.name
            assert persona.system_prompt
            assert persona.task_prompt_template
            assert persona.output_format_description


class TestBuildRetryContext:
    """Test suite for build_retry_context function."""

    def test_empty_feedback(self):
        """Test with no feedback."""
        result = build_retry_context(None, [])
        assert result == ""

    def test_feedback_only(self):
        """Test with feedback only."""
        result = build_retry_context("Fix the logic error", [])
        assert "Previous Attempt Feedback" in result
        assert "Fix the logic error" in result

    def test_feedback_with_issues(self):
        """Test with feedback and issues."""
        result = build_retry_context(
            "The fix is incorrect",
            ["Issue 1: Logic error", "Issue 2: Syntax error"]
        )
        assert "Previous Attempt Feedback" in result
        assert "The fix is incorrect" in result
        assert "Issue 1: Logic error" in result
        assert "Issue 2: Syntax error" in result
