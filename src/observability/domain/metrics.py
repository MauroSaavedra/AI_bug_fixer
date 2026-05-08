"""Metrics collection for bug fixing sessions.

This module provides comprehensive metrics tracking for the
bug fixing workflow, enabling performance analysis and optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentMetrics:
    """Metrics for a single agent execution.

    Tracks timing, token usage, and outcome for each agent
    in the orchestration pipeline.
    """

    agent_name: str
    start_time: datetime
    end_time: datetime | None = None
    duration_ms: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    success: bool = True
    error: str | None = None
    retry_count: int = 0

    def finalize(self, success: bool = True, error: str | None = None):
        """Finalize metrics after execution completes."""
        self.end_time = datetime.now()
        self.duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
        self.success = success
        self.error = error

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.tokens_input + self.tokens_output


@dataclass
class SessionMetrics:
    """Metrics for a complete bug fixing session.

    Aggregates all metrics from a single bug fixing workflow,
    from detection through final fix (or failure).
    """

    # Session identification
    session_id: str
    bug_description: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Timing
    total_duration_ms: int = 0
    detection_duration_ms: int = 0
    planning_duration_ms: int = 0
    retrieval_duration_ms: int = 0
    coding_duration_ms: int = 0
    review_duration_ms: int = 0

    # Token usage
    total_tokens_input: int = 0
    total_tokens_output: int = 0

    # Outcome
    final_status: str = "running"  # running, approved, rejected, failed
    retry_count: int = 0
    success: bool = False

    # Agent-specific metrics
    agent_metrics: list[AgentMetrics] = field(default_factory=list)

    # Additional data
    metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(self, status: str, success: bool = False):
        """Finalize session metrics."""
        self.end_time = datetime.now()
        self.total_duration_ms = int(
            (self.end_time - self.start_time).total_seconds() * 1000
        )
        self.final_status = status
        self.success = success

    def add_agent_metrics(self, metrics: AgentMetrics):
        """Add metrics from an agent execution."""
        self.agent_metrics.append(metrics)

        # Update aggregates
        self.total_tokens_input += metrics.tokens_input
        self.total_tokens_output += metrics.tokens_output

        if metrics.agent_name == "Planner":
            self.planning_duration_ms = metrics.duration_ms
        elif metrics.agent_name == "Coder":
            self.coding_duration_ms = metrics.duration_ms
        elif metrics.agent_name == "Reviewer":
            self.review_duration_ms = metrics.duration_ms

    @property
    def total_tokens(self) -> int:
        """Total tokens used in session."""
        return self.total_tokens_input + self.total_tokens_output

    @property
    def llm_duration_ms(self) -> int:
        """Total time spent in LLM calls."""
        return sum(m.duration_ms for m in self.agent_metrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "bug_description": self.bug_description,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "detection_duration_ms": self.detection_duration_ms,
            "planning_duration_ms": self.planning_duration_ms,
            "retrieval_duration_ms": self.retrieval_duration_ms,
            "coding_duration_ms": self.coding_duration_ms,
            "review_duration_ms": self.review_duration_ms,
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_tokens": self.total_tokens,
            "final_status": self.final_status,
            "success": self.success,
            "retry_count": self.retry_count,
            "agent_metrics": [
                {
                    "agent_name": m.agent_name,
                    "duration_ms": m.duration_ms,
                    "tokens_input": m.tokens_input,
                    "tokens_output": m.tokens_output,
                    "success": m.success,
                    "retry_count": m.retry_count,
                }
                for m in self.agent_metrics
            ],
            "metadata": self.metadata,
        }


class MetricsCollector:
    """Collects and stores metrics across multiple sessions.

    Provides statistics and reporting capabilities for analyzing
    system performance over time.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._sessions: list[SessionMetrics] = []

    def record_session(self, metrics: SessionMetrics) -> None:
        """Record a completed session."""
        self._sessions.append(metrics)

    def get_success_rate(self) -> float:
        """Calculate overall success rate.

        Returns:
            Success rate as percentage (0.0-1.0)
        """
        if not self._sessions:
            return 0.0
        successful = sum(1 for s in self._sessions if s.success)
        return successful / len(self._sessions)

    def get_average_duration_ms(self) -> float:
        """Calculate average session duration."""
        if not self._sessions:
            return 0.0
        return sum(s.total_duration_ms for s in self._sessions) / len(self._sessions)

    def get_average_retries(self) -> float:
        """Calculate average number of retries needed."""
        if not self._sessions:
            return 0.0
        return sum(s.retry_count for s in self._sessions) / len(self._sessions)

    def get_total_tokens(self) -> int:
        """Get total tokens used across all sessions."""
        return sum(s.total_tokens for s in self._sessions)

    def get_agent_success_rates(self) -> dict[str, float]:
        """Calculate success rate per agent.

        Returns:
            Dictionary mapping agent names to success rates
        """
        agent_stats: dict[str, dict] = {}

        for session in self._sessions:
            for metric in session.agent_metrics:
                if metric.agent_name not in agent_stats:
                    agent_stats[metric.agent_name] = {"success": 0, "total": 0}

                agent_stats[metric.agent_name]["total"] += 1
                if metric.success:
                    agent_stats[metric.agent_name]["success"] += 1

        return {
            name: stats["success"] / stats["total"]
            for name, stats in agent_stats.items()
            if stats["total"] > 0
        }

    def generate_report(self) -> str:
        """Generate human-readable summary report."""
        if not self._sessions:
            return "No sessions recorded yet."

        lines = [
            "=" * 60,
            "📊 METRICS REPORT",
            "=" * 60,
            f"",
            f"Total Sessions: {len(self._sessions)}",
            f"Success Rate: {self.get_success_rate():.1%}",
            f"Average Duration: {self.get_average_duration_ms()/1000:.2f}s",
            f"Average Retries: {self.get_average_retries():.1f}",
            f"Total Tokens: {self.get_total_tokens():,}",
            f"",
            "Agent Success Rates:",
        ]

        for agent, rate in self.get_agent_success_rates().items():
            lines.append(f"  {agent}: {rate:.1%}")

        return "\n".join(lines)

    def export_to_json(self, filepath: str) -> None:
        """Export all sessions to JSON file."""
        import json

        data = {
            "summary": {
                "total_sessions": len(self._sessions),
                "success_rate": self.get_success_rate(),
                "average_duration_ms": self.get_average_duration_ms(),
                "average_retries": self.get_average_retries(),
                "total_tokens": self.get_total_tokens(),
            },
            "sessions": [s.to_dict() for s in self._sessions],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# Global collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector
