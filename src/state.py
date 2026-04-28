"""Dataclasses for agent state and tool results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Result from executing a tool.

    success: whether the tool execution succeeded
    output:  output text (stdout or file content)
    error:   error message if execution failed
    """

    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class AgentState:
    """State carried through one `run()` invocation.

    Holds the task, the running history of thoughts/actions/observations,
    and (when planning is enabled) the active plan plus the step currently
    being executed.
    """

    task: str = ""
    thought_history: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    observation_history: List[str] = field(default_factory=list)
    iteration_count: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    current_step: Optional[Dict[str, Any]] = None
