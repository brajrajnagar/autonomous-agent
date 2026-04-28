"""AutonomousAgent: thin orchestrator wiring Planner, Executor, and Critic.

Entry point for `python src/agent.py [task]`. The actual phases live in
their own modules (planner.py, executor.py, critic.py); this file only
configures them from environment variables and chains them inside `run()`.
"""

import os
import sys

# Make sibling modules importable regardless of invocation style
# (`python src/agent.py` vs. `from src.agent import ...`).
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from agent/config/.env on import.
_project_root = os.path.dirname(_script_dir)
load_dotenv(os.path.join(_project_root, "config", ".env"))

from colors import C
from critic import Critic
from executor import Executor
from planner import Planner
from session_log import make_logger
from state import AgentState
from tools import Tools


class AutonomousAgent:
    """Top-level agent: plan → critique → refine → execute → critic.

    Each subsystem is constructed once in `__init__` from env vars; `run()`
    builds a fresh `AgentState` per task and threads it through the pipeline.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        self.planning_enabled = os.getenv("AGENT_PLANNING_ENABLED", "true").lower() == "true"
        max_plan_refinements = int(os.getenv("AGENT_MAX_PLAN_REFINEMENTS", "5"))

        # Per-loop max_tokens budgets.
        max_tokens_tao = int(os.getenv("AGENT_MAX_TOKENS_TAO", "15000"))
        max_tokens_critic = int(os.getenv("AGENT_MAX_TOKENS_CRITIC", "5000"))
        max_tokens_plan_initial = int(os.getenv("AGENT_MAX_TOKENS_PLAN_INITIAL", "10000"))
        max_tokens_plan_critique = int(os.getenv("AGENT_MAX_TOKENS_PLAN_CRITIQUE", "15000"))
        max_tokens_plan_refine = int(os.getenv("AGENT_MAX_TOKENS_PLAN_REFINE", "15000"))

        # Session logging — best-effort; resolves to a no-op logger if disabled.
        logging_enabled = os.getenv("AGENT_LOGGING_ENABLED", "true").lower() == "true"
        log_dir = os.getenv("AGENT_LOG_DIR", os.path.join(_project_root, "logs"))
        self.logger = make_logger(logging_enabled, log_dir)

        self.tools = Tools()
        self.planner = Planner(
            client=self.client, model=self.model,
            max_tokens_initial=max_tokens_plan_initial,
            max_tokens_critique=max_tokens_plan_critique,
            max_tokens_refine=max_tokens_plan_refine,
            max_refinements=max_plan_refinements,
            logger=self.logger,
        )
        self.executor = Executor(
            client=self.client, model=self.model, tools=self.tools,
            max_iterations=self.max_iterations, max_tokens_tao=max_tokens_tao,
            logger=self.logger,
        )
        self.critic = Critic(
            client=self.client, model=self.model, max_tokens=max_tokens_critic,
            logger=self.logger,
        )

        self.state = AgentState()

    def run(self, task: str) -> str:
        """Run the agent end-to-end: plan → execute → critic."""
        self.state = AgentState(task=task)
        self.logger.start(task)

        if self.planning_enabled:
            print(C.phase("\n🤖 Starting agent for task:") + f" {task}")
            try:
                plan = self.planner.plan_refinement_loop(task)
            except KeyboardInterrupt as e:
                msg = f"Cancelled at plan refinement: {e}"
                self.logger.end(msg, "CANCELLED")
                return msg
            result = self.executor.execute_plan(self.state, task, plan)
        else:
            result = self.executor.legacy_execute(self.state, task)

        print(C.phase("\n--- CRITIC REVIEW ---"))
        critic_feedback = self.critic.review(task, result, self.state.action_history)
        print(f"{C.BOLD}Critic:{C.RESET} {critic_feedback}")

        if "APPROVED" in critic_feedback.upper():
            print(C.ok("✅ Output approved by critic"))
        else:
            print(C.warn("⚠️ Critic suggested improvements"))
            result += f"\n\n[Critic Feedback]: {critic_feedback}"

        report_path = self.logger.end(result, critic_feedback)
        if report_path:
            print(C.dim(f"\n📝 Session report: {report_path}"))

        return result


# Re-export for backward compatibility with `from agent import AutonomousAgent`.
__all__ = ["AutonomousAgent"]


if __name__ == "__main__":
    from cli import main
    main()
