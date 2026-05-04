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

import ui
from colors import C
from context import make_context_manager
from critic import Critic, is_approved
from executor import Executor
from feedback import make_feedback_engine
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

        # Mid-execution replanning: when a step hits the iteration cap, ask
        # the planner what to do (retry / revise_step / revise_plan / skip / abort).
        self.replanning_enabled = os.getenv("AGENT_REPLANNING_ENABLED", "true").lower() == "true"
        max_replans_per_step = int(os.getenv("AGENT_MAX_REPLANS_PER_STEP", "2"))
        max_replans_per_run = int(os.getenv("AGENT_MAX_REPLANS_PER_RUN", "5"))

        # Autonomy mode selects how much ceremony each task gets:
        #   auto         → triage classifies (simple → no plan; standard → silent plan; complex → full review)
        #   interactive  → always full plan + critique + user review (current legacy)
        #   silent       → always plan, never prompt the user for review
        self.autonomy = os.getenv("AGENT_AUTONOMY", "auto").lower().strip()
        if self.autonomy not in ("auto", "interactive", "silent"):
            self.autonomy = "auto"

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

        # Feedback engine: post-tool deterministic verifiers (None when disabled).
        self.feedback = make_feedback_engine(logger=self.logger)

        # Context manager: compresses old messages into a running summary
        # before each LLM call (None when disabled).
        self.context = make_context_manager(logger=self.logger)

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
            logger=self.logger, feedback=self.feedback,
            context_manager=self.context,
            planner=self.planner,
            replanning_enabled=self.replanning_enabled,
            max_replans_per_step=max_replans_per_step,
            max_replans_per_run=max_replans_per_run,
            autonomy=self.autonomy,
        )
        self.critic = Critic(
            client=self.client, model=self.model, max_tokens=max_tokens_critic,
            logger=self.logger,
        )

        self.state = AgentState()

    def run(self, task: str) -> str:
        """Run the agent end-to-end: triage → execute → critic.

        Behavior depends on `self.autonomy`:
          - "auto" (default): triage classifies the task. Simple tasks skip
            planning entirely; standard tasks get a silent plan; complex
            tasks get full plan + critique + user review.
          - "interactive": forces complex path (always show plan + critique).
          - "silent": forces standard path (plan + execute, no user prompt).
        """
        self.state = AgentState(task=task)
        self.logger.start(task)

        if not self.planning_enabled:
            result = self.executor.legacy_execute(self.state, task)
        else:
            # Decide mode (skip the triage LLM call when overridden).
            if self.autonomy == "interactive":
                mode = "complex"
            elif self.autonomy == "silent":
                mode = "standard"
            else:
                mode = self.planner.triage(task)

            print(C.phase("\n🤖 Starting agent for task:") + f" {task}")
            mode_summary = {
                "simple":   "skipping plan, executing directly",
                "standard": "auto-approved plan, no review needed",
                "complex":  "full plan + critique + user review",
            }.get(mode, mode)
            print(C.dim(f"   autonomy={self.autonomy}, complexity={mode} → {mode_summary}\n"))

            try:
                if mode == "simple":
                    # Reset message accumulator so a fresh task doesn't reuse stale context.
                    self.executor._messages = []
                    result = self.executor.run_tao_loop(self.state, self.max_iterations)
                elif mode == "standard":
                    plan = self.planner.initial_plan(task)
                    self._print_plan_summary(plan)
                    self.logger.log("plan_approved", {
                        "plan": plan, "refinement_rounds": 0, "mode": "standard",
                    })
                    result = self.executor.execute_plan(self.state, task, plan)
                else:  # complex
                    plan = self.planner.plan_refinement_loop(task)
                    result = self.executor.execute_plan(self.state, task, plan)
            except KeyboardInterrupt as e:
                msg = f"Cancelled at plan refinement: {e}"
                self.logger.end(msg, "CANCELLED")
                return msg

        print(C.phase("\n--- CRITIC REVIEW ---"))
        critic_feedback = self.critic.review(task, result, self.state.action_history)
        approved = is_approved(critic_feedback)
        print(f"{C.BOLD}Critic:{C.RESET} {critic_feedback}")

        if not approved:
            result += f"\n\n[Critic Feedback]: {critic_feedback}"

        report_path = self.logger.end(result, critic_feedback)

        # End-of-run dashboard with aggregated stats from the session log.
        ui.render_dashboard(self._collect_stats(task, approved, critic_feedback, report_path))

        return result

    def _collect_stats(self, task, approved, critic_feedback, report_path):
        """Gather stats from logger events + state for the dashboard."""
        from collections import Counter
        events = list(getattr(self.logger, "events", []) or [])
        tool_counts: Counter = Counter()
        replans = 0
        compressions = 0
        steps_total = len(self.state.plan.get("steps", [])) if self.state.plan else 0
        steps_completed = 0
        for e in events:
            kind = e.kind if hasattr(e, "kind") else e.get("kind")
            data = e.data if hasattr(e, "data") else e.get("data", {})
            if kind == "tool_call":
                t = data.get("tool")
                if t:
                    tool_counts[t] += 1
            elif kind == "replan_decided":
                replans += 1
            elif kind == "context_compressed":
                compressions += 1
            elif kind == "step_complete" and data.get("completed"):
                steps_completed += 1
        duration = events[-1].elapsed_s if events and hasattr(events[-1], "elapsed_s") else 0.0
        # Critic summary: first ~200 chars of the verdict.
        critic_summary = (critic_feedback or "").strip().splitlines()[0] if critic_feedback else ""
        return {
            "task": task,
            "approved": approved,
            "duration_s": duration,
            "steps_total": steps_total,
            "steps_completed": steps_completed,
            "replans": replans,
            "compressions": compressions,
            "tool_counts": dict(tool_counts),
            "critic_summary": critic_summary,
            "log_path": str(report_path) if report_path else "",
        }

    def _print_plan_summary(self, plan):
        """Print the plan compactly without prompting (used by 'standard' mode)."""
        ui.render_plan(plan, suggestions=None, title="📋 Plan (auto-approved)")


# Re-export for backward compatibility with `from agent import AutonomousAgent`.
__all__ = ["AutonomousAgent"]


if __name__ == "__main__":
    from cli import main
    main()
