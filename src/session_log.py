"""Session logging: per-event JSONL stream + Markdown summary report.

One `SessionLogger` instance is created per `AutonomousAgent.run()`. It
appends structured events to `<log_dir>/<UTC-timestamp>-<task-slug>.jsonl`
as the agent works, then renders a human-readable Markdown report to a
sibling `.md` file at session end.

The logger is best-effort: failures to write logs never crash the agent.
"""

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SessionEvent:
    """One structured moment in a session.

    `kind` enumerates the event type (see SessionLogger docstring for the
    full set). `data` carries event-specific payload.
    """
    timestamp: str
    elapsed_s: float
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)


class _NullLogger:
    """No-op logger used when AGENT_LOGGING_ENABLED=false.

    All `log(...)` calls become no-ops; `start()` / `end()` return None.
    Subsystems can hold a logger reference unconditionally.
    """

    enabled = False

    def start(self, task: str) -> None: pass
    def log(self, kind: str, data: Optional[Dict[str, Any]] = None) -> None: pass
    def end(self, result: str, critic_verdict: str) -> Optional[Path]: return None


class SessionLogger:
    """Append-only structured logger for one agent session.

    Event kinds emitted by the agent subsystems:
      session_start, plan_generated, plan_critique, plan_refinement,
      plan_approved, step_start, iteration, tool_call, tool_result,
      parse_error, repeated_action, loop_break, step_complete,
      critic_review, session_end
    """

    enabled = True

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[SessionEvent] = []
        self.session_id: Optional[str] = None
        self.task: Optional[str] = None
        self.session_path: Optional[Path] = None
        self._t0: float = 0.0

    def start(self, task: str) -> None:
        self._t0 = time.monotonic()
        self.task = task
        slug = re.sub(r"[^a-z0-9]+", "-", task.lower())[:40].strip("-") or "task"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = f"{ts}-{slug}"
        self.session_path = self.log_dir / f"{self.session_id}.jsonl"
        self.log("session_start", {
            "task": task,
            "model": os.getenv("OPENAI_MODEL", ""),
            "planning_enabled": os.getenv("AGENT_PLANNING_ENABLED", "true"),
        })

    def log(self, kind: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self.session_path is None:
            return
        event = SessionEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_s=time.monotonic() - self._t0,
            kind=kind,
            data=data or {},
        )
        self.events.append(event)
        try:
            with self.session_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        except Exception:
            # Logger is best-effort: a write failure must not break the agent.
            pass

    def end(self, result: str, critic_verdict: str) -> Optional[Path]:
        self.log("session_end", {
            "result_preview": result[:500],
            "critic_verdict": critic_verdict,
            "total_events": len(self.events),
        })
        if self.session_path is None:
            return None
        report_path = self.session_path.with_suffix(".md")
        try:
            report_path.write_text(self._render_report(result, critic_verdict), encoding="utf-8")
        except Exception:
            return None
        return report_path

    # ---- Report rendering ---------------------------------------------------

    def _render_report(self, result: str, critic_verdict: str) -> str:
        approved = "APPROVED" in critic_verdict.upper()
        outcome = "✅ APPROVED" if approved else "⚠️ NEEDS WORK"
        duration = self.events[-1].elapsed_s if self.events else 0.0

        plan_rounds = self._collect("plan_generated") + self._collect("plan_refinement")
        refinements = self._collect("plan_refinement")
        steps = self._collect("step_start")
        completes = {e.data.get("step_id"): e for e in self._collect("step_complete")}
        tool_calls = self._collect("tool_call")
        parse_errors = self._collect("parse_error")
        repeated = self._collect("repeated_action")
        loop_breaks = self._collect("loop_break")

        out: List[str] = []
        out.append(f"# Session: {self.task}\n")
        out.append(f"**Session ID:** `{self.session_id}`  ")
        out.append(f"**Started:** {self.events[0].timestamp if self.events else ''}  ")
        out.append(f"**Duration:** {duration:.1f}s  ")
        out.append(f"**Outcome:** {outcome}\n")

        # ----- Plan refinement journey -----
        out.append("## Plan refinement journey\n")
        if not plan_rounds:
            out.append("_Planning was disabled or no plan was generated._\n")
        else:
            initial = self._first("plan_generated")
            if initial:
                out.append("### Round 0 — initial plan\n")
                out.append(_format_plan(initial.data.get("plan", {})))
            for i, ev in enumerate(refinements, 1):
                user_input = ev.data.get("user_input", "")
                new_plan = ev.data.get("plan", {})
                out.append(f"\n### Round {i} — user feedback\n")
                out.append(f"> {user_input}\n")
                out.append("\n**Refined plan:**\n")
                out.append(_format_plan(new_plan))
            approved_ev = self._first("plan_approved")
            if approved_ev:
                rounds_used = approved_ev.data.get("refinement_rounds", len(refinements))
                out.append(f"\n_Approved after {rounds_used} refinement round(s)._\n")

        # ----- Step execution table -----
        out.append("\n## Step execution\n")
        if not steps:
            out.append("_No plan steps recorded (legacy execution or skipped)._\n")
        else:
            out.append("| Step | Description | Iterations | Tools | Outcome |")
            out.append("|------|-------------|-----------:|-------|---------|")
            for s_ev in steps:
                sid = s_ev.data.get("step_id")
                desc = s_ev.data.get("description", "")[:80]
                done = completes.get(sid)
                iters = done.data.get("iterations_used", "?") if done else "?"
                step_tools = [t.data.get("tool", "") for t in tool_calls
                              if t.data.get("step_id") == sid]
                tool_summary = ", ".join(f"{k}×{v}" for k, v in Counter(step_tools).items()) or "—"
                outcome_cell = "✅" if done and done.data.get("completed") else "⚠️"
                out.append(f"| {sid} | {desc} | {iters} | {tool_summary} | {outcome_cell} |")

        # ----- Tool usage -----
        out.append("\n## Tool usage (overall)\n")
        if tool_calls:
            counts = Counter(t.data.get("tool", "") for t in tool_calls)
            for name, n in counts.most_common():
                out.append(f"- `{name}`: {n}")
        else:
            out.append("_No tool calls recorded._")

        # ----- Errors & warnings -----
        out.append("\n## Errors & warnings\n")
        any_error = False
        if parse_errors:
            out.append(f"- {len(parse_errors)} truncated / unparseable LLM responses")
            any_error = True
        if repeated:
            out.append(f"- {len(repeated)} repeated-action warnings")
            any_error = True
        if loop_breaks:
            out.append(f"- {len(loop_breaks)} forced loop breaks")
            any_error = True
        if not any_error:
            out.append("_None._")

        # ----- Critic verdict -----
        out.append("\n## Critic verdict\n")
        out.append(f"```\n{critic_verdict}\n```")

        # ----- Diagnostic signals (auto-extracted) -----
        out.append("\n## Diagnostic signals\n")
        signals = self._diagnostic_signals(refinements, steps, completes, parse_errors, repeated)
        if signals:
            for s in signals:
                out.append(f"- {s}")
        else:
            out.append("_No flags raised._")

        # ----- Raw event link -----
        out.append(f"\n---\n_Raw events: [{self.session_path.name}]({self.session_path.name}) "
                   f"({len(self.events)} events)_\n")
        return "\n".join(out)

    def _collect(self, kind: str) -> List[SessionEvent]:
        return [e for e in self.events if e.kind == kind]

    def _first(self, kind: str) -> Optional[SessionEvent]:
        for e in self.events:
            if e.kind == kind:
                return e
        return None

    def _diagnostic_signals(self, refinements, steps, completes,
                            parse_errors, repeated) -> List[str]:
        out: List[str] = []
        if len(refinements) >= 2:
            out.append(f"**Plan needed {len(refinements)} refinement rounds** — "
                       f"the planner likely missed some scope; review the user "
                       f"feedback verbatim in the plan-refinement section.")
        for s_ev in steps:
            sid = s_ev.data.get("step_id")
            done = completes.get(sid)
            iters = done.data.get("iterations_used", 0) if done else 0
            if iters >= 8:
                out.append(f"Step {sid} took {iters} iterations — its description "
                           f"or success criterion may be too vague.")
        if len(parse_errors) >= 2:
            out.append(f"{len(parse_errors)} truncated tool-call responses — "
                       f"consider raising `AGENT_MAX_TOKENS_TAO`.")
        if len(repeated) >= 3:
            out.append(f"{len(repeated)} repeated-action events — the agent got "
                       f"stuck retrying; inspect the offending tool call.")

        # Verification failures: a strong signal the agent wrote broken code.
        verifications = self._collect("verification")
        failures = [v for v in verifications
                    if not v.data.get("passed") and not v.data.get("skipped")]
        if failures:
            from collections import Counter
            by_verifier = Counter(v.data.get("verifier", "?") for v in failures)
            breakdown = ", ".join(f"{n}× {name}" for name, n in by_verifier.most_common())
            out.append(f"{len(failures)} post-tool verification failures ({breakdown}) — "
                       f"agent shipped broken code that automatic checks caught.")
        return out


def make_logger(enabled: bool, log_dir: str = "logs"):
    """Factory: returns a SessionLogger when enabled, else a _NullLogger."""
    if enabled:
        return SessionLogger(log_dir)
    return _NullLogger()


def _format_plan(plan: Dict[str, Any]) -> str:
    """Render a plan dict as compact Markdown."""
    if not plan:
        return "_(empty plan)_\n"
    lines = [f"**{plan.get('summary', '')}**\n"]
    for step in plan.get("steps", []):
        lines.append(f"  {step.get('id', '?')}. {step.get('description', '')}")
        crit = step.get("success_criterion", "")
        if crit:
            lines.append(f"     ↳ _success: {crit}_")
    return "\n".join(lines) + "\n"
