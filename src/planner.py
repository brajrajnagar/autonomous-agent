"""Planner: produces, critiques, and refines an executable plan with the user."""

import json
from typing import Any, Dict, List, Optional

import ui
from colors import C
from parsing import safe_json_parse
from prompts import (
    INITIAL_PLAN_PROMPT, PLAN_CRITIQUE_PROMPT, PLAN_REFINE_PROMPT,
    REPLAN_PROMPT, TRIAGE_PROMPT, system_prefix,
)


class Planner:
    """Owns the Plan → Critique → Refine loop.

    The loop runs entirely before any tool is executed. The user can approve
    (`go` / `ok` / Enter), apply numbered suggestions (`apply 1,3`), or
    describe changes in free form.
    """

    def __init__(self, client, model: str,
                 max_tokens_initial: int, max_tokens_critique: int,
                 max_tokens_refine: int, max_refinements: int,
                 logger: Optional[Any] = None):
        self.client = client
        self.model = model
        self.max_tokens_initial = max_tokens_initial
        self.max_tokens_critique = max_tokens_critique
        self.max_tokens_refine = max_tokens_refine
        self.max_refinements = max_refinements
        self.logger = logger

    # ---- LLM calls -----------------------------------------------------------

    def _llm_call(self, system_msg: str, user_msg: str,
                  temperature: float, max_tokens: int,
                  label: str = "Thinking") -> str:
        """Single-shot LLM call returning the raw text response."""
        with ui.thinking(label):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prefix() + system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return (response.choices[0].message.content or "").strip()

    def triage(self, task: str) -> str:
        """Classify task complexity. Returns 'simple' | 'standard' | 'complex'.

        Cheap LLM call (~150 tokens). On any parse / value failure, defaults
        to 'standard' so the agent still does *some* planning rather than
        nothing. Logs the classification + reasoning for later analysis.
        """
        prompt = TRIAGE_PROMPT.format(task=task)
        # Note: 2000 tokens (not the cheaper ~200 you'd expect) because
        # reasoning models (e.g. Qwen3.5) consume the budget on internal
        # chain-of-thought before emitting the visible JSON. Setting this
        # too low produces an empty `content` and finish_reason='length'.
        raw = self._llm_call(
            "You are a task complexity classifier. Output strict JSON only.",
            prompt, temperature=0.1, max_tokens=2000,
            label="Triaging task complexity",
        )
        parsed = safe_json_parse(raw)
        VALID = ("simple", "standard", "complex")

        if not parsed or "complexity" not in parsed:
            if self.logger:
                self.logger.log("triage", {
                    "complexity": "standard", "fallback": True,
                    "raw_preview": raw[:200],
                })
            return "standard"

        complexity = str(parsed.get("complexity", "")).lower().strip()
        if complexity not in VALID:
            complexity = "standard"

        if self.logger:
            self.logger.log("triage", {
                "complexity": complexity,
                "reasoning": parsed.get("reasoning", ""),
            })
        return complexity

    def initial_plan(self, task: str) -> Dict[str, Any]:
        prompt = INITIAL_PLAN_PROMPT.format(task=task)
        raw = self._llm_call(
            "You are a senior planning agent. Output strict JSON only.",
            prompt, temperature=0.5, max_tokens=self.max_tokens_initial,
            label="Generating plan",
        )
        parsed = safe_json_parse(raw)
        if not parsed or "steps" not in parsed or not parsed["steps"]:
            print(C.warn("⚠️  Plan generation failed; falling back to single-step plan."))
            return {
                "summary": task,
                "steps": [{"id": 1, "description": task, "success_criterion": "Task is complete"}],
            }
        return _normalize_plan(parsed, task)

    def critique_plan(self, task: str, plan: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = PLAN_CRITIQUE_PROMPT.format(task=task, plan_json=json.dumps(plan, indent=2))
        raw = self._llm_call(
            "You are a critical planning reviewer. Output strict JSON only.",
            prompt, temperature=0.3, max_tokens=self.max_tokens_critique,
            label="Reviewing plan",
        )
        parsed = safe_json_parse(raw)
        if not parsed or "suggestions" not in parsed:
            return []
        return [s for s in parsed["suggestions"]
                if isinstance(s, dict) and "issue" in s and "fix" in s]

    def replan(self, task: str, original_plan: Dict[str, Any],
               failed_step: Dict[str, Any], failure_reason: str,
               recent_observations: List[str]) -> Dict[str, Any]:
        """LLM call: choose how to recover from a failed step.

        Returns a dict with at least `action` ∈ {"retry", "revise_step",
        "revise_plan", "skip", "abort"} and `reasoning`. Falls back to
        "retry" when revision payloads are malformed (better to give the
        agent another shot than to abort on a parser glitch).
        """
        obs_text = "\n".join(
            f"- {(o or '').replace(chr(10), ' ')[:200]}"
            for o in (recent_observations[-5:] if recent_observations else [])
        ) or "(no observations recorded)"
        prompt = REPLAN_PROMPT.format(
            task=task,
            plan_json=json.dumps(original_plan, indent=2),
            failed_step_json=json.dumps(failed_step, indent=2),
            failure_reason=failure_reason,
            observations=obs_text,
        )
        raw = self._llm_call(
            "You are a recovery planner. Output strict JSON only.",
            prompt, temperature=0.4, max_tokens=2000,
            label="Replanning after failure",
        )
        parsed = safe_json_parse(raw)
        VALID = ("retry", "revise_step", "revise_plan", "skip", "abort")

        if not parsed or parsed.get("action") not in VALID:
            return {
                "action": "retry",
                "reasoning": "replan response unparseable; retrying with fresh budget",
            }

        action = parsed["action"]
        reasoning = str(parsed.get("reasoning", "")).strip() or "(no reasoning provided)"

        if action == "revise_step":
            rs = parsed.get("revised_step")
            if not isinstance(rs, dict) or not rs.get("description"):
                return {"action": "retry",
                        "reasoning": "revise_step payload malformed; retrying instead"}
            rs.setdefault("id", failed_step.get("id", 1))
            rs.setdefault("success_criterion", failed_step.get("success_criterion", "Step is complete"))
            return {"action": "revise_step", "reasoning": reasoning, "revised_step": rs}

        if action == "revise_plan":
            rs_list = parsed.get("revised_steps")
            if not isinstance(rs_list, list) or not rs_list:
                return {"action": "retry",
                        "reasoning": "revise_plan payload malformed; retrying instead"}
            normalized = []
            for i, step in enumerate(rs_list, start=failed_step.get("id", 1)):
                if not isinstance(step, dict) or not step.get("description"):
                    continue
                step.setdefault("id", i)
                step.setdefault("success_criterion", "Step is complete")
                normalized.append(step)
            if not normalized:
                return {"action": "retry",
                        "reasoning": "all revised_steps malformed; retrying instead"}
            return {"action": "revise_plan", "reasoning": reasoning,
                    "revised_steps": normalized}

        return {"action": action, "reasoning": reasoning}

    def refine_plan(self, task: str, plan: Dict[str, Any], user_feedback: str,
                    suggestions: List[Dict[str, str]]) -> Dict[str, Any]:
        prompt = PLAN_REFINE_PROMPT.format(
            task=task,
            plan_json=json.dumps(plan, indent=2),
            suggestions_json=json.dumps(suggestions, indent=2),
            user_text=user_feedback,
        )
        raw = self._llm_call(
            "You are a planning agent revising a plan. Output strict JSON only.",
            prompt, temperature=0.4, max_tokens=self.max_tokens_refine,
            label="Refining plan",
        )
        parsed = safe_json_parse(raw)
        if not parsed or "steps" not in parsed or not parsed["steps"]:
            print(C.warn("⚠️  Plan refinement failed; keeping previous plan."))
            return plan
        return _normalize_plan(parsed, task, fallback_summary=plan.get("summary"))

    # ---- User interaction ----------------------------------------------------

    def present_plan_to_user(self, plan: Dict[str, Any],
                             suggestions: List[Dict[str, str]]) -> str:
        """Render the plan and let the user pick what to do.

        Returns one of:
        - "APPROVE" sentinel               → the plan was accepted as-is
        - "Apply suggestion(s) i,j..."     → checkbox selection of suggestions
        - free-form text                   → user wants the planner to revise

        Two prompt shapes:
        - With suggestions: a single CHECKBOX so multi-select is the
          primary action. Suggestions list first; "Describe my own
          changes" and "Cancel" are also there as toggleable special
          options. Empty selection = approve.
        - Without suggestions: a single SELECT with Approve/Describe/Cancel.

        On non-TTY (piped/captured input) the picker degrades to a
        numbered prompt so automation keeps working.
        """
        ui.render_plan(plan, suggestions)

        # No suggestions → simple single-select.
        if not suggestions:
            print(C.ok("💡 No suggestions — plan looks solid."))
            try:
                action = ui.choose("What would you like to do?", [
                    ("Approve plan as-is and execute", "APPROVE"),
                    ("Describe changes in my own words...", "FREEFORM"),
                    ("Cancel run", "CANCEL"),
                ], default="APPROVE")
            except KeyboardInterrupt:
                raise
            if action == "APPROVE":
                return "APPROVE"
            if action == "CANCEL":
                raise KeyboardInterrupt("User cancelled at plan review")
            free_text = ui.text_input("Describe the change you want:")
            return free_text or "APPROVE"

        # With suggestions → single multi-select picker. Sentinel options
        # ("describe", "cancel") sit alongside the actual suggestions so
        # the user sees every available action in one place.
        FREEFORM = "__FREEFORM__"
        CANCEL = "__CANCEL__"
        choices: List = [
            (f"[{i}] {s.get('issue', '')[:80]}", i)
            for i, s in enumerate(suggestions, 1)
        ]
        choices.append(("→ Describe changes in my own words instead", FREEFORM))
        choices.append(("✕ Cancel run", CANCEL))

        try:
            picked = ui.checkbox(
                "Select what to do — pick MULTIPLE suggestions to apply, or leave empty to approve as-is:",
                choices,
                instruction="(↑↓ move, Space to toggle, Enter to confirm)",
            )
        except KeyboardInterrupt:
            raise

        # CANCEL takes precedence over everything.
        if CANCEL in picked:
            raise KeyboardInterrupt("User cancelled at plan review")

        # FREEFORM takes precedence over apply selections.
        if FREEFORM in picked:
            free_text = ui.text_input("Describe the change you want:")
            return free_text or "APPROVE"

        # Empty (or only sentinels filtered out) → approve.
        suggestion_indices = [v for v in picked if isinstance(v, int)]
        if not suggestion_indices:
            return "APPROVE"

        indices_str = ",".join(str(i) for i in suggestion_indices)
        return f"Apply suggestion(s) {indices_str} from the critic list."

    def plan_refinement_loop(self, task: str) -> Dict[str, Any]:
        """Plan → critique → user → refine, until approved or cap hit."""
        print(C.phase("\n🧭 Generating initial plan..."))
        plan = self.initial_plan(task)
        if self.logger:
            self.logger.log("plan_generated", {"plan": plan, "round": 0})

        refinement_rounds = 0
        for round_num in range(1, self.max_refinements + 1):
            print(C.phase(f"\n🔍 Critiquing plan (round {round_num})..."))
            suggestions = self.critique_plan(task, plan)
            if self.logger:
                self.logger.log("plan_critique", {"round": round_num, "suggestions": suggestions})
            user_response = self.present_plan_to_user(plan, suggestions)
            if user_response == "APPROVE":
                print(C.ok("✅ Plan approved. Beginning execution."))
                if self.logger:
                    self.logger.log("plan_approved", {
                        "plan": plan,
                        "refinement_rounds": refinement_rounds,
                    })
                return plan
            print(C.phase("\n✏️  Refining plan based on feedback..."))
            plan = self.refine_plan(task, plan, user_response, suggestions)
            refinement_rounds += 1
            if self.logger:
                self.logger.log("plan_refinement", {
                    "round": round_num,
                    "user_input": user_response,
                    "plan": plan,
                })
        print(C.warn(f"\n⚠️  Reached max refinement rounds ({self.max_refinements})."))
        try:
            final = input("Type 'go' to execute the current plan, or 'cancel' to abort: ").strip().lower()
        except EOFError:
            final = "go"
        if final == "cancel":
            raise KeyboardInterrupt("User cancelled at plan refinement cap")
        if self.logger:
            self.logger.log("plan_approved", {
                "plan": plan,
                "refinement_rounds": refinement_rounds,
                "forced": True,
            })
        return plan


def _normalize_plan(parsed: Dict[str, Any], task: str,
                    fallback_summary: Optional[str] = None) -> Dict[str, Any]:
    """Ensure each step has id/description/success_criterion and the plan has a summary."""
    for i, step in enumerate(parsed["steps"], 1):
        step.setdefault("id", i)
        step.setdefault("description", "")
        step.setdefault("success_criterion", "Step is complete")
    parsed.setdefault("summary", fallback_summary or task)
    return parsed
