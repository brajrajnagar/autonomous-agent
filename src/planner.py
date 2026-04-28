"""Planner: produces, critiques, and refines an executable plan with the user."""

import json
from typing import Any, Dict, List, Optional

from colors import C
from parsing import safe_json_parse
from prompts import (
    INITIAL_PLAN_PROMPT, PLAN_CRITIQUE_PROMPT, PLAN_REFINE_PROMPT,
    system_prefix,
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
                  temperature: float, max_tokens: int) -> str:
        """Single-shot LLM call returning the raw text response."""
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

    def initial_plan(self, task: str) -> Dict[str, Any]:
        prompt = INITIAL_PLAN_PROMPT.format(task=task)
        raw = self._llm_call(
            "You are a senior planning agent. Output strict JSON only.",
            prompt, temperature=0.5, max_tokens=self.max_tokens_initial,
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
        )
        parsed = safe_json_parse(raw)
        if not parsed or "suggestions" not in parsed:
            return []
        return [s for s in parsed["suggestions"]
                if isinstance(s, dict) and "issue" in s and "fix" in s]

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
        )
        parsed = safe_json_parse(raw)
        if not parsed or "steps" not in parsed or not parsed["steps"]:
            print(C.warn("⚠️  Plan refinement failed; keeping previous plan."))
            return plan
        return _normalize_plan(parsed, task, fallback_summary=plan.get("summary"))

    # ---- User interaction ----------------------------------------------------

    def present_plan_to_user(self, plan: Dict[str, Any],
                             suggestions: List[Dict[str, str]]) -> str:
        """Print the plan + suggestions and read one line from stdin.

        Returns the literal "APPROVE" sentinel for empty/`go`/`ok`/`yes`,
        a normalized "Apply suggestion(s) ..." string for `apply N`, or
        the raw user input otherwise.
        """
        print("\n" + C.header("═" * 60))
        print(C.header("📋 PROPOSED PLAN"))
        print(C.header("═" * 60))
        print(f"{C.BOLD}Summary:{C.RESET} {plan.get('summary', '')}\n")
        print(f"{C.BOLD}Steps:{C.RESET}")
        for step in plan.get("steps", []):
            print(f"  {C.step(str(step['id']) + '.')} {step['description']}")
            crit = step.get("success_criterion", "")
            if crit:
                print(C.dim(f"     ↳ success: {crit}"))
        if suggestions:
            print(C.hint("\n💡 Suggested improvements:"))
            for i, sug in enumerate(suggestions, 1):
                print(f"  {C.hint(f'[{i}]')} {sug.get('issue', '')}")
                print(f"      {C.dim('→')} {sug.get('fix', '')}")
        else:
            print(C.ok("\n💡 No suggestions — plan looks solid."))
        print(C.dim(
            "\nType 'go' / 'ok' / Enter to approve and execute,\n"
            "     'apply 1' or 'apply 1,2' to adopt suggestions,\n"
            "     or describe changes in your own words."
        ))
        try:
            user_input = input(f"{C.BR_GREEN}> {C.RESET}").strip()
        except EOFError:
            return "APPROVE"
        if user_input == "" or user_input.lower() in ("go", "ok", "yes", "y", "approve"):
            return "APPROVE"
        if user_input.lower().startswith("apply"):
            indices = user_input[len("apply"):].strip()
            return f"Apply suggestion(s) {indices} from the critic list."
        return user_input

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
