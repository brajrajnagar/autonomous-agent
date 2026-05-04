"""Executor: runs the Think → Act → Observe loop using OpenAI tool calling.

The model emits structured `tool_calls` (validated server-side) instead of
text-embedded JSON, and we accumulate a multi-turn message history across
iterations and plan steps. This eliminates the truncation / parser-fallthrough
failure mode of the old text-JSON pattern.

Flow:
  [system: agent instructions + (optional) current step]
  [user:   task, plus any earlier-step result summaries]
  [assistant: tool_calls=[...]]      ← model
  [tool:   tool_call_id, content=<observation>]
  ...
  [assistant: tool_calls=[complete_task(answer=...)]]   ← terminates the step

`execute_plan` keeps a single `_messages` list across all step calls so step N
sees what step N-1 did. `legacy_execute` builds a fresh list. `run_tao_loop`
operates on whatever list is currently held.
"""

import json
from typing import Any, Dict, List, Optional

import ui
from colors import C
from parsing import hash_action, parse_xml_tool_calls
from prompts import system_prefix
from state import AgentState, ToolResult
from tool_schemas import TOOL_DEFINITIONS, TOOL_NAMES
from tools import Tools


class Executor:
    """Runs the inner T-A-O loop and dispatches tool calls.

    `execute_plan` runs one T-A-O loop per plan step, sharing one messages
    list across steps so later steps see earlier observations.
    `legacy_execute` runs a single loop without a plan.
    """

    def __init__(self, client, model: str, tools: Tools,
                 max_iterations: int, max_tokens_tao: int,
                 logger: Optional[Any] = None,
                 feedback: Optional[Any] = None,
                 context_manager: Optional[Any] = None,
                 planner: Optional[Any] = None,
                 replanning_enabled: bool = True,
                 max_replans_per_step: int = 2,
                 max_replans_per_run: int = 5,
                 autonomy: str = "auto"):
        self.client = client
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_tokens_tao = max_tokens_tao
        self.logger = logger
        self.feedback = feedback
        self.context = context_manager
        self.planner = planner
        self.replanning_enabled = replanning_enabled
        self.max_replans_per_step = max_replans_per_step
        self.max_replans_per_run = max_replans_per_run
        self.autonomy = autonomy
        # Multi-turn message accumulator. Reset by execute_plan/legacy_execute
        # at the start of each top-level invocation; mutated by run_tao_loop.
        self._messages: List[Dict[str, Any]] = []

    # ---- Entry points --------------------------------------------------------

    def execute_plan(self, state: AgentState, task: str, plan: Dict[str, Any]) -> str:
        """Run T-A-O once per plan step. On step failure, optionally replan
        (retry / revise_step / revise_plan / skip / abort) before continuing.
        """
        state.task = task
        state.plan = plan
        self._messages = self._initial_messages(task)

        step_results: List[str] = []
        steps: List[Dict[str, Any]] = list(plan.get("steps", []))
        cursor = 0
        replans_this_run = 0

        while cursor < len(steps):
            step = steps[cursor]
            replans_this_step = 0
            attempt = 0

            while True:
                attempt += 1
                state.current_step = step

                # Header + step_start log only on the first attempt; later
                # attempts print a clear "retry" / "revised" banner instead.
                if attempt == 1:
                    ui.render_step_header(step, cursor + 1, len(steps))
                    if self.logger:
                        self.logger.log("step_start", {
                            "step_id": step.get("id"),
                            "description": step.get("description", ""),
                            "success_criterion": step.get("success_criterion", ""),
                        })
                    self._messages.append({
                        "role": "user",
                        "content": (
                            f"Now do Step {step.get('id')}: {step.get('description', '')}\n"
                            f"Success criterion: {step.get('success_criterion', '')}\n"
                            f"Call complete_task when this step is done."
                        ),
                    })

                step_result = self.run_tao_loop(state, self.max_iterations, step_id=step.get("id"))

                if self.logger:
                    self.logger.log("step_complete", {
                        "step_id": step.get("id"),
                        "iterations_used": state.iteration_count,
                        "completed": state.is_complete,
                        "attempt": attempt,
                    })

                # Success path.
                if state.is_complete:
                    step_results.append(f"[Step {step['id']}] {step_result}")
                    if self.context:
                        self._messages = self.context.compact_step_boundary(
                            self._messages, state, step, step_result,
                        )
                    cursor += 1
                    break

                # Failure path: hit the iteration cap. Decide whether to replan.
                failure_reason = (
                    f"iteration cap ({self.max_iterations}) reached without complete_task"
                )
                budget_exhausted = (
                    not self.replanning_enabled
                    or self.planner is None
                    or replans_this_step >= self.max_replans_per_step
                    or replans_this_run >= self.max_replans_per_run
                )
                if budget_exhausted:
                    print(C.err(
                        f"❌ Step {step['id']} did not complete and replan budget is exhausted; "
                        "treating step as failed."
                    ))
                    if self.logger:
                        self.logger.log("step_failed", {
                            "step_id": step.get("id"),
                            "reason": failure_reason,
                            "replans_used": replans_this_step,
                        })
                    step_results.append(f"[Step {step['id']} FAILED] {step_result}")
                    cursor += 1
                    break

                # Trigger replan.
                replans_this_step += 1
                replans_this_run += 1
                print(C.warn(
                    f"\n🔁 Step {step['id']} did not complete. Asking planner to replan "
                    f"(attempt {replans_this_step}/{self.max_replans_per_step})..."
                ))
                if self.logger:
                    self.logger.log("replan_triggered", {
                        "step_id": step.get("id"),
                        "reason": failure_reason,
                        "replan_count": replans_this_step,
                    })

                decision = self.planner.replan(
                    task, plan, step, failure_reason,
                    state.observation_history[-5:],
                )
                action = decision.get("action", "abort")
                reasoning = decision.get("reasoning", "")
                print(C.phase(f"   planner chose: {action}") + C.dim(f" — {reasoning}"))

                # Interactive override.
                if self.autonomy == "interactive":
                    action = self._prompt_user_for_replan(decision)

                if self.logger:
                    self.logger.log("replan_decided", {
                        "step_id": step.get("id"),
                        "action": action,
                        "reasoning": reasoning,
                    })

                # Apply the action.
                if action == "retry":
                    self._messages.append({
                        "role": "user",
                        "content": (
                            f"[Replan] Step {step['id']} did not complete in "
                            f"{self.max_iterations} iterations. Retry the same step "
                            f"with a fresh budget."
                        ),
                    })
                    continue  # inner retry loop

                if action == "revise_step":
                    revised = decision.get("revised_step") or step
                    revised.setdefault("id", step["id"])
                    steps[cursor] = revised
                    step = revised
                    self._messages.append({
                        "role": "user",
                        "content": (
                            f"[Replan] Step {step['id']} has been revised:\n"
                            f"  description: {revised.get('description', '')}\n"
                            f"  success_criterion: {revised.get('success_criterion', '')}\n"
                            f"Continue with the revised step."
                        ),
                    })
                    continue

                if action == "revise_plan":
                    revised_steps = decision.get("revised_steps") or [step]
                    steps = steps[:cursor] + list(revised_steps)
                    state.plan = {"summary": plan.get("summary", ""), "steps": steps}
                    step = steps[cursor]
                    self._messages.append({
                        "role": "user",
                        "content": (
                            f"[Replan] The plan from Step {step.get('id')} onward has been "
                            f"revised to {len(revised_steps)} new step(s). "
                            f"Continuing with the new plan."
                        ),
                    })
                    if self.logger:
                        self.logger.log("plan_revised", {
                            "from_step": step.get("id"),
                            "new_step_count": len(revised_steps),
                        })
                    continue

                if action == "skip":
                    print(C.warn(f"⏭  Skipping step {step['id']} per replan decision."))
                    if self.logger:
                        self.logger.log("step_skipped", {"step_id": step.get("id")})
                    step_results.append(f"[Step {step['id']} SKIPPED] {reasoning}")
                    cursor += 1
                    break

                if action == "abort":
                    print(C.err(f"🛑 Aborting run per replan decision."))
                    if self.logger:
                        self.logger.log("run_aborted", {
                            "step_id": step.get("id"),
                            "reasoning": reasoning,
                        })
                    step_results.append(f"[Step {step['id']} ABORTED] {reasoning}")
                    state.current_step = None
                    return "\n\n".join(step_results)

                # Unknown action — defensive: treat as abort.
                step_results.append(f"[Step {step['id']} ABORTED] unknown action: {action}")
                state.current_step = None
                return "\n\n".join(step_results)

        state.current_step = None
        return "\n\n".join(step_results)

    def _prompt_user_for_replan(self, decision: Dict[str, Any]) -> str:
        """Let the user accept or override the planner's recovery action.

        Returns one of: "retry", "revise_step", "revise_plan", "skip", "abort".
        Used only when autonomy='interactive'. On non-TTY, falls back to a
        numbered text prompt; an unrecognized choice defers to the planner.
        """
        action = decision.get("action", "abort")
        reasoning = decision.get("reasoning", "")
        try:
            return ui.choose(
                f"Planner suggests '{action}' — {reasoning}\nWhat should we do?",
                [
                    (f"Accept planner's choice: {action}", action),
                    ("Retry the same step", "retry"),
                    ("Skip this step", "skip"),
                    ("Abort the run", "abort"),
                ],
                default=action,
            )
        except KeyboardInterrupt:
            return "abort"

    def legacy_execute(self, state: AgentState, task: str) -> str:
        """No-plan path: a single T-A-O loop with a fresh messages list."""
        state.task = task
        print(C.phase(f"\n🤖 Starting agent for task:") + f" {task}")
        print(C.dim(f"Max iterations: {self.max_iterations}\n"))
        self._messages = self._initial_messages(task)
        return self.run_tao_loop(state, self.max_iterations)

    # ---- The Think-Act-Observe loop -----------------------------------------

    def run_tao_loop(self, state: AgentState, max_iters: int,
                     step_id: Optional[Any] = None) -> str:
        """Run T-A-O on `state` and `self._messages` until completion or cap.

        Resets per-step counters and `is_complete`. Mutates `self._messages`
        in place so cross-step continuity is preserved by callers that want it.
        Self-initializes the messages list if a caller (e.g. the 'simple'
        autonomy path) skipped `execute_plan` / `legacy_execute`.
        """
        if not self._messages:
            self._messages = self._initial_messages(state.task)

        state.iteration_count = 0
        state.is_complete = False
        last_action_hash: Optional[str] = None
        repeat_count = 0

        while not state.is_complete and state.iteration_count < max_iters:
            state.iteration_count += 1

            print(C.phase(f"\n--- Iteration {state.iteration_count} ---"))
            if self.logger:
                self.logger.log("iteration", {
                    "step_id": step_id, "iteration": state.iteration_count,
                })

            # Context compression: keeps the prompt small without dropping
            # essential anchors. Returns the input unchanged when below budget.
            if self.context:
                self._messages = self.context.maybe_compress(self._messages, state)

            with ui.thinking(f"Iteration {state.iteration_count} thinking"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=self.max_tokens_tao,
                )
            msg = response.choices[0].message

            # Capture model's natural-language explanation (if any) as the "thought".
            content_text = (msg.content or "").strip()
            if content_text:
                state.thought_history.append(content_text)
                print(f"{C.dim('Thought:')} {C.dim(content_text[:200])}...")

            # Normalize tool calls from either OpenAI's structured field or
            # Qwen's <tool_call> XML inside content. Each entry: id, name,
            # arguments, and "format" (used to choose the right message shape
            # when feeding the result back).
            normalized: List[Dict[str, Any]] = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        if self.logger:
                            self.logger.log("parse_error", {
                                "step_id": step_id, "iteration": state.iteration_count,
                                "tool": tc.function.name,
                                "raw": (tc.function.arguments or "")[:200],
                                "error": str(e),
                            })
                        args = {}
                    normalized.append({
                        "id": tc.id, "name": tc.function.name,
                        "arguments": args, "format": "openai",
                    })
            elif content_text:
                # Provider didn't translate `tools=` to structured tool_calls;
                # try to pick up Qwen-style <tool_call> blocks from the text.
                for i, qc in enumerate(parse_xml_tool_calls(content_text)):
                    normalized.append({
                        "id": f"xml_{state.iteration_count}_{i}",
                        "name": qc["name"], "arguments": qc["arguments"],
                        "format": "xml",
                    })

            # Append the assistant turn with the right message shape per format.
            if normalized and normalized[0]["format"] == "openai":
                self._messages.append({
                    "role": "assistant",
                    "content": content_text or None,
                    "tool_calls": [
                        {
                            "id": tc.id, "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (msg.tool_calls or [])
                    ],
                })
            else:
                # XML fallback OR no tool calls at all — just text.
                self._messages.append({
                    "role": "assistant", "content": content_text or "",
                })

            if not normalized:
                # No tool intent → implicit completion.
                state.is_complete = True
                state.final_answer = content_text or "(no response)"
                print(C.ok("✅ Step marked as complete (no further tool calls)"))
                break

            # Process each requested tool call sequentially.
            terminated = False
            for tc in normalized:
                tool_name = tc["name"]
                parameters = tc["arguments"] or {}
                tc_id = tc["id"]
                fmt = tc["format"]

                # Synthetic completion sentinel — terminate the step.
                if tool_name == "complete_task":
                    state.is_complete = True
                    state.final_answer = parameters.get("answer", "")
                    print(C.ok("✅ Step marked as complete"))
                    self._append_tool_result(fmt, tc_id, tool_name, "(step terminated)")
                    terminated = True
                    break

                # Unknown tool — feed the error back so the model can recover.
                if tool_name not in TOOL_NAMES:
                    err = f"Unknown tool: {tool_name}"
                    self._append_tool_result(fmt, tc_id, tool_name, f"ERROR: {err}")
                    continue

                # Repeated-action loop detection.
                current_hash = hash_action(tool_name, parameters)
                if current_hash == last_action_hash:
                    repeat_count += 1
                    print(C.warn(f"⚠️  Detected repeated action (count: {repeat_count})"))
                    if self.logger:
                        self.logger.log("repeated_action", {
                            "step_id": step_id, "tool": tool_name, "repeat_count": repeat_count,
                        })
                    if repeat_count >= 3:
                        print(C.err("🔄 Breaking loop - forcing progress..."))
                        if self.logger:
                            self.logger.log("loop_break", {
                                "step_id": step_id, "tool": tool_name,
                            })
                        nudge = (
                            f"NOTE: You've called '{tool_name}' with the same arguments "
                            "multiple times. Try a different approach or call complete_task."
                        )
                        self._append_tool_result(fmt, tc_id, tool_name, nudge)
                        repeat_count = 0
                        continue
                else:
                    repeat_count = 0
                    last_action_hash = current_hash

                # Dispatch.
                print(f"{C.label('ACT:')} Executing tool {C.BR_CYAN}'{tool_name}'{C.RESET}...")
                if self.logger:
                    self.logger.log("tool_call", {
                        "step_id": step_id, "tool": tool_name,
                        "parameters": parameters, "format": fmt,
                    })
                state.action_history.append({"tool": tool_name, "params": parameters})

                result = self._execute_tool(tool_name, parameters)
                observation = result.output if result.success else f"ERROR: {result.error}"

                # Post-tool deterministic verification (FeedbackEngine).
                if self.feedback:
                    extra = self.feedback.run_after(tool_name, parameters, result)
                    if extra:
                        observation = f"{observation}\n{extra}" if observation else extra

                state.observation_history.append(observation)
                print(f"{C.label('OBSERVE:')} {C.dim(observation[:200])}...")
                if self.logger:
                    self.logger.log("tool_result", {
                        "step_id": step_id, "tool": tool_name,
                        "success": result.success, "output_preview": observation[:200],
                    })

                self._append_tool_result(fmt, tc_id, tool_name, observation)

            if terminated:
                break

        if state.is_complete:
            return state.final_answer or ""
        tail = "\n".join(state.observation_history[-3:])
        return f"Iteration cap ({max_iters}) reached. Best attempt:\n{tail}"

    # ---- Helpers -------------------------------------------------------------

    def _append_tool_result(self, fmt: str, tc_id: str, tool_name: str, observation: str) -> None:
        """Feed a tool result back into the messages history, using the right
        message shape for the call's format.

        - "openai" tool_calls require a `role: tool` message with `tool_call_id`
          matching the assistant's tool_call.
        - "xml" (Qwen) tool calls came from text content and don't have a
          server-tracked id; we use a `role: user` message instead so the
          provider doesn't reject an orphan `tool_call_id`.
        """
        if fmt == "openai":
            self._messages.append({
                "role": "tool", "tool_call_id": tc_id, "content": observation,
            })
        else:
            self._messages.append({
                "role": "user",
                "content": f"<tool_response>\n{observation}\n</tool_response>",
            })

    def _initial_messages(self, task: str) -> List[Dict[str, Any]]:
        """Build the starting messages for a new top-level invocation."""
        system = (
            system_prefix()
            + "You are an autonomous agent. Use the provided tools to complete the task. "
            + "Think briefly before each tool call when useful, then call exactly one tool. "
            + "When the task or current step is finished, call the `complete_task` tool with "
            + "your final answer or a summary of what you did."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: {task}"},
        ]

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        if tool_name == "execute_shell":
            return self.tools.execute_shell(parameters.get("command", ""))
        if tool_name == "read_file":
            return self.tools.read_file(
                parameters.get("path", ""),
                mode=parameters.get("mode", "head"),
                offset=int(parameters.get("offset", 0)),
                length=int(parameters.get("length", 6000)),
            )
        if tool_name == "write_file":
            return self.tools.write_file(
                parameters.get("path", ""),
                parameters.get("content", ""),
            )
        if tool_name == "list_directory":
            return self.tools.list_directory(parameters.get("path", "."))
        if tool_name == "search_in_files":
            return self.tools.search_in_files(
                parameters.get("pattern", ""),
                parameters.get("path", "."),
                parameters.get("file_glob", "*"),
                int(parameters.get("max_matches", 100)),
            )
        if tool_name == "edit_file":
            return self.tools.edit_file(
                parameters.get("path", ""),
                parameters.get("old_string", ""),
                parameters.get("new_string", ""),
            )
        if tool_name == "run_python":
            return self.tools.run_python(
                code=parameters.get("code", ""),
                script_path=parameters.get("script_path", ""),
                timeout=int(parameters.get("timeout", 120)),
            )
        if tool_name == "browser_visit":
            return self.tools.browser_visit(
                parameters.get("url", ""),
                int(parameters.get("max_chars", 4000)),
                int(parameters.get("offset", 0)),
            )
        if tool_name == "web_search":
            return self.tools.web_search(
                parameters.get("query", ""),
                int(parameters.get("max_results", 5)),
            )
        return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")
