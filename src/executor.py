"""Executor: runs the Think → Act → Observe loop, optionally per plan step."""

from typing import Any, Dict, List, Optional

from colors import C
from parsing import hash_action, parse_response
from prompts import TOOLS_PROMPT
from state import AgentState, ToolResult
from tools import Tools


class Executor:
    """Runs the inner T-A-O loop and dispatches tool calls.

    `execute_plan` runs one T-A-O loop per plan step (with `current_step`
    injected into the prompt). `legacy_execute` runs a single loop without
    a plan. Both reuse `run_tao_loop`.
    """

    def __init__(self, client, model: str, tools: Tools,
                 max_iterations: int, max_tokens_tao: int,
                 logger: Optional[Any] = None):
        self.client = client
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_tokens_tao = max_tokens_tao
        self.logger = logger

    # ---- Entry points --------------------------------------------------------

    def execute_plan(self, state: AgentState, task: str, plan: Dict[str, Any]) -> str:
        """Run T-A-O once per plan step, accumulating results."""
        state.task = task
        state.plan = plan
        step_results: List[str] = []
        steps = plan.get("steps", [])
        for step in steps:
            state.current_step = step
            print(C.step(f"\n{'━' * 60}"))
            print(C.step(f"▶️  STEP {step['id']}/{len(steps)}:") + f" {step.get('description', '')}")
            print(f"   {C.dim('Success:')} {C.dim(step.get('success_criterion', ''))}")
            print(C.step(f"{'━' * 60}"))
            if self.logger:
                self.logger.log("step_start", {
                    "step_id": step.get("id"),
                    "description": step.get("description", ""),
                    "success_criterion": step.get("success_criterion", ""),
                })
            step_result = self.run_tao_loop(state, self.max_iterations,
                                            step_id=step.get("id"))
            step_results.append(f"[Step {step['id']}] {step_result}")
            if self.logger:
                self.logger.log("step_complete", {
                    "step_id": step.get("id"),
                    "iterations_used": state.iteration_count,
                    "completed": state.is_complete,
                })
        state.current_step = None
        return "\n\n".join(step_results)

    def legacy_execute(self, state: AgentState, task: str) -> str:
        """No-plan path: a single T-A-O loop."""
        state.task = task
        print(C.phase(f"\n🤖 Starting agent for task:") + f" {task}")
        print(C.dim(f"Max iterations: {self.max_iterations}\n"))
        return self.run_tao_loop(state, self.max_iterations)

    # ---- The Think-Act-Observe loop -----------------------------------------

    def run_tao_loop(self, state: AgentState, max_iters: int,
                     step_id: Optional[Any] = None) -> str:
        """Run T-A-O on `state` until completion or `max_iters`.

        Resets per-step counters and `is_complete`; preserves history so
        later plan steps see what earlier ones did. `step_id` is attached
        to logged events so per-step events can be grouped.
        """
        state.iteration_count = 0
        state.is_complete = False
        last_action_hash: Optional[str] = None
        repeat_count = 0

        while not state.is_complete and state.iteration_count < max_iters:
            state.iteration_count += 1

            print(C.phase(f"\n--- Iteration {state.iteration_count} ---"))
            print(C.label("THINKING..."))
            if self.logger:
                self.logger.log("iteration", {
                    "step_id": step_id,
                    "iteration": state.iteration_count,
                })

            prompt = self._build_prompt(state)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an autonomous agent. Use tools to complete tasks."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=self.max_tokens_tao,
            )

            llm_response = (response.choices[0].message.content or "I need to think more about this task.").strip()
            print(f"{C.dim('LLM Response:')} {C.dim(llm_response[:200])}...")

            parsed = parse_response(llm_response)

            looks_like_tool_call = '"tool"' in llm_response and '"parameters"' in llm_response
            parser_only_got_thought = set(parsed.keys()) == {"thought"}
            if looks_like_tool_call and parser_only_got_thought:
                print(C.warn("⚠️  Tool-call JSON appears truncated; nudging LLM to retry differently."))
                if self.logger:
                    self.logger.log("parse_error", {
                        "step_id": step_id,
                        "iteration": state.iteration_count,
                        "response_preview": llm_response[:200],
                    })
                state.thought_history.append(llm_response[:200])
                state.action_history.append({"parse_error": True})
                state.observation_history.append(
                    "ERROR: Your previous response was truncated mid-JSON, likely because "
                    "the 'content' field was too large. For big files, use execute_shell "
                    "with a heredoc (e.g. cat > path << 'EOF' ... EOF), or split the write "
                    "into multiple smaller write_file calls."
                )
                continue

            if parsed.get("complete"):
                state.is_complete = True
                state.final_answer = parsed.get("answer", llm_response)
                print(C.ok("✅ Step marked as complete"))
                break

            thought = parsed.get("thought", llm_response)
            state.thought_history.append(thought)
            print(f"{C.dim('Thought:')} {C.dim(thought[:100])}...")

            if "tool" in parsed and "parameters" in parsed:
                tool_name = parsed["tool"]
                parameters = parsed["parameters"]

                current_hash = hash_action(tool_name, parameters)
                if current_hash == last_action_hash:
                    repeat_count += 1
                    print(C.warn(f"⚠️  Detected repeated action (count: {repeat_count})"))
                    if self.logger:
                        self.logger.log("repeated_action", {
                            "step_id": step_id, "tool": tool_name,
                            "repeat_count": repeat_count,
                        })
                    if repeat_count >= 3:
                        print(C.err("🔄 Breaking loop - forcing progress..."))
                        if self.logger:
                            self.logger.log("loop_break", {
                                "step_id": step_id, "tool": tool_name,
                            })
                        state.observation_history.append(
                            f"NOTE: You've repeated the action '{tool_name}' multiple times. "
                            "Please try a different approach or mark the step as complete."
                        )
                        repeat_count = 0
                        continue
                else:
                    repeat_count = 0
                    last_action_hash = current_hash

                print(f"{C.label('ACT:')} Executing tool {C.BR_CYAN}'{tool_name}'{C.RESET}...")
                if self.logger:
                    self.logger.log("tool_call", {
                        "step_id": step_id, "tool": tool_name,
                        "parameters_preview": str(parameters)[:200],
                    })
                state.action_history.append({"tool": tool_name, "params": parameters})
                result = self._execute_tool(tool_name, parameters)
                observation = result.output if result.success else f"ERROR: {result.error}"
                state.observation_history.append(observation)
                print(f"{C.label('OBSERVE:')} {C.dim(observation[:200])}...")
                if self.logger:
                    self.logger.log("tool_result", {
                        "step_id": step_id, "tool": tool_name,
                        "success": result.success,
                        "output_preview": observation[:200],
                    })
            else:
                state.action_history.append({"thought_only": True})
                state.observation_history.append("Continuing...")

        if state.is_complete:
            return state.final_answer or ""
        tail = "\n".join(state.observation_history[-3:])
        return f"Iteration cap ({max_iters}) reached. Best attempt:\n{tail}"

    # ---- Helpers -------------------------------------------------------------

    def _build_prompt(self, state: AgentState) -> str:
        prompt = (
            f"Task: {state.task}\n\n"
            "You are an autonomous agent. Complete the task using a Think → Act → Observe loop.\n\n"
            "THINK: Consider what you know and what you need to do next.\n"
            "ACT: Use one of your tools to take action.\n"
            "OBSERVE: See the result of your action.\n"
            "REPEAT: Continue until the task is complete.\n\n"
        )

        if state.current_step:
            step = state.current_step
            prompt += (
                "\n--- Current Step ---\n"
                f"Step {step.get('id', '?')}: {step.get('description', '')}\n"
                f"Success criterion: {step.get('success_criterion', '')}\n"
                "Focus on completing THIS step. Mark complete when the success "
                "criterion is met, then the next step will begin.\n"
            )

        if state.thought_history:
            prompt += "\n--- History ---\n"
            for i, (thought, action, observation) in enumerate(zip(
                state.thought_history,
                state.action_history,
                state.observation_history,
            )):
                prompt += f"\nIteration {i + 1}:\n"
                prompt += f"Thought: {thought}\n"
                prompt += f"Action: {action}\n"
                prompt += f"Observation: {observation}\n"

        prompt += "\n--- Current Turn ---\n"
        prompt += f"Iteration {state.iteration_count + 1}:\n"
        prompt += "\nWhat do you think needs to be done next?\n"
        prompt += TOOLS_PROMPT
        return prompt

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        if tool_name == "execute_shell":
            return self.tools.execute_shell(parameters.get("command", ""))
        if tool_name == "read_file":
            return self.tools.read_file(parameters.get("path", ""))
        if tool_name == "write_file":
            return self.tools.write_file(
                parameters.get("path", ""),
                parameters.get("content", ""),
            )
        if tool_name == "list_directory":
            return self.tools.list_directory(parameters.get("path", "."))
        return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")
