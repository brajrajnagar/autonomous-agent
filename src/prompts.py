"""Prompt templates used by the agent's LLM calls.

The planning prompts use `str.format()` placeholders, so JSON example braces
in the templates are escaped as `{{` / `}}`.
"""


TOOLS_PROMPT = """
You have access to these tools. To use a tool, respond with exactly this JSON format:
{"tool": "tool_name", "parameters": {"param1": "value1"}}

Available tools:
1. execute_shell: Run shell commands. Parameters: {"command": "string"}
2. read_file: Read file contents. Parameters: {"path": "string"}
3. write_file: Write to a file. Parameters: {"path": "string", "content": "string"}
4. list_directory: List directory contents. Parameters: {"path": "string"} (default: ".")

When you have completed the task, respond with:
{"complete": true, "answer": "your final answer"}
"""


INITIAL_PLAN_PROMPT = """You are a senior planning agent. Decompose the user's request into a concrete, ordered, executable plan. Anticipate implicit requirements: if the user says "build a model", they almost always also want training, real data, and a README. If the user says "make a script", they usually also want example usage.

Constraints:
- Maximum 8 steps. If trivial, 1 step is fine.
- Each step must be independently verifiable (give a concrete success_criterion).
- Order steps by dependency.
- Use the agent's tools (execute_shell, read_file, write_file, list_directory) — no external services.

Respond with strict JSON only (no prose, no markdown fences):
{{"summary": "...", "steps": [{{"id": 1, "description": "...", "success_criterion": "..."}}, ...]}}

USER REQUEST: {task}"""


PLAN_CRITIQUE_PROMPT = """You are a critical planning reviewer. Review this plan and identify SPECIFIC, ACTIONABLE improvements. Look for:
- Missing steps the user probably wants but didn't say (data, training, docs, tests, examples).
- Vague descriptions or unverifiable success_criteria.
- Wrong ordering / missing dependencies.
- Steps that could be merged or split.

Respond with strict JSON only (no prose, no markdown fences):
{{"suggestions": [{{"issue": "...", "fix": "..."}}, ...]}}
Empty suggestions list is fine if the plan is excellent.

USER REQUEST: {task}
PLAN: {plan_json}"""


PLAN_REFINE_PROMPT = """Update the plan based on user feedback. The user may reference suggestions by number (e.g., "apply 1 and 3"), describe changes in free form, or both.

Respond with strict JSON only (no prose, no markdown fences) in the same plan format:
{{"summary": "...", "steps": [{{"id": N, "description": "...", "success_criterion": "..."}}, ...]}}
Preserve step ids where the step is unchanged; assign new ids for new steps; renumber as needed.

USER REQUEST: {task}
CURRENT PLAN: {plan_json}
CRITIC SUGGESTIONS: {suggestions_json}
USER FEEDBACK: {user_text}"""


CRITIC_REVIEW_PROMPT = """
You are a critic/reviewer. Review the following task completion:

ORIGINAL TASK: {task}

PROPOSED ANSWER:
{result}

ACTION HISTORY:
{action_history}

Please review this output:
1. Is the task actually complete?
2. Are there any errors or issues?
3. Is the answer clear and accurate?
4. Any improvements needed?

Respond with either:
- "APPROVED" if the output is satisfactory
- Or describe what needs to be fixed
"""
