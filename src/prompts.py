"""Prompt templates used by the agent's LLM calls.

The planning prompts use `str.format()` placeholders, so JSON example braces
in the templates are escaped as `{{` / `}}`.

`system_prefix()` returns a small dynamic block (today's date) prepended to
every system message — this prevents the model from fabricating future
content or claiming ignorance of the present moment.
"""

from datetime import datetime


def system_prefix() -> str:
    """Common system-message prefix injected before each LLM call.

    Date-grounds the model so it doesn't fabricate future events or claim
    its training cutoff is "now". Re-evaluated each call so long-running
    sessions stay accurate.
    """
    return (
        f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
        "Only claim facts about events on or before this date. If you do not "
        "know whether something happened, say so explicitly rather than "
        "guessing or fabricating dates / IDs / names.\n\n"
    )


TOOLS_PROMPT = """
You have access to these tools. To use a tool, respond with exactly this JSON format:
{"tool": "tool_name", "parameters": {"param1": "value1"}}

Available tools:
1. execute_shell: Run shell commands. Parameters: {"command": "string"}
2. read_file: Read file contents. Parameters: {"path": "string"}
3. write_file: Write to a file (overwrites). Parameters: {"path": "string", "content": "string"}
4. list_directory: List directory contents. Parameters: {"path": "string"} (default: ".")
5. search_in_files: Recursively grep a regex pattern across files. Parameters: {"pattern": "string", "path": "string (default '.')", "file_glob": "string (default '*')"}
6. edit_file: Replace exactly one occurrence of old_string with new_string in a file. Fails if old_string is missing or appears more than once — give enough surrounding context to make it unique. Parameters: {"path": "string", "old_string": "string", "new_string": "string"}
7. run_python: Run Python code in a subprocess (no shell-quoting). Provide one of: {"code": "string"} OR {"script_path": "string"}. Captures stdout; appends stderr if present.
8. browser_visit: HTTP-fetch a URL and return the main content as Markdown (boilerplate stripped). Parameters: {"url": "http(s) URL", "max_chars": "int (default 4000)", "offset": "int (default 0)"}. Re-visits cached. If a page is long, the response includes the next offset to read.
9. web_search: Search the web via DuckDuckGo. Parameters: {"query": "string", "max_results": "int (default 5)"}. Returns ranked {title, url, snippet}. Pair with browser_visit on the most relevant URL.

Choosing between tools:
- For *small* edits to existing files, prefer edit_file over write_file (avoids token-heavy rewrites and the truncation risk).
- For finding code/text in the project, prefer search_in_files over execute_shell + grep.
- For running a Python script you just wrote, prefer run_python over execute_shell (avoids quoting bugs).
- For looking up info/docs on the web, prefer web_search to discover URLs, then browser_visit on the best match. Don't use execute_shell with curl.

When you have completed the task, respond with:
{"complete": true, "answer": "your final answer"}
"""


INITIAL_PLAN_PROMPT = """You are a senior planning agent. Decompose the user's request into a concrete, ordered, executable plan. Anticipate implicit requirements: if the user says "build a model", they almost always also want training, real data, and a README. If the user says "make a script", they usually also want example usage.

Constraints:
- Maximum 8 steps. If trivial, 1 step is fine.
- Each step must be independently verifiable (give a concrete success_criterion).
- Order steps by dependency.

The execution agent has these tools available — plan in terms of them:
- execute_shell: run any shell command
- read_file / write_file / edit_file: file I/O. Prefer edit_file for small changes.
- list_directory / search_in_files: discover files and find code/text by regex
- run_python: run Python code or a script (no shell-quoting required)
- web_search: search the web (DuckDuckGo)
- browser_visit: HTTP-fetch a URL and read its main content as Markdown

Rules of thumb:
- For "latest" / "current" / "news" tasks → web_search first, then browser_visit on the most relevant URL. Do not invent URLs; only visit URLs returned by web_search or supplied by the user.
- For research tasks, cap source visits to 5–7 and require citations.
- Do not say "browsing is not permitted" — these tools exist; use them.

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
