"""OpenAI tool/function schemas for the inner Think-Act-Observe loop.

The Executor passes `TOOL_DEFINITIONS` to `chat.completions.create(tools=...)`
so the model emits structured `tool_calls` instead of text-embedded JSON.
This eliminates a whole class of fragility (truncation mid-JSON, prose
mixed with the tool object, parser fall-through to a "thought") because
tool calls are validated server-side.

Each schema mirrors a method on `Tools` plus a synthetic `complete_task`
sentinel that replaces the old `{"complete": true, "answer": "..."}` text
convention with an explicit, schema-validated completion signal.
"""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell",
            "description": (
                "Run a shell command and capture stdout/stderr. Use for one-off "
                "commands. For Python execution prefer `run_python`; for grepping "
                "files prefer `search_in_files`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (e.g. 'ls -la').",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file with paginated/clipped output. Use mode='head' "
                "for the start of a file (default, safe for code), "
                "mode='tail' for the end of a log, or mode='slice' with "
                "offset+length to paginate large files. The result tells "
                "you the next offset when there's more to read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "mode": {
                        "type": "string",
                        "description": "'head' | 'tail' | 'slice' (default 'head').",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Start position when mode='slice' (default 0).",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Max chars to read (default 6000).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with the given content. For *small* "
                "edits to an existing file prefer `edit_file` to avoid rewriting "
                "the whole file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List entries in a directory with [DIR] / [FILE] prefixes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path. Defaults to '.'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": (
                "Recursively grep a regex pattern across files matching `file_glob`. "
                "Skips noise dirs (venv, __pycache__, .git, node_modules, logs). "
                "Caps results at `max_matches`. Use this instead of `execute_shell + grep`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {"type": "string", "description": "Directory to search (default '.')."},
                    "file_glob": {"type": "string", "description": "Glob like '*.py' (default '*')."},
                    "max_matches": {"type": "integer", "description": "Cap on matches (default 100)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace exactly one occurrence of `old_string` with `new_string` in a "
                "file. Fails if `old_string` is missing or appears more than once — "
                "include enough surrounding context to make it unique. Prefer this "
                "over `write_file` for small edits to existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "old_string": {
                        "type": "string",
                        "description": "Exact substring to replace (must be unique in the file).",
                    },
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python in a subprocess (no shell-quoting). Provide either `code` "
                "(inline snippet) or `script_path` (relative .py file), not both. Uses "
                "the same Python interpreter as the agent. Captures stdout; appends "
                "stderr if present."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Inline Python code to execute."},
                    "script_path": {"type": "string", "description": "Relative path to a .py script."},
                    "timeout": {"type": "integer", "description": "Seconds before kill (default 120)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_visit",
            "description": (
                "HTTP-fetch a URL and return its main content as Markdown (boilerplate "
                "stripped). Re-visits in the same session are cached. For long pages, "
                "the response includes the next offset to read. Only call on URLs that "
                "came from `web_search` or were given by the user — never invent URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full http(s) URL."},
                    "max_chars": {"type": "integer", "description": "Max chars to return (default 4000)."},
                    "offset": {"type": "integer", "description": "Start position into the page (default 0)."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo. Returns ranked {title, url, snippet}. "
                "Pair with `browser_visit` on the most relevant URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "Number of results (1..20, default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": (
                "Call this when the task or current step is finished. Provide your "
                "final answer or a summary of what was done. After this call, the loop "
                "ends and the next step (if any) begins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Final answer or summary of what was accomplished.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


# Convenience: set of all tool names defined here, used for routing/validation.
TOOL_NAMES = {t["function"]["name"] for t in TOOL_DEFINITIONS}
