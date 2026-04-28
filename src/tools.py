"""Tool implementations available to the agent.

Each tool is a static method returning a `ToolResult`. Absolute paths are
blocked for file I/O. Shell commands run with the current user's permissions.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

from state import ToolResult


# Directories search_in_files skips to keep results signal-rich.
_SEARCH_SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", "logs"}


class Tools:
    """Static tool implementations dispatched by `Executor._execute_tool`."""

    @staticmethod
    def execute_shell(command: str, timeout: int = 60) -> ToolResult:
        """Run a shell command and capture stdout/stderr."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, output="",
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def read_file(path: str) -> ToolResult:
        """Read the contents of a file. Absolute paths are rejected."""
        try:
            if path.startswith("/"):
                return ToolResult(
                    success=False, output="",
                    error="Absolute paths not allowed. Use relative paths.",
                )
            if not os.path.exists(path):
                return ToolResult(success=False, output="", error=f"File not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def write_file(path: str, content: str) -> ToolResult:
        """Write content to a file. Creates the file if it doesn't exist. Absolute paths rejected."""
        try:
            if path.startswith("/"):
                return ToolResult(
                    success=False, output="",
                    error="Absolute paths not allowed. Use relative paths.",
                )
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to {path}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def list_directory(path: str = ".") -> ToolResult:
        """List entries in a directory with `[DIR]` / `[FILE]` prefixes."""
        try:
            if not os.path.exists(path):
                return ToolResult(success=False, output="", error=f"Directory not found: {path}")
            if not os.path.isdir(path):
                return ToolResult(success=False, output="", error=f"Not a directory: {path}")
            items = os.listdir(path)
            output = []
            for item in sorted(items):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    output.append(f"[DIR]  {item}")
                else:
                    output.append(f"[FILE] {item}")
            return ToolResult(success=True, output="\n".join(output))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def search_in_files(pattern: str, path: str = ".", file_glob: str = "*",
                        max_matches: int = 100) -> ToolResult:
        """Recursively grep `pattern` (regex) in files matching `file_glob`.

        Output format: `path:line_no:line_content` (one match per line). Skips
        well-known noise directories (venv, __pycache__, .git, node_modules,
        logs). Caps results at `max_matches` to keep observations short.
        """
        if path.startswith("/"):
            return ToolResult(
                success=False, output="",
                error="Absolute paths not allowed. Use relative paths.",
            )
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex: {e}")

        base = Path(path)
        if not base.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")

        matches: List[str] = []
        truncated = False
        for f in base.rglob(file_glob):
            if not f.is_file():
                continue
            if any(part in _SEARCH_SKIP_DIRS for part in f.parts):
                continue
            try:
                with f.open("r", encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if regex.search(line):
                            matches.append(f"{f}:{i}:{line.rstrip()}")
                            if len(matches) >= max_matches:
                                truncated = True
                                break
            except (OSError, PermissionError):
                continue
            if truncated:
                break

        if not matches:
            return ToolResult(success=True, output=f"No matches for pattern: {pattern}")
        out = "\n".join(matches)
        if truncated:
            out += f"\n... (truncated at {max_matches} matches)"
        return ToolResult(success=True, output=out)

    @staticmethod
    def edit_file(path: str, old_string: str, new_string: str) -> ToolResult:
        """Replace exactly one occurrence of `old_string` with `new_string`.

        Fails if `old_string` is missing or appears more than once — provide
        more surrounding context to disambiguate. For renames or bulk edits,
        use multiple `edit_file` calls or fall back to `read_file` + `write_file`.
        """
        if path.startswith("/"):
            return ToolResult(
                success=False, output="",
                error="Absolute paths not allowed. Use relative paths.",
            )
        if not os.path.exists(path):
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

        count = content.count(old_string)
        if count == 0:
            return ToolResult(
                success=False, output="",
                error=f"old_string not found in {path}",
            )
        if count > 1:
            return ToolResult(
                success=False, output="",
                error=(f"old_string appears {count} times in {path} — provide "
                       f"more surrounding context to uniquely identify the location"),
            )

        new_content = content.replace(old_string, new_string, 1)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
        return ToolResult(
            success=True,
            output=f"Edited {path}: replaced {len(old_string)} chars with {len(new_string)} chars",
        )

    @staticmethod
    def browser_visit(url: str, max_chars: int = 4000, offset: int = 0) -> ToolResult:
        """HTTP fetch a URL, extract main content as Markdown, paginate.

        Re-visits in the same session hit a per-process cache (no re-download).
        Use `offset` to read further into long pages.
        """
        # Lazy import keeps Tools loadable when browser deps aren't installed.
        from browser import visit_url
        return visit_url(url, max_chars, offset)

    @staticmethod
    def web_search(query: str, max_results: int = 5) -> ToolResult:
        """Search the web via DuckDuckGo and return top-N {title, url, snippet}.

        Pair with `browser_visit` on the most relevant URL returned.
        """
        from browser import search_web
        return search_web(query, max_results)

    @staticmethod
    def run_python(code: str = "", script_path: str = "",
                   timeout: int = 120) -> ToolResult:
        """Run Python code (inline) or a script file in a subprocess.

        Provide exactly one of `code` (inline snippet) or `script_path`
        (relative path to a .py file). Uses the same Python interpreter
        as the agent. Captures stdout; appends stderr after a separator
        when present.
        """
        if not code and not script_path:
            return ToolResult(
                success=False, output="",
                error="Provide either 'code' or 'script_path'.",
            )
        if code and script_path:
            return ToolResult(
                success=False, output="",
                error="Provide only one of 'code' or 'script_path', not both.",
            )

        if script_path:
            if script_path.startswith("/"):
                return ToolResult(
                    success=False, output="",
                    error="Absolute paths not allowed. Use relative paths.",
                )
            if not os.path.exists(script_path):
                return ToolResult(
                    success=False, output="",
                    error=f"Script not found: {script_path}",
                )
            cmd = [sys.executable, script_path]
        else:
            cmd = [sys.executable, "-c", code]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=os.getcwd(),
            )
            output = result.stdout
            if result.stderr:
                output = (output + "\n--- stderr ---\n" + result.stderr).strip()
            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Exited with code {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error=f"Timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
