"""Tool implementations available to the agent.

Each tool is a static method returning a `ToolResult`. Absolute paths are
blocked for file I/O. Shell commands run with the current user's permissions.
"""

import os
import subprocess

from state import ToolResult


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
