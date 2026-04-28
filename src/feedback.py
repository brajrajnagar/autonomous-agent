"""Post-tool verification: deterministic checks fed back as observations.

After mutating tool calls (write_file, edit_file), the FeedbackEngine runs
configured Verifiers and produces a short text block appended to the
observation the agent sees on its next iteration. This gives the agent
ground-truth signal (syntax errors, lint warnings, test failures) instead
of inferring correctness from the tool's direct output.

Each Verifier:
- declares which tool calls it cares about (`applies_to`)
- runs cheaply and never raises (failures resolve to a "skipped" result)
- returns a one-line summary plus optional detail

Run-after policy: silent passes are dropped from the observation (to keep
prompts tight) but every result is still logged via the SessionLogger.
Verifiers can opt out of silent-on-pass via `silent_on_pass=False`.
"""

import ast
import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VerifyResult:
    """Outcome of one Verifier.verify() call."""
    name: str
    passed: bool
    summary: str
    detail: str = ""
    skipped: bool = False


def _written_path(tool_name: str, params: Dict[str, Any]) -> Optional[str]:
    """Path of the file mutated by `tool_name`, if any. Used by Verifier.applies_to."""
    if tool_name in ("write_file", "edit_file"):
        path = params.get("path", "")
        return path or None
    return None


# ---- Verifier interface --------------------------------------------------

class Verifier(ABC):
    """A single check run after a tool call.

    Subclasses set `name` and (optionally) `silent_on_pass`. They implement
    `applies_to` (cheap predicate) and `verify` (the actual check, which
    must never raise — wrap in try/except and return a skipped result).
    """

    name: str = "verifier"
    silent_on_pass: bool = True  # don't clutter observations with "OK" lines by default

    @abstractmethod
    def applies_to(self, tool_name: str, params: Dict[str, Any]) -> bool: ...

    @abstractmethod
    def verify(self, tool_name: str, params: Dict[str, Any], tool_result: Any) -> VerifyResult: ...


# ---- Concrete verifiers --------------------------------------------------

class PythonSyntaxVerifier(Verifier):
    """Run `ast.parse` on .py files just written or edited."""

    name = "py_syntax"
    silent_on_pass = True

    def applies_to(self, tool_name: str, params: Dict[str, Any]) -> bool:
        path = _written_path(tool_name, params)
        return bool(path) and path.endswith(".py")

    def verify(self, tool_name: str, params: Dict[str, Any], tool_result: Any) -> VerifyResult:
        path = params.get("path", "")
        if not os.path.exists(path):
            return VerifyResult(self.name, True, "py_syntax: file vanished", skipped=True)
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=path)
            return VerifyResult(self.name, True, f"py_syntax: OK ({path})")
        except SyntaxError as e:
            text = (e.text or "").strip()
            return VerifyResult(
                self.name, False,
                f"py_syntax: {e.msg} at {path}:{e.lineno}",
                detail=f"line {e.lineno}, col {e.offset}: {text}".strip(),
            )
        except Exception as e:
            return VerifyResult(self.name, True, f"py_syntax: skipped ({e})", skipped=True)


class JsonSyntaxVerifier(Verifier):
    """Run `json.load` on .json files just written or edited."""

    name = "json_syntax"
    silent_on_pass = True

    def applies_to(self, tool_name: str, params: Dict[str, Any]) -> bool:
        path = _written_path(tool_name, params)
        return bool(path) and path.endswith(".json")

    def verify(self, tool_name: str, params: Dict[str, Any], tool_result: Any) -> VerifyResult:
        path = params.get("path", "")
        if not os.path.exists(path):
            return VerifyResult(self.name, True, "json_syntax: file vanished", skipped=True)
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
            return VerifyResult(self.name, True, f"json_syntax: OK ({path})")
        except json.JSONDecodeError as e:
            return VerifyResult(
                self.name, False,
                f"json_syntax: {e.msg} at {path}:{e.lineno}",
                detail=f"line {e.lineno}, col {e.colno}",
            )
        except Exception as e:
            return VerifyResult(self.name, True, f"json_syntax: skipped ({e})", skipped=True)


class PythonLintVerifier(Verifier):
    """Lint .py files using `ruff` if available, else `pyflakes`. Off by default."""

    name = "py_lint"
    silent_on_pass = True

    def __init__(self):
        if shutil.which("ruff"):
            self._cmd: List[str] = ["ruff", "check", "--no-fix"]
            self._available = True
        elif shutil.which("pyflakes"):
            self._cmd = ["pyflakes"]
            self._available = True
        else:
            self._cmd = []
            self._available = False

    def applies_to(self, tool_name: str, params: Dict[str, Any]) -> bool:
        path = _written_path(tool_name, params)
        return self._available and bool(path) and path.endswith(".py")

    def verify(self, tool_name: str, params: Dict[str, Any], tool_result: Any) -> VerifyResult:
        path = params.get("path", "")
        if not self._available:
            return VerifyResult(self.name, True, "py_lint: skipped (no linter installed)", skipped=True)
        try:
            r = subprocess.run(
                [*self._cmd, path], capture_output=True, text=True, timeout=10,
            )
            output = (r.stdout + r.stderr).strip()
            if r.returncode == 0:
                return VerifyResult(self.name, True, f"py_lint: clean ({path})")
            return VerifyResult(
                self.name, False,
                f"py_lint: issues in {path}",
                detail=output,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(self.name, True, "py_lint: timed out", skipped=True)
        except Exception as e:
            return VerifyResult(self.name, True, f"py_lint: skipped ({e})", skipped=True)


class TestVerifier(Verifier):
    """Run a user-configured test command after every file mutation.

    Off by default — only active when AGENT_FEEDBACK_TEST_CMD is set.
    Loud on pass (the agent benefits from confirmation), loud on fail.
    """

    name = "tests"
    silent_on_pass = False  # confirm passes — useful signal

    def __init__(self, cmd: str, timeout: int = 120):
        self.cmd = cmd
        self.timeout = timeout
        self._available = bool(cmd.strip())

    def applies_to(self, tool_name: str, params: Dict[str, Any]) -> bool:
        return self._available and tool_name in ("write_file", "edit_file")

    def verify(self, tool_name: str, params: Dict[str, Any], tool_result: Any) -> VerifyResult:
        try:
            r = subprocess.run(
                self.cmd, shell=True, capture_output=True, text=True, timeout=self.timeout,
            )
            if r.returncode == 0:
                return VerifyResult(self.name, True, f"tests: PASS ({self.cmd})")
            output = (r.stdout + r.stderr).strip()
            return VerifyResult(
                self.name, False,
                f"tests: FAIL ({self.cmd})",
                detail=output,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(
                self.name, False,
                f"tests: TIMEOUT after {self.timeout}s ({self.cmd})",
                skipped=True,
            )
        except Exception as e:
            return VerifyResult(self.name, True, f"tests: skipped ({e})", skipped=True)


# ---- Engine + factory ---------------------------------------------------

class FeedbackEngine:
    """Coordinates verifiers and produces the observation-suffix text.

    `run_after` is called by the executor after every tool call. It:
      1. asks each verifier whether it applies
      2. invokes those that do (catching exceptions defensively)
      3. logs every result via the SessionLogger (if attached)
      4. returns a multi-line string of `[verify] ...` lines for the
         observation, suppressing silent passes to keep prompts tight
    """

    def __init__(self, verifiers: List[Verifier], output_cap: int = 500, logger: Any = None):
        self.verifiers = verifiers
        self.output_cap = output_cap
        self.logger = logger

    def run_after(self, tool_name: str, params: Dict[str, Any], tool_result: Any) -> str:
        lines: List[str] = []
        for v in self.verifiers:
            try:
                if not v.applies_to(tool_name, params):
                    continue
                result = v.verify(tool_name, params, tool_result)
            except Exception as e:
                result = VerifyResult(v.name, True, f"{v.name}: skipped ({e})", skipped=True)

            if self.logger:
                self.logger.log("verification", {
                    "verifier": v.name,
                    "tool": tool_name,
                    "passed": result.passed,
                    "skipped": result.skipped,
                    "summary": result.summary,
                })

            # Suppress in-prompt noise: silent passes and skipped no-ops aren't shown.
            if result.skipped:
                continue
            if result.passed and v.silent_on_pass and not result.detail:
                continue

            line = f"[verify] {result.summary}"
            if result.detail:
                detail = result.detail[: self.output_cap].strip()
                if detail:
                    indent = "\n  ".join(detail.splitlines())
                    line += f"\n  {indent}"
            lines.append(line)
        return "\n".join(lines)


def make_feedback_engine(logger: Any = None) -> Optional[FeedbackEngine]:
    """Build a FeedbackEngine from env vars. Returns None if disabled or empty."""
    if os.getenv("AGENT_FEEDBACK_ENABLED", "true").lower() != "true":
        return None

    verifiers: List[Verifier] = []
    if os.getenv("AGENT_FEEDBACK_PYTHON_SYNTAX", "true").lower() == "true":
        verifiers.append(PythonSyntaxVerifier())
    if os.getenv("AGENT_FEEDBACK_JSON_SYNTAX", "true").lower() == "true":
        verifiers.append(JsonSyntaxVerifier())
    if os.getenv("AGENT_FEEDBACK_PYTHON_LINT", "false").lower() == "true":
        verifiers.append(PythonLintVerifier())
    test_cmd = os.getenv("AGENT_FEEDBACK_TEST_CMD", "").strip()
    if test_cmd:
        verifiers.append(TestVerifier(test_cmd))

    if not verifiers:
        return None
    cap = int(os.getenv("AGENT_FEEDBACK_OUTPUT_CAP", "500"))
    return FeedbackEngine(verifiers, output_cap=cap, logger=logger)
