"""Terminal UI helpers built on `rich` (display) and `questionary` (input).

Centralized visual output for the agent:
- spinners with elapsed-time around LLM calls
- boxed plan/step rendering
- end-of-run dashboard
- arrow-key + Enter selection prompts (`choose` / `checkbox`) with
  graceful non-TTY fallback to numbered prompts so piped/captured runs
  (CI, smoke tests, automation) keep working.

Rich auto-detects whether stdout is a TTY and degrades gracefully when
piped or captured (no escape sequences, plain text).
"""

import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# Single console instance shared across the agent.
# rich auto-detects TTY: when stdout isn't a terminal, color/markup is
# stripped and Status spinners degrade to plain prints.
console = Console()


def _interactive_capable() -> bool:
    """True when both stdin and stdout are attached to a terminal.

    Picker prompts (`choose`, `checkbox`) require this; piped/captured runs
    fall back to numbered text prompts so existing automation keeps working.
    """
    return console.is_terminal and sys.stdin.isatty()


# -----------------------------------------------------------------------------
# Spinner with elapsed timer for LLM calls
# -----------------------------------------------------------------------------

@contextmanager
def thinking(label: str = "THINKING", show_elapsed: bool = True) -> Iterator[None]:
    """Spinner with elapsed-time around an LLM call.

    Usage:
        with ui.thinking("Generating plan"):
            response = client.chat.completions.create(...)

    On TTY: shows an animated spinner with the label, replaces it with an
    elapsed-time line on exit. On non-TTY (piped, captured): prints the
    label up-front and the elapsed line on exit, no spinner.
    """
    start = time.monotonic()
    if console.is_terminal:
        with console.status(f"[bold yellow]{label}...[/bold yellow]", spinner="dots"):
            yield
    else:
        console.print(f"[bold yellow]{label}...[/bold yellow]")
        yield
    if show_elapsed:
        elapsed = time.monotonic() - start
        console.print(f"[dim]  ↳ {elapsed:.1f}s[/dim]")


# -----------------------------------------------------------------------------
# Plan rendering
# -----------------------------------------------------------------------------

def render_plan(plan: Dict[str, Any],
                suggestions: Optional[List[Dict[str, Any]]] = None,
                title: str = "📋 Proposed Plan") -> None:
    """Render a plan with optional critic suggestions as boxed panels."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan", width=4, no_wrap=True)
    table.add_column(no_wrap=False)

    for step in plan.get("steps", []) or []:
        sid = step.get("id", "?")
        desc = step.get("description", "")
        table.add_row(f"{sid}.", desc)
        crit = step.get("success_criterion", "")
        if crit:
            table.add_row("", f"[dim italic]↳ {crit}[/dim italic]")

    summary = plan.get("summary", "") or ""
    panel = Panel(
        table,
        title=f"[bold cyan]{title}[/bold cyan]",
        subtitle=f"[italic dim]{summary}[/italic dim]" if summary else None,
        border_style="cyan",
        expand=False,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)

    if suggestions:
        sug_table = Table(show_header=False, box=None, padding=(0, 1))
        sug_table.add_column(style="bold magenta", width=5, no_wrap=True)
        sug_table.add_column(no_wrap=False)
        for i, s in enumerate(suggestions, 1):
            sug_table.add_row(f"[{i}]", s.get("issue", ""))
            sug_table.add_row("", f"[dim]→ {s.get('fix', '')}[/dim]")
        console.print(Panel(
            sug_table,
            title="[bold magenta]💡 Suggested improvements[/bold magenta]",
            border_style="magenta",
            expand=False,
            padding=(1, 2),
        ))


def render_step_header(step: Dict[str, Any], cursor_1based: int, total: int) -> None:
    """Render a step header inside execute_plan. cursor_1based is 1..total."""
    sid = step.get("id", "?")
    desc = step.get("description", "")
    crit = step.get("success_criterion", "")

    header = Text()
    header.append(f"▶️  STEP {sid}", style="bold bright_cyan")
    header.append(f"  ({cursor_1based}/{total})", style="dim")
    header.append("\n")
    header.append(desc)
    if crit:
        header.append(f"\n↳ success: {crit}", style="dim")

    console.print()
    console.print(Panel(header, border_style="bright_cyan", expand=False, padding=(0, 2)))


# -----------------------------------------------------------------------------
# End-of-run dashboard
# -----------------------------------------------------------------------------

def render_dashboard(stats: Dict[str, Any]) -> None:
    """End-of-run summary card.

    Recognized keys (all optional, gracefully missing):
        task: str
        approved: bool
        duration_s: float
        steps_total: int
        steps_completed: int
        tool_counts: dict[str, int]
        replans: int
        compressions: int
        log_path: str
        critic_summary: str
    """
    approved = stats.get("approved", False)
    title = ("[bold green]✅ TASK COMPLETE[/bold green]" if approved
             else "[bold yellow]⚠️  TASK NEEDS REVIEW[/bold yellow]")
    border = "green" if approved else "yellow"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(no_wrap=False)

    if "task" in stats and stats["task"]:
        task_disp = stats["task"]
        if len(task_disp) > 90:
            task_disp = task_disp[:87] + "..."
        table.add_row("Task", task_disp)

    if "duration_s" in stats:
        d = stats["duration_s"]
        if d < 60:
            dstr = f"{d:.1f}s"
        else:
            dstr = f"{int(d // 60)}m {int(d % 60)}s"
        table.add_row("Duration", dstr)

    if stats.get("steps_total"):
        completed = stats.get("steps_completed", 0)
        total = stats["steps_total"]
        bar = "█" * completed + "░" * max(0, total - completed)
        table.add_row("Steps", f"{completed}/{total}  [bold cyan]{bar}[/bold cyan]")

    if stats.get("replans"):
        table.add_row("Replans", str(stats["replans"]))

    if stats.get("compressions"):
        table.add_row("Compressions", str(stats["compressions"]))

    if stats.get("tool_counts"):
        ordered = sorted(stats["tool_counts"].items(), key=lambda x: -x[1])
        tools_str = ", ".join(f"{n}× [cyan]{t}[/cyan]" for t, n in ordered)
        table.add_row("Tools used", tools_str)

    if stats.get("critic_summary"):
        cs = stats["critic_summary"]
        if len(cs) > 200:
            cs = cs[:197] + "..."
        table.add_row("Critic", cs)

    if "log_path" in stats and stats["log_path"]:
        table.add_row("Log", f"[dim]{stats['log_path']}[/dim]")

    console.print()
    console.print(Panel(table, title=title, border_style=border, expand=False, padding=(1, 2)))


# -----------------------------------------------------------------------------
# Interactive pickers (arrow-key navigation with Enter to select)
# -----------------------------------------------------------------------------

def choose(message: str, choices: List[Tuple[str, Any]],
           default: Optional[Any] = None) -> Any:
    """Single-select prompt: arrow keys to move, Enter to select.

    `choices` is a list of `(label, value)` pairs. The label is shown to
    the user; the corresponding value is returned.

    On TTY: uses `questionary.select` (arrow keys, mouse on supported
    terminals). On non-TTY (piped/captured stdin): falls back to a
    numbered text prompt so automation keeps working. EOF / Ctrl-C raise
    KeyboardInterrupt — the caller can decide whether to abort.
    """
    if _interactive_capable():
        try:
            import questionary
            label_to_value = {label: value for label, value in choices}
            default_label = next(
                (label for label, value in choices if value == default), None,
            )
            picked = questionary.select(
                message,
                choices=[label for label, _ in choices],
                default=default_label,
            ).ask()
            if picked is None:
                # User pressed Ctrl-C inside questionary.
                raise KeyboardInterrupt("user cancelled selection")
            return label_to_value[picked]
        except ImportError:
            pass  # fall through to text fallback

    # Non-TTY / questionary unavailable: numbered text prompt.
    console.print(message)
    for i, (label, _) in enumerate(choices, 1):
        console.print(f"  {i}. {label}")
    try:
        raw = input("Enter number (or Enter for default): ").strip()
    except EOFError:
        return default if default is not None else choices[0][1]
    if not raw:
        return default if default is not None else choices[0][1]
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx][1]
    except ValueError:
        pass
    return default if default is not None else choices[0][1]


def checkbox(message: str, choices: List[Tuple[str, Any]],
             instruction: str = "(Space to toggle, Enter to confirm)") -> List[Any]:
    """Multi-select prompt: Space to toggle, Enter to confirm.

    Returns the list of selected values (empty list when nothing selected).

    On TTY: `questionary.checkbox` with the controls hint shown inline so
    users don't have to guess. On non-TTY: comma-separated number fallback
    (e.g. `1,3` selects choices 1 and 3, empty input = nothing).
    """
    if _interactive_capable():
        try:
            import questionary
            label_to_value = {label: value for label, value in choices}
            picked = questionary.checkbox(
                message,
                choices=[label for label, _ in choices],
                instruction=instruction,
            ).ask()
            if picked is None:
                raise KeyboardInterrupt("user cancelled selection")
            return [label_to_value[label] for label in picked]
        except ImportError:
            pass

    # Non-TTY fallback.
    console.print(message)
    for i, (label, _) in enumerate(choices, 1):
        console.print(f"  {i}. {label}")
    try:
        raw = input("Enter numbers separated by commas (or Enter for none): ").strip()
    except EOFError:
        return []
    if not raw:
        return []
    selected: List[Any] = []
    for part in raw.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(choices):
                selected.append(choices[idx][1])
        except ValueError:
            continue
    return selected


def text_input(message: str) -> str:
    """Free-form text input. Always plain `input()` (questionary's text
    prompt adds little here and breaks our piped-input automation)."""
    console.print(message)
    try:
        return input("> ").strip()
    except EOFError:
        return ""
