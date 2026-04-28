"""Cross-session analyzer: roll-up across all session JSONL files.

Usage:
    python src/analyze.py              # all sessions
    python src/analyze.py --last 10    # most recent N sessions
    python src/analyze.py --dir logs   # explicit log dir

Surfaces patterns that point at agent shortcomings:
- average / median refinement rounds (high → planner under-decomposing)
- top phrases users typed in refinement (top tokens → things planner misses)
- average iterations per step (high → vague descriptions or tool gaps)
- error / warning counts overall
- critic approval rate
- struggling sessions list
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "with",
    "from", "that", "this", "these", "those", "it", "its", "you", "your",
    "we", "our", "if", "then", "but", "not", "no", "do", "does", "did",
    "can", "could", "should", "would", "will", "i", "me", "my", "also",
    "have", "has", "had", "so", "into", "out", "up", "down", "more",
    "less", "than", "just", "want", "wanted", "wants", "make", "makes",
    "made", "use", "uses", "used", "using", "apply", "suggestion", "suggestions",
    "step", "steps", "plan", "go", "ok", "yes",
}


def _load_session(path: Path) -> List[Dict[str, Any]]:
    events = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception as e:
        print(f"warning: skipping {path.name}: {e}", file=sys.stderr)
    return events


def _events_of_kind(events: Iterable[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("kind") == kind]


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower())
            if len(t) > 2 and t not in _STOPWORDS]


def _summarize_session(path: Path) -> Dict[str, Any]:
    events = _load_session(path)
    refinements = _events_of_kind(events, "plan_refinement")
    steps = _events_of_kind(events, "step_start")
    completes = _events_of_kind(events, "step_complete")
    tools = _events_of_kind(events, "tool_call")
    parse_errors = _events_of_kind(events, "parse_error")
    repeated = _events_of_kind(events, "repeated_action")
    loop_breaks = _events_of_kind(events, "loop_break")
    critic = _events_of_kind(events, "critic_review")

    iter_counts = [c["data"].get("iterations_used", 0) for c in completes]
    return {
        "path": path,
        "task": (events[0]["data"].get("task", "") if events else ""),
        "duration_s": (events[-1].get("elapsed_s", 0.0) if events else 0.0),
        "refinement_rounds": len(refinements),
        "user_inputs": [r["data"].get("user_input", "") for r in refinements],
        "steps": len(steps),
        "iter_counts": iter_counts,
        "tools_used": [t["data"].get("tool", "") for t in tools],
        "parse_errors": len(parse_errors),
        "repeated_actions": len(repeated),
        "loop_breaks": len(loop_breaks),
        "approved": bool(critic and critic[-1]["data"].get("approved")),
        "events": len(events),
    }


def _struggling(s: Dict[str, Any]) -> List[str]:
    flags = []
    if s["refinement_rounds"] >= 3:
        flags.append(f"{s['refinement_rounds']} refinements")
    if s["iter_counts"] and max(s["iter_counts"]) >= 8:
        flags.append(f"max {max(s['iter_counts'])} iters/step")
    if s["parse_errors"] >= 2:
        flags.append(f"{s['parse_errors']} parse errors")
    if s["repeated_actions"] >= 3:
        flags.append(f"{s['repeated_actions']} repeated actions")
    if not s["approved"]:
        flags.append("critic flagged")
    return flags


def _print_section(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll-up across agent session logs.")
    parser.add_argument("--dir", default="logs", help="Log directory (default: logs)")
    parser.add_argument("--last", type=int, default=0,
                        help="Limit to the N most recent sessions (default: all)")
    args = parser.parse_args()

    log_dir = Path(args.dir)
    if not log_dir.exists():
        print(f"No log directory at {log_dir}. Nothing to analyze.")
        return

    paths = sorted(log_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True)
    if args.last > 0:
        paths = paths[:args.last]
    if not paths:
        print(f"No .jsonl session files in {log_dir}.")
        return

    sessions = [_summarize_session(p) for p in paths]

    _print_section(f"Sessions analyzed: {len(sessions)}")
    total_dur = sum(s["duration_s"] for s in sessions)
    print(f"Total wall-clock time: {total_dur:.1f}s")
    approved = sum(1 for s in sessions if s["approved"])
    print(f"Critic approval rate: {approved}/{len(sessions)} "
          f"({100 * approved / len(sessions):.0f}%)")

    # Refinement rounds
    rounds = [s["refinement_rounds"] for s in sessions]
    if rounds:
        _print_section("Plan refinement rounds")
        print(f"  avg:    {statistics.mean(rounds):.2f}")
        print(f"  median: {statistics.median(rounds):.0f}")
        print(f"  max:    {max(rounds)}")

    # Top user-added phrases in refinements
    all_user_text = " ".join(t for s in sessions for t in s["user_inputs"])
    tokens = Counter(_tokenize(all_user_text))
    if tokens:
        _print_section("Top phrases users typed in refinement (planner-improvement signals)")
        for word, n in tokens.most_common(15):
            print(f"  {word:<20} {n}")

    # Iterations per step
    all_iters = [i for s in sessions for i in s["iter_counts"]]
    if all_iters:
        _print_section("Iterations per step")
        print(f"  avg:    {statistics.mean(all_iters):.2f}")
        print(f"  median: {statistics.median(all_iters):.0f}")
        print(f"  max:    {max(all_iters)}")

    # Tool usage
    all_tools = Counter(t for s in sessions for t in s["tools_used"])
    if all_tools:
        _print_section("Tool calls overall")
        for name, n in all_tools.most_common():
            print(f"  {name:<20} {n}")

    # Errors / warnings
    parse_errs = sum(s["parse_errors"] for s in sessions)
    repeats = sum(s["repeated_actions"] for s in sessions)
    breaks = sum(s["loop_breaks"] for s in sessions)
    _print_section("Errors / warnings")
    print(f"  parse_error events:       {parse_errs}")
    print(f"  repeated_action events:   {repeats}")
    print(f"  loop_break events:        {breaks}")

    # Struggling sessions
    flagged: List[Tuple[Dict[str, Any], List[str]]] = []
    for s in sessions:
        f = _struggling(s)
        if f:
            flagged.append((s, f))
    if flagged:
        _print_section(f"Struggling sessions ({len(flagged)})")
        for s, f in flagged:
            task = s["task"][:60] + ("…" if len(s["task"]) > 60 else "")
            print(f"  {s['path'].name}")
            print(f"    task:  {task}")
            print(f"    flags: {', '.join(f)}")


if __name__ == "__main__":
    main()
