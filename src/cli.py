"""Command-line entry points for the agent.

Two modes:
- `python src/agent.py "task"` — single-task mode, runs once and prints the result.
- `python src/agent.py` (or `--interactive`) — interactive REPL with a small
  per-process conversation history that's prepended to each new task as context.
"""

import sys

from agent import AutonomousAgent
from colors import C


_INTRO = """
Type your task, or use commands:
  - 'history'  - Show conversation history
  - 'clear'    - Clear history and start fresh
  - 'quit' or 'exit' or Ctrl+D - Exit the agent

The agent remembers context across multiple turns!
"""


def _interactive_loop() -> None:
    print("\n" + C.header("=" * 60))
    print(C.header("🤖  Autonomous Agent - Interactive Mode"))
    print(C.header("=" * 60))
    print(C.dim(_INTRO))

    agent = AutonomousAgent()
    conversation_history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input(f"\n{C.BR_GREEN}> You:{C.RESET} ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print(C.dim("\n👋 Goodbye!"))
                break

            if user_input.lower() == "history":
                print(C.phase("\n--- Conversation History ---"))
                for i, (task, result) in enumerate(conversation_history, 1):
                    print(f"\n{C.BOLD}{i}. Task:{C.RESET} {task[:50]}...")
                    print(f"   {C.dim('Result:')} {C.dim(result[:100])}...")
                if not conversation_history:
                    print(C.dim("(No history yet)"))
                continue

            if user_input.lower() == "clear":
                conversation_history = []
                print(C.ok("\n🗑️  History cleared. Starting fresh!"))
                continue

            # Build a context-aware task by prepending the last few turns.
            if conversation_history:
                recent_context = "\n".join(
                    f"Previous task {i}: {task}"
                    for i, (task, _) in enumerate(conversation_history[-3:], 1)
                )
                context_task = (
                    "You are in an ongoing conversation. Here is recent context:\n\n"
                    f"{recent_context}\n\n"
                    f"Current request from user: {user_input}\n\n"
                    "If the current request refers to previous items (like \"that file\", "
                    "\"the content\", etc.), use the context above to understand what is "
                    "being referenced."
                )
            else:
                context_task = user_input

            result = agent.run(context_task)

            print("\n" + C.header("=" * 50))
            print(C.header("🤖 Result:"))
            print(C.header("=" * 50))
            print(result)

            conversation_history.append((user_input, result))
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        except EOFError:
            print(C.dim("\n\n👋 Goodbye!"))
            break
        except KeyboardInterrupt:
            print(C.dim("\n\n👋 Goodbye!"))
            break


def _single_task(task: str) -> None:
    if task.lower() in ("quit", "exit"):
        return
    agent = AutonomousAgent()
    result = agent.run(task)
    print("\n" + C.header("=" * 50))
    print(C.header("FINAL RESULT:"))
    print(C.header("=" * 50))
    print(result)


def main() -> None:
    """CLI entry point. Used by `python src/agent.py [args]`."""
    interactive_mode = "--interactive" in sys.argv or len(sys.argv) == 1
    if interactive_mode:
        _interactive_loop()
    else:
        _single_task(" ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
