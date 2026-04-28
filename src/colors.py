"""ANSI color helpers for terminal output.

Auto-disables when stdout isn't a TTY or when the NO_COLOR env var is set
(https://no-color.org). Codes resolve to empty strings in that case so call
sites stay clean.
"""

import os
import sys


class C:
    """ANSI color codes and named-role helper functions."""

    _enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    RESET = "\033[0m" if _enabled else ""
    BOLD = "\033[1m" if _enabled else ""
    DIM = "\033[2m" if _enabled else ""
    RED = "\033[31m" if _enabled else ""
    GREEN = "\033[32m" if _enabled else ""
    YELLOW = "\033[33m" if _enabled else ""
    BLUE = "\033[34m" if _enabled else ""
    MAGENTA = "\033[35m" if _enabled else ""
    CYAN = "\033[36m" if _enabled else ""
    BR_GREEN = "\033[92m" if _enabled else ""
    BR_YELLOW = "\033[93m" if _enabled else ""
    BR_BLUE = "\033[94m" if _enabled else ""
    BR_MAGENTA = "\033[95m" if _enabled else ""
    BR_CYAN = "\033[96m" if _enabled else ""

    @staticmethod
    def header(s: str) -> str:
        """Bold cyan: top-level section banners."""
        return f"{C.BOLD}{C.CYAN}{s}{C.RESET}"

    @staticmethod
    def phase(s: str) -> str:
        """Bold blue: phase markers (planning, critique, refine, iteration headers)."""
        return f"{C.BOLD}{C.BLUE}{s}{C.RESET}"

    @staticmethod
    def step(s: str) -> str:
        """Bright cyan: per-step execution headers."""
        return f"{C.BOLD}{C.BR_CYAN}{s}{C.RESET}"

    @staticmethod
    def ok(s: str) -> str:
        """Green: success / approval."""
        return f"{C.GREEN}{s}{C.RESET}"

    @staticmethod
    def warn(s: str) -> str:
        """Yellow: warnings, retries, soft failures."""
        return f"{C.YELLOW}{s}{C.RESET}"

    @staticmethod
    def err(s: str) -> str:
        """Red: hard failures, loop-breaks."""
        return f"{C.RED}{s}{C.RESET}"

    @staticmethod
    def dim(s: str) -> str:
        """Dim: secondary content (LLM thoughts, response previews)."""
        return f"{C.DIM}{s}{C.RESET}"

    @staticmethod
    def hint(s: str) -> str:
        """Magenta: suggestions and tips."""
        return f"{C.MAGENTA}{s}{C.RESET}"

    @staticmethod
    def label(s: str) -> str:
        """Bold yellow: T-A-O phase labels (THINKING, ACT, OBSERVE)."""
        return f"{C.BOLD}{C.YELLOW}{s}{C.RESET}"
