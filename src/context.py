"""Short-term context management: keep the agent's prompt small without
losing important execution facts.

The shape sent to the LLM each iteration is:

    system + original task + compressed summary + current step + recent raw turns

Compression triggers when the estimated prompt size exceeds
`token_budget * trigger_ratio`. Older messages are replaced with a
deterministic "running summary" built from `state.action_history` /
`state.observation_history`, and old assistant `tool_call.arguments`
bodies (e.g. a 6KB write_file content) get stubbed.

This is purely deterministic — no LLM call. LLM-based summarization
can layer on top later by replacing `_build_summary_text`.

Tool-call integrity rules (the linchpin for OpenAI-format messages):
- Never split an assistant message with tool_calls from its tool result
  messages — they must travel together or the API rejects the request.
- The "recent window" must start at an assistant message, not mid-tool-result.
- Don't compress while a tool call is pending (last message is an
  assistant with tool_calls and no following tool result). Defensive.
"""

import json
from typing import Any, Dict, List, Optional


SUMMARY_MARKER = "[Compressed prior context]"
ARG_STUB_MIN_CHARS = 300  # only stub args longer than this


class ContextManager:
    """Maintains a small, useful prompt window across iterations.

    Public API:
        maybe_compress(messages, state) -> messages  (call before each LLM call)
        compact_step_boundary(messages, state, step, final_answer) -> messages
            (call after a plan step finishes; deterministic, free)
    """

    def __init__(self, enabled: bool = True, token_budget: int = 24000,
                 trigger_ratio: float = 0.75, recent_turns: int = 6,
                 summary_max_chars: int = 10000, logger: Optional[Any] = None):
        self.enabled = enabled
        self.token_budget = token_budget
        self.trigger_ratio = trigger_ratio
        self.recent_turns = recent_turns
        self.summary_max_chars = summary_max_chars
        self.logger = logger

    # ---- Token estimation ----------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap token estimate; ~4 chars/token works well in practice."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def estimate_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content")
            if content:
                total += self.estimate_tokens(str(content))
            for tc in (msg.get("tool_calls") or []):
                args = tc.get("function", {}).get("arguments", "")
                total += self.estimate_tokens(args)
        return total

    # ---- Public entry points ------------------------------------------------

    def maybe_compress(self, messages: List[Dict[str, Any]],
                       state: Any) -> List[Dict[str, Any]]:
        """Return a (possibly compressed) copy of messages safe to send.

        Returns `messages` unchanged when compression isn't needed or is
        disabled. Logs a `context_compressed` event when it does compress.
        """
        if not self.enabled or len(messages) < 4:
            return messages
        if self._has_pending_tool_call(messages):
            # Mid-iteration; never compress here.
            return messages

        before_tokens = self.estimate_messages_tokens(messages)
        threshold = int(self.token_budget * self.trigger_ratio)
        if before_tokens < threshold:
            return messages

        compressed = self._compress(messages, state)
        # Validate tool-call integrity. If broken, abandon and return original
        # (better to overshoot the budget than send a malformed message list).
        if not self._is_tool_call_integrity_intact(compressed):
            return messages

        after_tokens = self.estimate_messages_tokens(compressed)
        if self.logger:
            self.logger.log("context_compressed", {
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "before_messages": len(messages),
                "after_messages": len(compressed),
                "kept_recent_turns": self.recent_turns,
            })
        return compressed

    def compact_step_boundary(self, messages: List[Dict[str, Any]], state: Any,
                              step: Dict[str, Any],
                              step_final_answer: str) -> List[Dict[str, Any]]:
        """Replace the just-completed step's messages with one summary line.

        Run by `execute_plan` *after* a step finishes. Identifies the user
        message that scoped this step (the "Now do Step N: ..." line) and
        compacts everything from that point forward into a single
        `[Step N complete: ...]` user message. Free, deterministic, no LLM.
        """
        if not self.enabled or not messages:
            return messages

        step_id = step.get("id")
        if step_id is None:
            return messages

        # Locate the "Now do Step N:" user message that scoped this step.
        marker = f"Now do Step {step_id}"
        scope_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and marker in (msg.get("content") or ""):
                scope_idx = i
                break
        if scope_idx is None:
            return messages

        # Verify trailing messages form a complete tool-call cycle (so we
        # can safely drop them all). If anything looks pending, bail out.
        if self._has_pending_tool_call(messages[scope_idx:]):
            return messages

        # Build the summary line.
        description = step.get("description", "")
        success_criterion = step.get("success_criterion", "")
        answer_preview = (step_final_answer or "").strip()
        if len(answer_preview) > 800:
            answer_preview = answer_preview[:800] + "..."

        summary = (
            f"[Step {step_id} complete] {description}\n"
            f"Success criterion: {success_criterion}\n"
            f"Outcome: {answer_preview}"
        )

        compacted = messages[:scope_idx] + [
            {"role": "user", "content": summary},
        ]

        if self.logger:
            self.logger.log("step_compacted", {
                "step_id": step_id,
                "messages_dropped": len(messages) - len(compacted),
                "before_messages": len(messages),
                "after_messages": len(compacted),
            })
        return compacted

    # ---- Compression internals ----------------------------------------------

    def _compress(self, messages: List[Dict[str, Any]],
                  state: Any) -> List[Dict[str, Any]]:
        """Build: system + summary + recent_turns_at_clean_boundary."""
        system_msgs: List[Dict[str, Any]] = [
            m for m in messages if m.get("role") == "system"
        ]
        non_system = [m for m in messages if m.get("role") != "system"]

        recent_start = self._find_recent_window_start(non_system)
        if recent_start <= 0:
            # Recent window covers everything — nothing to compress.
            return messages

        old_block = non_system[:recent_start]
        recent = non_system[recent_start:]

        # Apply argument stubbing to the OLD block's assistant messages,
        # not to the recent window (model needs full context for those).
        old_block = self._stub_old_arguments(old_block)

        # Build a single deterministic summary message replacing the old block.
        summary_text = self._build_summary_text(state, old_block, messages)
        summary_msg = {"role": "user", "content": summary_text}

        return system_msgs + [summary_msg] + recent

    def _find_recent_window_start(self, non_system: List[Dict[str, Any]]) -> int:
        """Return the index where the recent window starts.

        A "turn" = one assistant message + its tool result messages. The
        recent window covers the last `recent_turns` turns and MUST start
        at an assistant message so we don't orphan a tool result.
        """
        seen_assistants = 0
        for i in range(len(non_system) - 1, -1, -1):
            if non_system[i].get("role") == "assistant":
                seen_assistants += 1
                if seen_assistants >= self.recent_turns:
                    return i
        # Less than recent_turns assistant messages exist — keep everything.
        return 0

    def _stub_old_arguments(self,
                            messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replace large `tool_call.arguments` string fields with stubs.

        Operates on a list of (typically old) messages. Only string fields
        within parsed arguments are stubbed; numeric/bool args pass through.
        Tool_call ids are preserved exactly so the API still accepts them.
        """
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                out.append(msg)
                continue
            new_tool_calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "")
                if len(args_str) <= ARG_STUB_MIN_CHARS:
                    new_tool_calls.append(tc)
                    continue
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, ValueError):
                    new_tool_calls.append(tc)
                    continue
                if not isinstance(args, dict):
                    new_tool_calls.append(tc)
                    continue
                stubbed = {}
                for k, v in args.items():
                    if isinstance(v, str) and len(v) > 200:
                        stubbed[k] = f"<{len(v)} chars; included in earlier turn>"
                    else:
                        stubbed[k] = v
                new_tc = dict(tc)
                new_tc["function"] = {**fn, "arguments": json.dumps(stubbed)}
                new_tool_calls.append(new_tc)
            new_msg = dict(msg)
            new_msg["tool_calls"] = new_tool_calls
            out.append(new_msg)
        return out

    def _build_summary_text(self, state: Any,
                            old_block: List[Dict[str, Any]],
                            full_messages: List[Dict[str, Any]]) -> str:
        """Build a deterministic running summary from state + the old block.

        Sections (only emitted when non-empty):
            Original task, Plan progress, Files modified, Commands run,
            Web searches / Pages visited, Recent observation previews,
            Notable errors.
        """
        action_history = list(getattr(state, "action_history", []) or [])
        observation_history = list(getattr(state, "observation_history", []) or [])
        plan = getattr(state, "plan", None)
        task = getattr(state, "task", "")

        parts: List[str] = [SUMMARY_MARKER]

        if task:
            parts.append(f"Original task: {task[:300]}")

        if plan:
            steps = plan.get("steps", []) or []
            parts.append(f"Plan: {plan.get('summary', '')[:200]}")
            parts.append(f"Plan steps total: {len(steps)}")

        files = self._collect_paths(action_history, ("write_file", "edit_file"), "path")
        if files:
            parts.append("Files modified:")
            for f in sorted(files)[:20]:
                parts.append(f"  - {f}")

        commands = self._collect_strings(action_history, "execute_shell", "command")
        if commands:
            parts.append("Recent shell commands:")
            for cmd in commands[-10:]:
                parts.append(f"  - {cmd[:120]}")

        searches = self._collect_strings(action_history, "web_search", "query")
        if searches:
            parts.append("Web searches:")
            for q in searches[-8:]:
                parts.append(f"  - {q[:120]}")

        visits = self._collect_strings(action_history, "browser_visit", "url")
        if visits:
            parts.append("Pages visited:")
            for u in visits[-10:]:
                parts.append(f"  - {u[:200]}")

        # Highlight the most recent error observations.
        errors = [o for o in observation_history if o and "ERROR" in o[:200]]
        if errors:
            parts.append("Notable errors:")
            for e in errors[-5:]:
                preview = e.replace("\n", " ")[:200]
                parts.append(f"  - {preview}")

        if observation_history:
            parts.append("Recent observation previews:")
            for obs in observation_history[-3:]:
                preview = (obs or "").replace("\n", " ")[:200]
                parts.append(f"  - {preview}{'...' if len(obs or '') > 200 else ''}")

        text = "\n".join(parts)
        # Final safety cap.
        if len(text) > self.summary_max_chars:
            text = text[: self.summary_max_chars] + "\n[summary truncated]"
        return text

    @staticmethod
    def _collect_paths(actions: List[Dict[str, Any]], tools: tuple,
                       key: str) -> set:
        out = set()
        for a in actions:
            if a.get("tool") in tools:
                v = (a.get("params") or {}).get(key)
                if v:
                    out.add(v)
        return out

    @staticmethod
    def _collect_strings(actions: List[Dict[str, Any]], tool: str,
                         key: str) -> List[str]:
        out = []
        for a in actions:
            if a.get("tool") == tool:
                v = (a.get("params") or {}).get(key)
                if v:
                    out.append(str(v))
        return out

    # ---- Tool-call integrity --------------------------------------------------

    def _has_pending_tool_call(self, messages: List[Dict[str, Any]]) -> bool:
        """True if the last message is an assistant with tool_calls but no
        following tool/user response yet (i.e. mid-iteration)."""
        if not messages:
            return False
        last = messages[-1]
        return (
            last.get("role") == "assistant"
            and bool(last.get("tool_calls"))
        )

    def _is_tool_call_integrity_intact(self,
                                       messages: List[Dict[str, Any]]) -> bool:
        """Verify every assistant tool_call has a matching tool result.

        For the OpenAI-structured path: every tool_call_id in an assistant
        message must appear as `tool_call_id` on a later `role: tool` message
        before the next assistant message. For the XML fallback path the
        rule doesn't apply (tool calls are text inside content); we only
        validate when tool_calls field is populated.
        """
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                expected_ids = {
                    tc.get("id") for tc in msg["tool_calls"] if tc.get("id")
                }
                # Walk forward until next assistant or end of list, gathering
                # tool_call_ids we see on tool messages.
                seen = set()
                j = i + 1
                while j < len(messages) and messages[j].get("role") != "assistant":
                    if messages[j].get("role") == "tool":
                        cid = messages[j].get("tool_call_id")
                        if cid:
                            seen.add(cid)
                    j += 1
                if not expected_ids.issubset(seen):
                    return False
                i = j
            else:
                i += 1
        return True


def make_context_manager(logger: Optional[Any] = None) -> Optional[ContextManager]:
    """Build a ContextManager from env vars. Returns None when disabled."""
    import os
    enabled = os.getenv("AGENT_CONTEXT_COMPRESSION_ENABLED", "true").lower() == "true"
    if not enabled:
        return None
    return ContextManager(
        enabled=True,
        token_budget=int(os.getenv("AGENT_CONTEXT_TOKEN_BUDGET", "24000")),
        trigger_ratio=float(os.getenv("AGENT_CONTEXT_TRIGGER_RATIO", "0.75")),
        recent_turns=int(os.getenv("AGENT_CONTEXT_RECENT_TURNS", "6")),
        summary_max_chars=int(os.getenv("AGENT_CONTEXT_SUMMARY_MAX_CHARS", "10000")),
        logger=logger,
    )
