"""Critic: post-execution quality review pass."""

import re
from typing import Any, Dict, List, Optional

from prompts import CRITIC_REVIEW_PROMPT, system_prefix


def is_approved(verdict: Optional[str]) -> bool:
    """True if a critic verdict signals approval.

    Naive substring matching on "APPROVED" is unsafe because "NOT APPROVED"
    contains it. We explicitly reject negated forms first, then look for
    the word as a whole token.
    """
    if not verdict:
        return False
    upper = verdict.upper()
    if "NOT APPROVED" in upper or "NOT_APPROVED" in upper or "DISAPPROVED" in upper:
        return False
    return bool(re.search(r"\bAPPROVED\b", upper))


class Critic:
    """Reviews the final aggregated result of one `run()`.

    Returns either an "APPROVED" verdict or descriptive feedback that the
    orchestrator appends to the result. Use `is_approved(verdict)` to
    interpret the return value (handles "NOT APPROVED" correctly).
    """

    def __init__(self, client, model: str, max_tokens: int = 5000, logger: Optional[Any] = None):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.logger = logger

    def review(self, task: str, result: str, action_history: List[Dict[str, Any]]) -> str:
        prompt = CRITIC_REVIEW_PROMPT.format(
            task=task, result=result, action_history=action_history,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prefix() + "You are a critical reviewer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        verdict = content.strip() if content else "APPROVED (no feedback generated)"
        if self.logger:
            self.logger.log("critic_review", {
                "verdict": verdict,
                "approved": is_approved(verdict),
            })
        return verdict
