"""Critic: post-execution quality review pass."""

from typing import Any, Dict, List, Optional

from prompts import CRITIC_REVIEW_PROMPT


class Critic:
    """Reviews the final aggregated result of one `run()`.

    Returns either the literal "APPROVED" (case-insensitive substring is
    enough) or descriptive feedback the orchestrator appends to the result.
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
                {"role": "system", "content": "You are a critical reviewer."},
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
                "approved": "APPROVED" in verdict.upper(),
            })
        return verdict
