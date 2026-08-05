"""When to escalate, and what the router reports back.

Kept separate from `router.py` so the *decision* is inspectable and testable
without running a brain: `RoutePolicy.decide()` is pure, takes the difficulty
score and the tier's profiled latency, and returns why it chose what it chose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RouteDecision:
    """The full record of one routed request -- answer plus its audit trail."""

    tier_answered: Literal["local", "cloud"]
    difficulty_score: float
    est_latency_ms: float
    est_cost_usd: float
    pii_entities_masked: int
    answer: str
    notes: list[str]


@dataclass
class RoutePolicy:
    """Thresholds that decide local vs. cloud.

    Both defaults come from the profiled envelopes in `data/profile_workload/`:
    a query is escalated when it looks too hard for the fast brain, or when the
    fast brain's own profiled per-token rate says it would blow the latency
    budget anyway.
    """

    escalate_threshold: float = 0.55
    #: Local answer is preferred whenever it fits this budget at the tier's
    #: profiled per-token rate (data/profile_workload/<tier>.json).
    local_latency_budget_ms: float = 3000
    #: Escalated context is trimmed to this many characters before it leaves
    #: the device.
    max_context_chars: int = 800

    def estimate_local_latency_ms(self, profile: dict, query: str) -> float:
        """What the fast brain would cost for this query, per its profile."""
        latency_ms = profile["latency_ms"]
        n_tokens = max(len(query.split()) * 2, 16)
        return latency_ms["ttft_mean"] + n_tokens * latency_ms["per_token_mean"]

    def should_escalate(self, difficulty: float, local_latency_est_ms: float) -> bool:
        return (
            difficulty >= self.escalate_threshold
            or local_latency_est_ms > self.local_latency_budget_ms
        )

    def escalation_note(self, difficulty: float, local_latency_est_ms: float) -> str:
        return (
            f"escalating: difficulty={difficulty:.2f} (threshold {self.escalate_threshold}) "
            f"or local_latency_est={local_latency_est_ms:.0f}ms > "
            f"budget {self.local_latency_budget_ms}ms"
        )

    def local_note(self, difficulty: float, local_latency_est_ms: float) -> str:
        return (
            f"answering locally: difficulty={difficulty:.2f} < "
            f"threshold {self.escalate_threshold}, "
            f"local_latency_est={local_latency_est_ms:.0f}ms within budget"
        )

    def compress_context(self, context: str) -> tuple[str, bool]:
        """Cheap context compression before escalation: drop to the last
        `max_context_chars`, on a sentence boundary when possible. A real
        deployment would summarize with the fast brain itself before
        escalating; that needs a runnable local model, which convert_model
        couldn't produce here (same blocker as the difficulty estimator).

        Returns (compressed_text, was_compressed).
        """
        if len(context) <= self.max_context_chars:
            return context, False
        tail = context[-self.max_context_chars:]
        boundary = tail.find(". ")
        if boundary != -1:
            tail = tail[boundary + 2:]
        return tail, True
