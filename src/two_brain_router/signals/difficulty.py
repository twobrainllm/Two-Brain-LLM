"""Query difficulty estimation -- the signal that drives escalation."""
from __future__ import annotations

HARD_QUERY_MARKERS: tuple[str, ...] = (
    "prove", "derive", "optimi", "algorithm", "complexity", "multi-step",
    "compare and contrast", "write a function", "debug", "root cause",
    "step by step", "architecture", "trade-off", "tradeoff",
)


class DifficultyEstimator:
    """Stand-in for the fast brain's own confidence/logit signal.

    No compiled model exists in this environment to produce a real confidence
    score (convert_model is blocked -- see data/convert_model/_real_attempts_log.md),
    so this scores difficulty from surface features of the query instead.
    This is the seam to replace with a real logprob/entropy signal from the
    fast-brain runtime once a working NPU artifact exists.
    """

    def score(self, query: str) -> float:
        q = query.lower()
        score = 0.0
        score += min(len(query) / 400.0, 0.4)
        score += 0.15 * sum(1 for m in HARD_QUERY_MARKERS if m in q)
        score += 0.2 if "?" in query and query.count("?") > 1 else 0.0
        return min(score, 1.0)
