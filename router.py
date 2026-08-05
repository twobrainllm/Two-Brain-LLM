"""Two-Brain LLM -- Privacy & Power-Aware Orchestrator.

Archetype: Privacy-Aware Edge<->Cloud Inference Routing. A local "fast brain"
(Mobile 1B or AI PC 3B, whichever tier this process is running as) answers
what it can; hard queries are masked, compressed, and escalated to a Cloud
AI 100 "deep brain". Routing thresholds are driven by real QUAD
hardware_detect output plus profile_workload/orchestrate_workload signals
(mocked where the underlying convert_model artifact couldn't be produced --
see data/*/_real_*_log.md for exactly which numbers are real vs. mocked and
why).

Run: python router.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from privacy_mask import MaskResult, PIIGuard, assert_masked_token_invariant

DATA_DIR = Path(__file__).parent / "data"
Tier = Literal["mobile", "pc"]

_HARD_QUERY_MARKERS = (
    "prove", "derive", "optimi", "algorithm", "complexity", "multi-step",
    "compare and contrast", "write a function", "debug", "root cause",
    "step by step", "architecture", "trade-off", "tradeoff",
)


def _load(tool: str, name: str) -> dict:
    return json.loads((DATA_DIR / tool / f"{name}.json").read_text())


@dataclass
class TierSignals:
    device_tier: str
    hardware: dict
    convert: dict
    profile: dict
    orchestrate: dict | None

    @classmethod
    def load(cls, tier_name: str, hw_file: str) -> "TierSignals":
        return cls(
            device_tier=tier_name,
            hardware=_load("hardware_detect", hw_file),
            convert=_load("convert_model", tier_name),
            profile=_load("profile_workload", tier_name),
            orchestrate=_load("orchestrate_workload", tier_name)
            if (DATA_DIR / "orchestrate_workload" / f"{tier_name}.json").exists()
            else None,
        )


@dataclass
class RouteDecision:
    tier_answered: Literal["local", "cloud"]
    difficulty_score: float
    est_latency_ms: float
    est_cost_usd: float
    pii_entities_masked: int
    answer: str
    notes: list[str]


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
        score += 0.15 * sum(1 for m in _HARD_QUERY_MARKERS if m in q)
        score += 0.2 if "?" in query and query.count("?") > 1 else 0.0
        return min(score, 1.0)


class TwoBrainRouter:
    """Routes a query to the local fast brain or escalates to the Cloud AI 100
    deep brain, masking PII and compressing context before anything leaves
    the device boundary.
    """

    ESCALATE_THRESHOLD = 0.55
    # Local answer is preferred whenever it fits this latency budget at the
    # tier's profiled per-token rate (data/profile_workload/<tier>.json).
    LOCAL_LATENCY_BUDGET_MS = 3000

    def __init__(self, tier: Tier) -> None:
        hw_file = "ai_pc" if tier == "pc" else "mobile"
        tier_name = "pc_3b" if tier == "pc" else "mobile_1b"
        self.tier = tier
        self.local = TierSignals.load(tier_name, hw_file)
        self.cloud = TierSignals.load("cloud_large", "cloud_ai100")
        self.difficulty = DifficultyEstimator()

    def _compress(self, context: str, max_chars: int = 800) -> tuple[str, bool]:
        """Cheap context compression before escalation: drop to the last
        max_chars, on a sentence boundary when possible. A real deployment
        would summarize with the fast brain itself before escalating; that
        needs a runnable local model, which convert_model couldn't produce
        here (same blocker as the difficulty estimator above).
        """
        if len(context) <= max_chars:
            return context, False
        tail = context[-max_chars:]
        boundary = tail.find(". ")
        if boundary != -1:
            tail = tail[boundary + 2:]
        return tail, True

    def _local_answer(self, masked_query: str) -> tuple[str, float]:
        """Fast-brain call. No compiled artifact exists to actually run
        (see data/convert_model/_real_attempts_log.md), so this returns a
        labeled stub. Latency is estimated from profile_workload's real
        envelope shape (data/profile_workload/<tier>.json).
        """
        prof = self.local.profile
        n_tokens = max(len(masked_query.split()) * 2, 16)
        est_latency = prof["latency_ms"]["ttft_mean"] + n_tokens * prof["latency_ms"]["per_token_mean"]
        answer = f"[local:{self.tier} mock fast-brain response to: {masked_query!r}]"
        return answer, est_latency

    def _cloud_answer(self, masked_query: str, compressed_context: str) -> tuple[str, float, float]:
        """Deep-brain call. Cloud AI 100 has no real plumbing in QUAD at all
        (data/hardware_detect/cloud_ai100.json) so this is a labeled stub too.
        Latency/cost are estimated from data/profile_workload/cloud_large.json.
        """
        prof = self.cloud.profile
        n_tokens = max(len(masked_query.split()) * 2, 16)
        latency = (
            prof["latency_ms"]["network_rtt_mean"]
            + prof["latency_ms"]["ttft_mean"]
            + n_tokens * prof["latency_ms"]["per_token_mean"]
        )
        cost = (n_tokens / 1000.0) * prof["token_cost_usd_per_1k"]
        answer = (
            f"[cloud:ai100 mock deep-brain response to: {masked_query!r} "
            f"| context_used={compressed_context!r}]"
        )
        return answer, latency, cost

    def route(self, query: str, context: str = "") -> RouteDecision:
        notes: list[str] = []
        guard = PIIGuard()
        masked_query_result = guard.mask(query)
        assert_masked_token_invariant(query, masked_query_result)
        n_entities = len(masked_query_result.vault)
        if n_entities:
            notes.append(f"masked {n_entities} PII entit{'y' if n_entities == 1 else 'ies'} before any routing decision")

        difficulty = self.difficulty.score(query)
        local_prof = self.local.profile
        n_tokens_est = max(len(query.split()) * 2, 16)
        local_latency_est = (
            local_prof["latency_ms"]["ttft_mean"] + n_tokens_est * local_prof["latency_ms"]["per_token_mean"]
        )

        escalate = difficulty >= self.ESCALATE_THRESHOLD or local_latency_est > self.LOCAL_LATENCY_BUDGET_MS
        if escalate:
            notes.append(
                f"escalating: difficulty={difficulty:.2f} (threshold {self.ESCALATE_THRESHOLD}) "
                f"or local_latency_est={local_latency_est:.0f}ms > budget {self.LOCAL_LATENCY_BUDGET_MS}ms"
            )
            masked_context_result = guard.mask(context) if context else MaskResult(masked_text="", vault={})
            compressed, was_compressed = self._compress(
                masked_context_result.masked_text if context else ""
            )
            if was_compressed:
                notes.append(f"compressed escalated context to {len(compressed)} chars")

            cloud_masked_vault = dict(masked_query_result.vault)
            if context:
                cloud_masked_vault.update(masked_context_result.vault)
            if n_entities:
                notes.append(f"sent off-device (masked): {masked_query_result.masked_text!r}")

            raw_answer, latency, cost = self._cloud_answer(masked_query_result.masked_text, compressed)
            answer = guard.rehydrate(raw_answer, cloud_masked_vault)
            return RouteDecision(
                tier_answered="cloud",
                difficulty_score=difficulty,
                est_latency_ms=latency,
                est_cost_usd=cost,
                pii_entities_masked=n_entities,
                answer=answer,
                notes=notes,
            )

        notes.append(
            f"answering locally: difficulty={difficulty:.2f} < threshold {self.ESCALATE_THRESHOLD}, "
            f"local_latency_est={local_latency_est:.0f}ms within budget"
        )
        raw_answer, latency = self._local_answer(masked_query_result.masked_text)
        answer = guard.rehydrate(raw_answer, masked_query_result.vault)
        return RouteDecision(
            tier_answered="local",
            difficulty_score=difficulty,
            est_latency_ms=latency,
            est_cost_usd=0.0,
            pii_entities_masked=n_entities,
            answer=answer,
            notes=notes,
        )


def _demo() -> None:
    router = TwoBrainRouter(tier="pc")
    examples = [
        ("What time zone is Tokyo in?", ""),
        (
            "Derive the time complexity of merge sort step by step and compare it "
            "to quicksort's worst case, then explain the trade-offs.",
            "",
        ),
        (
            "My email is jane.doe@example.com and my phone is 555-123-4567 -- "
            "can you draft a reply telling the sender their SSN 123-45-6789 was "
            "found in an old backup and needs to be rotated?",
            "",
        ),
    ]
    for query, ctx in examples:
        decision = router.route(query, ctx)
        print(f"\n> {query}")
        print(f"  routed to: {decision.tier_answered} | difficulty={decision.difficulty_score:.2f} "
              f"| est_latency_ms={decision.est_latency_ms:.0f} | est_cost_usd={decision.est_cost_usd:.5f}")
        for note in decision.notes:
            print(f"  - {note}")
        print(f"  answer: {decision.answer}")


if __name__ == "__main__":
    _demo()
