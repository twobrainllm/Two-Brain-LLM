"""The Two-Brain router: mask, decide, answer, rehydrate.

The ordering here is the privacy guarantee, and it is deliberate:

    1. mask the query                     <- before any routing decision is made
    2. assert the masked-token invariant  <- fatal if raw PII survived
    3. score difficulty / estimate latency
    4. answer locally, or mask+compress context and escalate
    5. rehydrate the answer               <- only after it is back on-device

Only masked text ever reaches `CloudDeepBrain`. The vault never leaves this
process.
"""
from __future__ import annotations

import os
from typing import Literal

from two_brain_router.privacy import MaskResult, PIIGuard, assert_masked_token_invariant
from two_brain_router.routing.brains import (
    Brain,
    CloudDeepBrain,
    GpuLocalBrain,
    LocalFastBrain,
    NpuFastBrain,
)
from two_brain_router.routing.policy import RouteDecision, RoutePolicy
from two_brain_router.signals import DifficultyEstimator, TierSignals

Tier = Literal["mobile", "pc"]

#: Which data/ captures back each local tier.
_TIER_FILES: dict[str, tuple[str, str]] = {
    # tier -> (hardware_detect file, model/tier name used by the other tools)
    "mobile": ("mobile", "mobile_1b"),
    "pc": ("ai_pc", "pc_3b"),
}

#: Opt-in switch for the real on-device NPU brain (pc tier only -- mobile
#: stays a stub, see brains.py). Unset by default so the base package's test
#: suite and CLI demo stay stdlib-only and fast; set to "1" from the
#: .venv-npu environment that actually has the runtime + hardware for it
#: (see superpowers/deploy-local-brain-npu.md Phase 1).
_NPU_BRAIN_ENV_VAR = "TWO_BRAIN_NPU_BRAIN"
_GPU_BRAIN_ENV_VAR = "TWO_BRAIN_GPU_BRAIN"


def _build_fast_brain(tier: Tier, signals: TierSignals) -> Brain:
    """Pick the AI-PC tier's fast brain; both real backends are opt-in.

    Neither real brain is the default -- unset, the base package stays
    stdlib-only and uses the mock. Which of the two *should* be preferred is an
    open question: the GPU is ~3x faster on throughput, but the NPU exists for
    power efficiency and perf-per-watt has not been measured. See
    docs/local-inference-status.md. GPU wins if both are set, purely so the
    combination is deterministic rather than an error.
    """
    if tier == "pc" and os.environ.get(_GPU_BRAIN_ENV_VAR) == "1":
        return GpuLocalBrain(tier, signals)
    if tier == "pc" and os.environ.get(_NPU_BRAIN_ENV_VAR) == "1":
        return NpuFastBrain(tier, signals)
    return LocalFastBrain(tier, signals)


class TwoBrainRouter:
    """Routes a query to the local fast brain or escalates to the Cloud AI 100
    deep brain, masking PII and compressing context before anything leaves
    the device boundary.
    """

    def __init__(self, tier: Tier, policy: RoutePolicy | None = None) -> None:
        hw_file, tier_name = _TIER_FILES[tier]
        self.tier = tier
        self.policy = policy or RoutePolicy()
        self.local = TierSignals.load(tier_name, hw_file)
        self.cloud = TierSignals.load("cloud_large", "cloud_ai100")
        self.difficulty = DifficultyEstimator()
        self.fast_brain = _build_fast_brain(tier, self.local)
        self.deep_brain = CloudDeepBrain(self.cloud)

    def route(self, query: str, context: str = "") -> RouteDecision:
        notes: list[str] = []
        guard = PIIGuard()

        # 1-2. Mask before anything else, and refuse to continue if the mask leaked.
        masked_query_result = guard.mask(query)
        assert_masked_token_invariant(query, masked_query_result)
        n_entities = len(masked_query_result.vault)
        if n_entities:
            notes.append(
                f"masked {n_entities} PII "
                f"entit{'y' if n_entities == 1 else 'ies'} before any routing decision"
            )

        # 3. Decide.
        difficulty = self.difficulty.score(query)
        local_latency_est = self.policy.estimate_local_latency_ms(self.local.profile, query)

        if self.policy.should_escalate(difficulty, local_latency_est):
            notes.append(self.policy.escalation_note(difficulty, local_latency_est))
            return self._escalate(guard, masked_query_result, context, difficulty, notes)

        notes.append(self.policy.local_note(difficulty, local_latency_est))
        return self._answer_locally(guard, masked_query_result, difficulty, notes)

    def _answer_locally(
        self,
        guard: PIIGuard,
        masked_query: MaskResult,
        difficulty: float,
        notes: list[str],
    ) -> RouteDecision:
        response = self.fast_brain.answer(masked_query.masked_text)
        return RouteDecision(
            tier_answered="local",
            difficulty_score=difficulty,
            est_latency_ms=response.latency_ms,
            est_cost_usd=response.cost_usd,
            pii_entities_masked=len(masked_query.vault),
            answer=guard.rehydrate(response.text, masked_query.vault),
            notes=notes,
        )

    def _escalate(
        self,
        guard: PIIGuard,
        masked_query: MaskResult,
        context: str,
        difficulty: float,
        notes: list[str],
    ) -> RouteDecision:
        # 4. Context crosses the boundary too, so it is masked and compressed.
        masked_context = guard.mask(context) if context else MaskResult(masked_text="", vault={})
        compressed, was_compressed = self.policy.compress_context(masked_context.masked_text)
        if was_compressed:
            notes.append(f"compressed escalated context to {len(compressed)} chars")

        vault = dict(masked_query.vault)
        if context:
            vault.update(masked_context.vault)
        if masked_query.vault:
            notes.append(f"sent off-device (masked): {masked_query.masked_text!r}")

        response = self.deep_brain.answer(masked_query.masked_text, compressed)
        # 5. Rehydrate only now, on-device, after the answer is back.
        return RouteDecision(
            tier_answered="cloud",
            difficulty_score=difficulty,
            est_latency_ms=response.latency_ms,
            est_cost_usd=response.cost_usd,
            pii_entities_masked=len(masked_query.vault),
            answer=guard.rehydrate(response.text, vault),
            notes=notes,
        )
