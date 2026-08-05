"""The two brains, behind one interface.

Neither brain runs a real model here: `convert_model` could not produce an
artifact on any execution path (gap #3/#3b), and the Cloud AI 100 has no
plumbing in QUAD-Client at all (gap #4). Both therefore return a *labeled*
stub answer with latency/cost estimated from the tier's real
`profile_workload` envelope shape.

This is the seam to replace with real inference: implement `Brain.answer` and
keep the returned `BrainResponse` shape, and the router, policy, and privacy
guarantees above it are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from two_brain_router.signals.loader import TierSignals


@dataclass
class BrainResponse:
    """What any brain must return: the text, plus what it cost to get it."""

    text: str
    latency_ms: float
    cost_usd: float = 0.0


class Brain(Protocol):
    """Implement this to plug a real runtime in behind the router."""

    def answer(self, masked_query: str, context: str = "") -> BrainResponse: ...


def _estimate_tokens(masked_query: str) -> int:
    """Token count the latency/cost estimates are driven off."""
    return max(len(masked_query.split()) * 2, 16)


class LocalFastBrain:
    """On-device fast brain (Mobile 1B / AI PC 3B).

    Returns a labeled stub -- no compiled artifact exists to actually run (see
    data/convert_model/_real_attempts_log.md). Latency is estimated from
    profile_workload's real envelope shape (data/profile_workload/<tier>.json).
    """

    def __init__(self, tier: str, signals: TierSignals) -> None:
        self.tier = tier
        self.signals = signals

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        latency_ms = self.signals.profile["latency_ms"]
        n_tokens = _estimate_tokens(masked_query)
        est_latency = latency_ms["ttft_mean"] + n_tokens * latency_ms["per_token_mean"]
        return BrainResponse(
            text=f"[local:{self.tier} mock fast-brain response to: {masked_query!r}]",
            latency_ms=est_latency,
            cost_usd=0.0,
        )


class CloudDeepBrain:
    """Off-device deep brain (Cloud AI 100).

    Also a labeled stub: the Cloud AI 100 has no real plumbing in QUAD at all
    (data/hardware_detect/cloud_ai100.json). Latency and cost are estimated
    from data/profile_workload/cloud_large.json, network RTT included.
    """

    def __init__(self, signals: TierSignals) -> None:
        self.signals = signals

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        prof = self.signals.profile
        latency_ms = prof["latency_ms"]
        n_tokens = _estimate_tokens(masked_query)
        est_latency = (
            latency_ms["network_rtt_mean"]
            + latency_ms["ttft_mean"]
            + n_tokens * latency_ms["per_token_mean"]
        )
        cost = (n_tokens / 1000.0) * prof["token_cost_usd_per_1k"]
        return BrainResponse(
            text=(
                f"[cloud:ai100 mock deep-brain response to: {masked_query!r} "
                f"| context_used={context!r}]"
            ),
            latency_ms=est_latency,
            cost_usd=cost,
        )
