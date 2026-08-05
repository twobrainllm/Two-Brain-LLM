"""Two-Brain LLM -- Privacy & Power-Aware Orchestrator.

Archetype: Privacy-Aware Edge<->Cloud Inference Routing. A local "fast brain"
(Mobile 1B or AI PC 3B, whichever tier this process is running as) answers what
it can; hard queries are masked, compressed, and escalated to a Cloud AI 100
"deep brain".

Package map -- each subpackage is one replaceable seam:

    privacy/   PII detect/mask/rehydrate. Mock stand-in for `quad.privacy`
               (gap G8). Swap `guard.PIIGuard` for the real component.
    signals/   Loads hardware_detect / convert_model / profile_workload /
               orchestrate_workload responses from `data/`, plus the query
               difficulty estimator. Swap `difficulty.DifficultyEstimator` for
               a real logprob/entropy signal once a fast-brain artifact runs.
    routing/   The policy (when to escalate), the two brain adapters, and the
               router that ties masking + policy + brains together. Swap
               `brains.LocalFastBrain` / `brains.CloudDeepBrain` for real
               inference calls once convert_model produces an artifact.

See docs/GAPS.md for which seams are mocked and exactly why.
"""
from __future__ import annotations

__version__ = "0.1.0"

from two_brain_router.privacy import MaskResult, PIIGuard, assert_masked_token_invariant
from two_brain_router.routing import RouteDecision, TwoBrainRouter
from two_brain_router.signals import DifficultyEstimator, TierSignals

__all__ = [
    "MaskResult",
    "PIIGuard",
    "assert_masked_token_invariant",
    "RouteDecision",
    "TwoBrainRouter",
    "DifficultyEstimator",
    "TierSignals",
    "__version__",
]
