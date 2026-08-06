"""The Two-Brain router: mask, decide, answer, rehydrate.

The ordering here is the privacy guarantee, and it is deliberate:

    1. mask the query                     <- before any routing decision is made
    2. assert the masked-token invariant  <- fatal if raw PII survived
    3. score difficulty / estimate latency
    4. answer locally, or mask+compress context and escalate
    5. rehydrate the answer               <- only after it is back on-device

Only masked text ever reaches `CloudDeepBrain`. The vault never leaves this
process.

Step 3 has two shapes, depending on what the tier's fast brain can tell us --
see `docs/ORCHESTRATOR.md`. When the brain self-rates
(`Brain.reports_confidence`), its confidence *is* the difficulty signal and
arrives attached to the answer, so the brain has to be asked before the
decision instead of after it. Masking still happens first either way: step 1
is ahead of step 3 in both shapes, which is the invariant that matters.

Step 4's "escalate" destination is no longer always the cloud. When a tier's
fast brain isn't confident and a second, better *local* opinion is
configured (`self.escalation_brain` -- currently: mobile's `PhoneFastBrain`
escalating to the AI PC's own `NpuFastBrain`), that's asked directly instead
of falling back to a keyword/length heuristic or paying cloud cost/egress for
a query a local model could plausibly still answer. Still `tier_answered =
"local"`: nothing about this crosses the boundary `CloudDeepBrain` represents.
`self.deep_brain` remains the fallback whenever no escalation brain is
configured -- unset by default, same opt-in posture as every real brain here.
"""
from __future__ import annotations

import os
from typing import Literal

from two_brain_router.privacy import MaskResult, PIIGuard, assert_masked_token_invariant
from two_brain_router.routing.brains import (
    Brain,
    BrainResponse,
    CloudDeepBrain,
    LocalFastBrain,
    NpuFastBrain,
    PhoneFastBrain,
)
from two_brain_router.routing.policy import RouteDecision, RoutePolicy
from two_brain_router.signals import DifficultyEstimator, TierSignals
from two_brain_router.signals.confidence import confidence_to_difficulty

Tier = Literal["mobile", "pc"]

#: Which data/ captures back each local tier.
_TIER_FILES: dict[str, tuple[str, str]] = {
    # tier -> (hardware_detect file, model/tier name used by the other tools)
    "mobile": ("mobile", "mobile_1b"),
    "pc": ("ai_pc", "pc_3b"),
}

#: Opt-in switch for the real on-device NPU brain (pc tier). Unset by default
#: so the base package's test suite and CLI demo stay stdlib-only and fast;
#: set to "1" from the .venv-npu environment that actually has the runtime +
#: hardware for it (see superpowers/deploy-local-brain-npu.md Phase 1).
_NPU_BRAIN_ENV_VAR = "TWO_BRAIN_NPU_BRAIN"

#: Opt-in switch for the real phone fast brain (mobile tier), plus where to
#: reach it. Same rationale as the NPU switch: unset by default so nothing in
#: the base suite depends on a served endpoint being up. The URL defaults to
#: the `adb reverse` loopback address from L_INTERFACE_CONTRACT.md.
_PHONE_BRAIN_ENV_VAR = "TWO_BRAIN_PHONE_BRAIN"
_PHONE_URL_ENV_VAR = "TWO_BRAIN_PHONE_URL"
_PHONE_MODEL_ENV_VAR = "TWO_BRAIN_PHONE_MODEL"
#: Escape hatch for PhoneFastBrain's on-device host check. Deliberately
#: separate from the enable switch so pointing the brain off-device is always
#: a distinct, deliberate act.
_PHONE_ALLOW_REMOTE_ENV_VAR = "TWO_BRAIN_PHONE_ALLOW_REMOTE"

#: `RoutePolicy.should_escalate` ORs a difficulty test with a latency test.
#: The confidence path has to evaluate those two at *different moments* -- the
#: budget before spending an inference, the difficulty only after the brain has
#: answered -- so each call passes a neutral value for the term it is not
#: asking about. Doing it this way keeps `policy.py` pure and unmodified, which
#: WALKTHROUGH next-step #4 asked for explicitly.
_NO_DIFFICULTY_SIGNAL_YET = 0.0
_BUDGET_ALREADY_CHECKED = 0.0


def _build_fast_brain(tier: Tier, signals: TierSignals) -> Brain:
    if tier == "pc" and os.environ.get(_NPU_BRAIN_ENV_VAR) == "1":
        return NpuFastBrain(tier, signals)
    if tier == "mobile" and os.environ.get(_PHONE_BRAIN_ENV_VAR) == "1":
        return PhoneFastBrain(
            tier,
            signals,
            base_url=os.environ.get(_PHONE_URL_ENV_VAR),
            model=os.environ.get(_PHONE_MODEL_ENV_VAR),
            allow_remote=os.environ.get(_PHONE_ALLOW_REMOTE_ENV_VAR) == "1",
        )
    return LocalFastBrain(tier, signals)


def _build_escalation_brain(tier: Tier, fast_brain: Brain) -> Brain | None:
    """A second, better *local* opinion for a self-rating fast brain that
    wasn't confident -- today, specifically: the mobile tier's
    `PhoneFastBrain` escalating to the AI PC's real `NpuFastBrain`, the
    exact same model/artifact the `pc` tier's own fast brain would use.

    Reuses `TWO_BRAIN_NPU_BRAIN` rather than inventing a second switch --
    it gates the same real hardware/runtime dependency either way, whether
    NpuFastBrain is this tier's *primary* fast brain or another tier's
    escalation target.

    None (no second opinion, `route()` falls back to the cloud exactly as
    before) when there is nothing to escalate a low confidence away from
    (the fast brain doesn't self-rate at all) or when the tier already *is*
    the escalation target (`pc`'s own fast brain -- see `_build_fast_brain`
    above -- already is this model directly; a second instance would just
    mean two live Genie dialog sessions, which this hardware/runtime does
    not support -- see `tests/test_npu_brain.py`'s `npu_brain` fixture).
    """
    if tier != "mobile" or not fast_brain.reports_confidence:
        return None
    if os.environ.get(_NPU_BRAIN_ENV_VAR) != "1":
        return None
    return NpuFastBrain("pc", TierSignals.load("pc_3b", "ai_pc"))


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
        self.escalation_brain = _build_escalation_brain(tier, self.fast_brain)
        self.deep_brain = CloudDeepBrain(self.cloud)

    def close(self) -> None:
        """Release any real resources a brain holds (e.g. `NpuFastBrain`'s
        Genie dialog session) -- safe to call regardless of which brains
        here are real vs. stubs, and whether an escalation brain exists."""
        for brain in (self.fast_brain, self.escalation_brain):
            close = getattr(brain, "close", None)
            if callable(close):
                close()

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
        local_latency_est = self.policy.estimate_local_latency_ms(self.local.profile, query)

        if self.fast_brain.reports_confidence:
            return self._route_on_confidence(
                guard, masked_query_result, query, context, local_latency_est, notes
            )

        difficulty = self.difficulty.score(query)
        if self.policy.should_escalate(difficulty, local_latency_est):
            notes.append(self.policy.escalation_note(difficulty, local_latency_est))
            return self._escalate(guard, masked_query_result, context, difficulty, notes)

        notes.append(self.policy.local_note(difficulty, local_latency_est))
        return self._answer_locally(guard, masked_query_result, difficulty, notes)

    def _route_on_confidence(
        self,
        guard: PIIGuard,
        masked_query: MaskResult,
        query: str,
        context: str,
        local_latency_est: float,
        notes: list[str],
    ) -> RouteDecision:
        """Ask the fast brain first, then route on the confidence it returns.

        Used when the tier's brain self-rates. The answer and the confidence
        arrive together (one inference, not two -- see
        `signals/confidence.py`), which means the brain is asked *before* the
        local-vs-cloud decision and its answer is discarded if the decision
        goes elsewhere. That cost is the accepted trade for a real signal
        instead of a keyword heuristic -- and it is now paid twice on the
        "not confident" path when an escalation brain is configured (see
        `_route_away_from_fast_brain`): two real local inferences on one
        query, deliberately not optimized for latency.

        Two details that are easy to get wrong, both load-bearing:

        - **The latency budget is checked before the call, not after.** Its job
          is to avoid *starting* a local inference that cannot finish in time.
          Re-applying it once the answer is already in hand would be actively
          harmful: escalating at that point adds the next brain's latency on
          top of the local time already spent, so it can only make the total
          worse.
        - **Masking still happens first.** This runs after step 1-2 in
          `route()`, so every brain downstream -- the mobile tier's fast
          brain over HTTP, and now potentially the AI PC's escalation brain
          too -- only ever sees masked text.
        """
        if self.policy.should_escalate(_NO_DIFFICULTY_SIGNAL_YET, local_latency_est):
            difficulty = 1.0  # no signal was even attempted -- not a heuristic score
            notes.append(
                f"skipped the fast brain: its profiled latency estimate "
                f"({local_latency_est:.0f}ms) already exceeds the "
                f"{self.policy.local_latency_budget_ms:.0f}ms budget, so a local "
                f"answer would have been discarded anyway"
            )
            notes.append(self.policy.escalation_note(difficulty, local_latency_est))
            return self._route_away_from_fast_brain(guard, masked_query, context, difficulty, notes)

        local = self.fast_brain.answer(masked_query.masked_text)
        if local.error:
            notes.append(f"fast brain reported a problem: {local.error}")

        if local.confidence is None:
            # No keyword/length heuristic fallback here on purpose -- a
            # brain that didn't report a number gets treated as maximally
            # uncertain (difficulty 1.0), not scored by a different signal
            # entirely. See docs/ORCHESTRATOR.md.
            difficulty = 1.0
            notes.append("fast brain returned no usable confidence signal")
        else:
            difficulty = confidence_to_difficulty(local.confidence)
            notes.append(
                f"fast brain self-reported confidence={local.confidence:.2f} "
                f"-> difficulty={difficulty:.2f}"
            )

        if self.policy.should_escalate(difficulty, _BUDGET_ALREADY_CHECKED):
            # Not `policy.escalation_note`: that one describes both terms of the
            # OR, and quoting a latency-vs-budget comparison here would be
            # misleading -- the budget was settled before the call and cannot be
            # what fired.
            notes.append(
                f"escalating: difficulty={difficulty:.2f} >= threshold "
                f"{self.policy.escalate_threshold} "
                f"(the local answer took {local.latency_ms:.0f}ms and was not used)"
            )
            return self._route_away_from_fast_brain(
                guard, masked_query, context, difficulty, notes, discarded=local
            )

        notes.append(self.policy.local_note(difficulty, local.latency_ms))
        return self._answer_locally(guard, masked_query, difficulty, notes, response=local)

    def _route_away_from_fast_brain(
        self,
        guard: PIIGuard,
        masked_query: MaskResult,
        context: str,
        difficulty: float,
        notes: list[str],
        discarded: BrainResponse | None = None,
    ) -> RouteDecision:
        """Where a "not confident" verdict actually goes.

        Prefers a second, better *local* opinion (`self.escalation_brain`)
        when one is configured; falls back to the original cloud escalation
        otherwise, unchanged. The trigger for reaching this method at all is
        still `RoutePolicy.should_escalate` in both callers above -- only the
        *destination* changed, not the decision logic, which is exactly why
        `routing/policy.py` needed no edits for this.
        """
        if self.escalation_brain is not None:
            return self._answer_via_escalation_brain(guard, masked_query, difficulty, notes, discarded)
        return self._escalate(guard, masked_query, context, difficulty, notes, discarded=discarded)

    def _answer_via_escalation_brain(
        self,
        guard: PIIGuard,
        masked_query: MaskResult,
        difficulty: float,
        notes: list[str],
        discarded: BrainResponse | None,
    ) -> RouteDecision:
        """A second opinion from `self.escalation_brain` (the AI PC's real
        model) instead of the cloud.

        Still `tier_answered="local"`: `NpuFastBrain` runs in-process on
        this machine (docs/npu-deployment.md), so nothing here crosses the
        boundary `CloudDeepBrain` represents -- the privacy invariant this
        project is built around is specifically about *that* boundary, and
        this path never reaches it. `discarded` is `None` when the fast
        brain was never even called (the budget pre-check fired); otherwise
        it is billed into the reported latency, same accounting `_escalate`
        already does for a discarded local answer.
        """
        notes.append(
            "not confident enough -- getting a second opinion from the AI PC's "
            "NpuFastBrain instead of falling back to a heuristic or escalating "
            "to the cloud"
        )
        response = self.escalation_brain.answer(masked_query.masked_text)

        est_latency_ms = response.latency_ms
        if discarded is not None:
            est_latency_ms += discarded.latency_ms
            notes.append(
                f"AI PC answered in {response.latency_ms:.0f}ms; reported latency "
                f"also includes the phone's discarded {discarded.latency_ms:.0f}ms"
            )

        return RouteDecision(
            tier_answered="local",
            difficulty_score=difficulty,
            est_latency_ms=est_latency_ms,
            est_cost_usd=response.cost_usd,
            pii_entities_masked=len(masked_query.vault),
            answer=guard.rehydrate(response.text, masked_query.vault),
            notes=notes,
        )

    def _answer_locally(
        self,
        guard: PIIGuard,
        masked_query: MaskResult,
        difficulty: float,
        notes: list[str],
        response: BrainResponse | None = None,
    ) -> RouteDecision:
        # `response` is already populated on the confidence path -- reusing it
        # is what keeps that path to a single inference call.
        if response is None:
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
        discarded: BrainResponse | None = None,
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

        # A speculative local answer that lost is still time the user waited
        # for, so the reported latency includes it. Hiding it would make the
        # confidence path look free when it is not.
        est_latency_ms = response.latency_ms
        if discarded is not None:
            est_latency_ms += discarded.latency_ms
            notes.append(
                f"discarded the local answer after {discarded.latency_ms:.0f}ms; "
                f"reported latency includes it"
            )

        # 5. Rehydrate only now, on-device, after the answer is back.
        return RouteDecision(
            tier_answered="cloud",
            difficulty_score=difficulty,
            est_latency_ms=est_latency_ms,
            est_cost_usd=response.cost_usd,
            pii_entities_masked=len(masked_query.vault),
            answer=guard.rehydrate(response.text, vault),
            notes=notes,
        )
