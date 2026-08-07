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
import time
from pathlib import Path
from typing import Literal

from two_brain_router.privacy import MaskResult, PIIGuard, assert_masked_token_invariant
from two_brain_router.routing.brains import (
    Brain,
    CirrascaleDeepBrain,
    CloudDeepBrain,
    GpuLocalBrain,
    LocalFastBrain,
    NpuFastBrain,
)
from two_brain_router.routing.policy import RouteDecision, RoutePolicy
from two_brain_router.signals import DifficultyEstimator, TierSignals

Tier = Literal["mobile", "pc"]
#: Which brain answered / may be forced to answer.
Tier2 = Literal["local", "cloud"]

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
_CLOUD_BRAIN_ENV_VAR = "TWO_BRAIN_CLOUD_BRAIN"

#: Demo/UI-testing switch. `route(force_tier=...)` is honoured ONLY when this
#: is "1". Unset or "0" -- which is every normal run, the test suite, and the
#: CLI -- a forced tier is ignored and `policy.should_escalate` decides, so a
#: demo affordance can never quietly become the routing behaviour in
#: production. The flag gates the *override*, never the privacy ordering.
_UI_TEST_ENV_VAR = "UI_TEST"


def ui_test_enabled() -> bool:
    """True only when UI_TEST=1, the one condition under which a tier may be forced."""
    return os.environ.get(_UI_TEST_ENV_VAR, "0") == "1"


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


def _build_deep_brain(signals: TierSignals) -> Brain:
    """The cloud tier, real when opted in.

    Same opt-in shape as the fast brain: unset, the stub keeps the base
    package stdlib-only *and* offline, so the demo and the test suite never
    depend on a network call or a credential.
    """
    if os.environ.get(_CLOUD_BRAIN_ENV_VAR) == "1":
        return CirrascaleDeepBrain(signals)
    return CloudDeepBrain(signals)


class _IncrementalRehydrator:
    """Rehydrates placeholders in a stream, without splitting one across chunks.

    Rehydration is a plain string replace on a complete answer. Streaming breaks
    that: `[PII_EMAIL_1]` can arrive as `[PII_EM` then `AIL_1]`, and replacing
    per chunk would emit the placeholder verbatim to the user.

    So text is held back whenever the tail could still become a placeholder --
    anything after an unmatched `[` -- and released once the bracket closes or
    the tail can no longer be a prefix of one. Invariant #5 is unchanged: this
    runs on-device, after the answer is back, and the vault never leaves.
    """

    def __init__(self, guard: PIIGuard, vault: dict[str, str]) -> None:
        self._guard = guard
        self._vault = vault
        self._pending = ""

    def feed(self, chunk: str) -> str:
        self._pending += chunk
        cut = self._pending.rfind("[")
        if cut == -1:
            out, self._pending = self._pending, ""
        elif "]" in self._pending[cut:]:
            # The last bracket is closed, so nothing is mid-placeholder.
            out, self._pending = self._pending, ""
        else:
            out, self._pending = self._pending[:cut], self._pending[cut:]
        return self._guard.rehydrate(out, self._vault) if out else ""

    def flush(self) -> str:
        out, self._pending = self._pending, ""
        return self._guard.rehydrate(out, self._vault) if out else ""


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
        self.deep_brain = _build_deep_brain(self.cloud)

    def route(
        self,
        query: str,
        context: str = "",
        image: Path | None = None,
        force_tier: Tier2 | None = None,
    ) -> RouteDecision:
        """Route one query.

        `force_tier` pins which brain answers, for the chat UI's local/cloud
        switch. **It is honoured only when `UI_TEST=1`.** Everywhere else --
        every normal run, the test suite, the CLI -- it is ignored, a note says
        so, and `policy.should_escalate` decides as usual. A demo affordance
        should not be able to become the production routing behaviour by
        accident.

        Even when honoured it bypasses the difficulty heuristic and nothing
        else: masking still happens first, the invariant still runs, and an
        escalated image is still described on-device with the description
        masked. Forcing a tier is never a way around the privacy guarantee.
        """
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

        if image is not None:
            if not getattr(self.fast_brain, "can_see", False):
                raise ValueError(
                    "an image was supplied but the fast brain cannot see -- "
                    "enable TWO_BRAIN_GPU_BRAIN=1 and build it with "
                    "GpuLocalBrain.for_vision()"
                )
            notes.append("image stays on-device: the cloud tier has no VLM")

        # 3. Decide.
        difficulty = self.difficulty.score(query)
        local_latency_est = self.policy.estimate_local_latency_ms(self.local.profile, query)

        if force_tier is not None:
            would_be = "cloud" if self.policy.should_escalate(difficulty, local_latency_est) else "local"
            if ui_test_enabled():
                notes.append(
                    f"tier forced to {force_tier} (UI_TEST=1; policy would have said {would_be})"
                )
                if force_tier == "cloud":
                    return self._escalate(
                        guard, masked_query_result, context, difficulty, notes, image
                    )
                return self._answer_locally(
                    guard, masked_query_result, difficulty, notes, image
                )
            # Not a silent no-op: say plainly that the override was refused, so
            # a caller who expected it to work finds out here rather than by
            # misreading which brain answered.
            notes.append(
                f"ignored force_tier={force_tier}: UI_TEST is not enabled, "
                "so the policy decides"
            )

        if self.policy.should_escalate(difficulty, local_latency_est):
            notes.append(self.policy.escalation_note(difficulty, local_latency_est))
            return self._escalate(guard, masked_query_result, context, difficulty, notes, image)

        notes.append(self.policy.local_note(difficulty, local_latency_est))
        return self._answer_locally(guard, masked_query_result, difficulty, notes, image)

    def route_stream(
        self,
        query: str,
        context: str = "",
        image: Path | None = None,
        force_tier: Tier2 | None = None,
    ):
        """Route one query, yielding the answer as it is generated.

        Yields `("meta", {...})` once the tier is decided, then `("delta", str)`
        repeatedly, then `("done", {...})`.

        The privacy ordering is identical to `route()` and deliberately still
        blocking where it must be: masking, the invariant, and -- for an
        escalated image -- describing the image and masking that description all
        complete *before* the first delta. Only generation is streamed.
        """
        notes: list[str] = []
        guard = PIIGuard()

        masked_query_result = guard.mask(query)
        assert_masked_token_invariant(query, masked_query_result)
        if masked_query_result.vault:
            n = len(masked_query_result.vault)
            notes.append(f"masked {n} PII entit{'y' if n == 1 else 'ies'} before any routing decision")

        if image is not None and not getattr(self.fast_brain, "can_see", False):
            raise ValueError(
                "an image was supplied but the fast brain cannot see -- "
                "enable TWO_BRAIN_GPU_BRAIN=1 and build it with GpuLocalBrain.for_vision()"
            )
        if image is not None:
            notes.append("image stays on-device: the cloud tier has no VLM")

        difficulty = self.difficulty.score(query)
        local_latency_est = self.policy.estimate_local_latency_ms(self.local.profile, query)

        if force_tier is not None and ui_test_enabled():
            would_be = "cloud" if self.policy.should_escalate(difficulty, local_latency_est) else "local"
            notes.append(f"tier forced to {force_tier} (UI_TEST=1; policy would have said {would_be})")
            escalate = force_tier == "cloud"
        else:
            if force_tier is not None:
                notes.append(f"ignored force_tier={force_tier}: UI_TEST is not enabled, so the policy decides")
            escalate = self.policy.should_escalate(difficulty, local_latency_est)
            notes.append(
                self.policy.escalation_note(difficulty, local_latency_est)
                if escalate
                else self.policy.local_note(difficulty, local_latency_est)
            )

        vault = dict(masked_query_result.vault)
        if escalate:
            # Same as _escalate, and for the same reasons -- the image is turned
            # into words here, before anything is streamed, and those words are
            # masked like any other text.
            if image is not None:
                described = self.fast_brain.describe_image(image, masked_query_result.masked_text)
                masked_description = guard.mask(described.text)
                assert_masked_token_invariant(described.text, masked_description)
                vault.update(masked_description.vault)
                context = (
                    f"{context}\n\n{masked_description.masked_text}".strip()
                    if context
                    else masked_description.masked_text
                )
                notes.append(
                    f"image described on-device into {len(masked_description.masked_text)} chars; "
                    f"masked {len(masked_description.vault)} PII entit"
                    f"{'y' if len(masked_description.vault) == 1 else 'ies'} in the description"
                )
            masked_context = guard.mask(context) if context else MaskResult(masked_text="", vault={})
            vault.update(masked_context.vault)
            compressed, was_compressed = self.policy.compress_context(masked_context.masked_text)
            if was_compressed:
                notes.append(f"compressed escalated context to {len(compressed)} chars")
            if masked_query_result.vault:
                notes.append(f"sent off-device (masked): {masked_query_result.masked_text!r}")
            stream = self.deep_brain.answer_stream(masked_query_result.masked_text, compressed)
        else:
            kwargs = {"image": image} if image is not None else {}
            stream = self.fast_brain.answer_stream(masked_query_result.masked_text, **kwargs)

        tier = "cloud" if escalate else "local"
        yield ("meta", {
            "tier_answered": tier,
            "difficulty_score": difficulty,
            "pii_entities_masked": len(masked_query_result.vault),
            "notes": notes,
        })

        rehydrator = _IncrementalRehydrator(guard, vault)
        started = time.perf_counter()
        n_chars = 0
        for piece in stream:
            n_chars += len(piece)
            out = rehydrator.feed(piece)
            if out:
                yield ("delta", out)
        tail = rehydrator.flush()
        if tail:
            yield ("delta", tail)

        yield ("done", {
            "tier_answered": tier,
            "est_latency_ms": (time.perf_counter() - started) * 1000,
            "est_cost_usd": 0.0,
            "chars": n_chars,
        })

    def _answer_locally(
        self,
        guard: PIIGuard,
        masked_query: MaskResult,
        difficulty: float,
        notes: list[str],
        image: Path | None = None,
    ) -> RouteDecision:
        if image is not None:
            # The image never left the device, so the local VLM sees it directly.
            response = self.fast_brain.answer(masked_query.masked_text, image=image)
        else:
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
        image: Path | None = None,
    ) -> RouteDecision:
        # 3b. An image cannot cross the boundary -- the deep brain is a text-only
        # LLM with no vision support at all. So the local VLM converts it to
        # words here, on-device, and only those words are eligible to leave.
        # The description is steered by the query: a physics diagram needs the
        # mechanical arrangement, "what is this?" needs identification.
        description_vault: dict[str, str] = {}
        if image is not None:
            described = self.fast_brain.describe_image(image, masked_query.masked_text)
            # The description is newly generated text that has never been
            # masked. It can easily contain PII the query did not -- a name on
            # a document, an address on a sign, a face described in words -- so
            # it is masked exactly like any other text before it can escalate,
            # and the invariant is asserted on it too.
            masked_description = guard.mask(described.text)
            assert_masked_token_invariant(described.text, masked_description)
            context = f"{context}\n\n{masked_description.masked_text}".strip() if context else masked_description.masked_text
            notes.append(
                f"image described on-device into {len(masked_description.masked_text)} chars; "
                f"masked {len(masked_description.vault)} PII entit"
                f"{'y' if len(masked_description.vault) == 1 else 'ies'} in the description"
            )

        # 4. Context crosses the boundary too, so it is masked and compressed.
        # The description above is already masked; masking it again is a no-op
        # on placeholders and keeps a single path for everything that leaves.
        masked_context = guard.mask(context) if context else MaskResult(masked_text="", vault={})
        compressed, was_compressed = self.policy.compress_context(masked_context.masked_text)
        if was_compressed:
            notes.append(f"compressed escalated context to {len(compressed)} chars")

        vault = dict(masked_query.vault)
        if context:
            vault.update(masked_context.vault)
        # The description was masked *before* being folded into context, so
        # re-masking context cannot rediscover its entities -- placeholders are
        # not email- or phone-shaped. Its vault has to be merged explicitly or
        # the answer comes back with a bare [PII_*] token the user never sees
        # resolved. Safe to merge because placeholders are unique per guard.
        if description_vault:
            vault.update(description_vault)
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
