"""The Two-Brain router: detect, answer, mask at the boundary, rehydrate.

The ordering here is the privacy guarantee, and it is deliberate:

    1. detect PII in the query               <- for the audit trail, not a mask
    2. answer on-device, with the raw query  <- the local model is inside the boundary
    3. if anything must leave: mask it, and assert the masked-token invariant
    4. only then call the deep brain
    5. rehydrate the answer                  <- on-device, after it is back

**Masking happens at the boundary, not at the front door.** An earlier version
of this file masked the query before *anything* touched it, including the local
model. That was the wrong place. The AI PC's fast brain executes on this
machine -- `NpuFastBrain` in this very process through `ctypes`,
`GpuLocalBrain` in a `llama-server` child bound to loopback -- so nothing it is
given is transmitted anywhere, and masking it bought no privacy while
measurably costing answer quality: a model asked to draft a reply to
`[PII_EMAIL_1]` writes a worse reply than one that can see the address.

So the boundary is where the mask goes, and `Brain.trusted_with_raw_pii` is
what marks which side of it a brain sits on. Everything outside -- both cloud
brains, and `PhoneFastBrain`, which runs on a physically separate device even
though `adb reverse` makes the hop look like loopback -- still receives masked
text and nothing else. That flag defaults to False wherever it is read, so a
brain that forgets to declare it gets masked input rather than a leak.

What this costs: the routing decision is now made on raw text. That is
acceptable precisely because the thing making it is on-device -- the difficulty
heuristic is a pure local function, and the confidence/gap signals come from the
local model itself, which was already trusted with the query by the time it
produced them.

Step 2 has three shapes, depending on what the tier's fast brain can tell us --
see `docs/ORCHESTRATOR.md`. When the brain self-rates
(`Brain.reports_confidence`), its confidence *is* the difficulty signal and
arrives attached to the answer, so the brain has to be asked before the
decision instead of after it. When it also reports *gaps*
(`Brain.reports_gaps`), the decision stops being "which brain answers this" and
becomes "how is this query split between them": the local model's partial
answer is kept and shown, and only the part it named as beyond it is masked and
sent on.

Step 3's destination is not always the cloud. When a tier's fast brain isn't
confident and a second, better *local* opinion is configured
(`self.escalation_brain` -- currently: mobile's `PhoneFastBrain` escalating to
the AI PC's own real brain), that is asked directly instead. Still
`tier_answered = "local"`: nothing about it crosses the boundary
`CloudDeepBrain` represents, so nothing about it needs masking either.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from two_brain_router.privacy import MaskResult, PIIGuard, assert_masked_token_invariant
from two_brain_router.routing.brains import (
    Brain,
    BrainResponse,
    CirrascaleDeepBrain,
    CloudDeepBrain,
    GpuLocalBrain,
    LocalFastBrain,
    NpuFastBrain,
    PhoneFastBrain,
)
from two_brain_router.routing.policy import RouteDecision, RoutePolicy
from two_brain_router.signals import DifficultyEstimator, TierSignals
from two_brain_router.signals.confidence import confidence_to_difficulty
from two_brain_router.trace import Tracer, trace_enabled

Tier = Literal["mobile", "pc"]
#: Which brain answered / may be forced to answer.
Tier2 = Literal["local", "cloud"]

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
_GPU_BRAIN_ENV_VAR = "TWO_BRAIN_GPU_BRAIN"
_CLOUD_BRAIN_ENV_VAR = "TWO_BRAIN_CLOUD_BRAIN"

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
#: asking about. Doing it this way keeps `policy.py` pure, which
#: WALKTHROUGH next-step #4 asked for explicitly.
_NO_DIFFICULTY_SIGNAL_YET = 0.0
_BUDGET_ALREADY_CHECKED = 0.0


def _trusted_with_raw_pii(brain: object) -> bool:
    """Whether `brain` may be handed the query as the user typed it.

    `getattr` with a False default rather than a bare attribute read, so the
    unsafe direction is never the accidental one: a brain that predates this
    flag, or forgets to declare it, gets masked input. Opting in wrongly would
    leak; opting out wrongly only costs answer quality.
    """
    return getattr(brain, "trusted_with_raw_pii", False)


@dataclass
class _LocalView:
    """What a local brain is given, and how to read its answer back.

    Two cases, and the vault is what distinguishes them:

    - **Trusted brain** -- `text` is the raw query and `vault` is empty, so the
      rehydrate on the way out is a no-op. Nothing was masked because nothing
      needed to be.
    - **Untrusted brain** (`PhoneFastBrain`) -- `text` is masked and `vault`
      holds the placeholders, so its answer comes back in masked space and has
      to be rehydrated before the user sees it.

    Carrying both together means the callers below never have to ask which case
    they are in; they just rehydrate with whatever vault they were given.
    """

    text: str
    vault: dict[str, str] = field(default_factory=dict)


@dataclass
class _Request:
    """One in-flight `route()` call.

    A single object rather than eight positional parameters threaded through
    six methods -- which is how a real bug got in once before, when a new
    optional `image` argument silently bound to a different parameter at a call
    site that passed positionally.

    `query` is the **raw** text throughout. Anything crossing the boundary is
    masked from it at the point of crossing, never carried around pre-masked,
    so there is exactly one kind of string in this object and no chance of
    sending the wrong one.
    """

    guard: PIIGuard
    query: str
    context: str
    notes: list[str]
    #: How many PII entities the query contained, whether or not any were
    #: masked. Always recorded; masking is a separate event.
    pii_detected: int
    #: What the *fast* brain sees. Not used for anything that leaves.
    view: _LocalView
    image: Path | None = None

    def mask_for_boundary(self, text: str) -> MaskResult:
        """Mask `text` and assert the invariant, immediately before it leaves.

        Every path to `deep_brain.answer` goes through this. Keeping it a named
        method rather than two inline lines is what makes the guarantee
        greppable: if a mask/assert pair is missing somewhere, the absence of
        this call is what shows it.
        """
        result = self.guard.mask(text)
        assert_masked_token_invariant(text, result)
        return result


#: Demo/UI-testing switch. `route(force_tier=...)` is honoured ONLY when this
#: is "1". Unset or "0" -- every normal run, the test suite, the CLI -- a
#: forced tier is ignored and the policy decides, so a demo affordance can
#: never quietly become production routing. Gates the *override* only,
#: never the privacy ordering.
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
    if tier == "mobile" and os.environ.get(_PHONE_BRAIN_ENV_VAR) == "1":
        return PhoneFastBrain(
            tier,
            signals,
            base_url=os.environ.get(_PHONE_URL_ENV_VAR),
            model=os.environ.get(_PHONE_MODEL_ENV_VAR),
            allow_remote=os.environ.get(_PHONE_ALLOW_REMOTE_ENV_VAR) == "1",
        )
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


def _build_escalation_brain(tier: Tier, fast_brain: Brain) -> Brain | None:
    """A second, better *local* opinion for a self-rating fast brain that
    wasn't confident -- today: the mobile tier's `PhoneFastBrain` escalating
    to whichever real AI-PC brain this machine is configured for.

    Delegates to `_build_fast_brain("pc", ...)` rather than hardcoding
    `NpuFastBrain` -- the AI-PC tier's own fast-brain choice (GPU preferred
    over NPU, see `_build_fast_brain`) and the mobile tier's escalation
    target should never drift apart; whichever real brain the `pc` tier
    would run for itself is exactly the "second, better local opinion"
    mobile wants too, and this stays correct automatically if a third AI-PC
    backend is ever added.

    None (no second opinion, `route()` falls back to the cloud exactly as
    before) when: there is nothing to escalate a low confidence away from
    (the fast brain doesn't self-rate at all); the tier already *is* the
    escalation target (`pc`'s own fast brain would just be asking itself --
    and for `NpuFastBrain` specifically, a second instance would mean two
    live Genie dialog sessions, which this hardware/runtime does not
    support, see `tests/test_npu_brain.py`'s `npu_brain` fixture); or
    neither `TWO_BRAIN_GPU_BRAIN` nor `TWO_BRAIN_NPU_BRAIN` is set, so
    `_build_fast_brain("pc", ...)` would only return the stub -- and a stub
    is not a real second opinion.
    """
    if tier != "mobile" or not fast_brain.reports_confidence:
        return None
    candidate = _build_fast_brain("pc", TierSignals.load("pc_3b", "ai_pc"))
    if isinstance(candidate, LocalFastBrain):
        return None
    return candidate


class _IncrementalRehydrator:
    """Rehydrates placeholders in a stream, without splitting one across chunks.

    Rehydration is a plain string replace over a finished answer. Streaming
    breaks that: `[PII_EMAIL_1]` can arrive as `[PII_EM` then `AIL_1]`, and
    replacing per chunk would emit the placeholder verbatim to the user.

    So text is held back whenever the tail could still become a placeholder --
    anything after an unmatched `[` -- and released once the bracket closes.
    Step 5 is unchanged: this runs on-device, after the answer is back, and the
    vault never leaves. For a trusted local brain the vault is empty and every
    call is a pass-through.
    """

    def __init__(self, guard: PIIGuard, vault: dict[str, str]) -> None:
        self._guard = guard
        self._vault = vault
        self._pending = ""

    def feed(self, chunk: str) -> str:
        self._pending += chunk
        cut = self._pending.rfind("[")
        if cut == -1 or "]" in self._pending[cut:]:
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

    def __init__(
        self, tier: Tier, policy: RoutePolicy | None = None, trace: bool | Tracer | None = None
    ) -> None:
        hw_file, tier_name = _TIER_FILES[tier]
        self.tier = tier
        self.policy = policy or RoutePolicy()
        # None -> read TWO_BRAIN_TRACE, defaulting off, so importing this class
        # as a library stays silent and the test suite is unaffected. `cli.py`
        # and `api.py` pass True: a human watching a terminal is who it's for.
        self.tracer = trace if isinstance(trace, Tracer) else Tracer(
            enabled=trace_enabled() if trace is None else bool(trace)
        )
        self.local = TierSignals.load(tier_name, hw_file)
        self.cloud = TierSignals.load("cloud_large", "cloud_ai100")
        self.difficulty = DifficultyEstimator()
        self.fast_brain = _build_fast_brain(tier, self.local)
        self.escalation_brain = _build_escalation_brain(tier, self.fast_brain)
        self.deep_brain = _build_deep_brain(self.cloud)

    def close(self) -> None:
        """Release any real resources a brain holds (e.g. `NpuFastBrain`'s
        Genie dialog session, or `GpuLocalBrain`'s `llama-server` child
        process) -- safe to call regardless of which brains here are real
        vs. stubs, and whether an escalation brain exists. `deep_brain`
        needs no entry here: neither `CloudDeepBrain` nor
        `CirrascaleDeepBrain` holds a persistent resource."""
        for brain in (self.fast_brain, self.escalation_brain):
            close = getattr(brain, "close", None)
            if callable(close):
                close()

    def _local_view(self, guard: PIIGuard, brain: object, query: str, notes: list[str]) -> _LocalView:
        """Decide what `brain` is allowed to see, and say so in the audit trail.

        The one place `trusted_with_raw_pii` is consulted. Both outcomes are
        noted explicitly -- "the raw query stayed on-device" is as much a
        privacy claim as "it was masked", and a reader should not have to infer
        which happened from the absence of a note.
        """
        if _trusted_with_raw_pii(brain):
            if self._n(query, guard):
                notes.append(
                    f"{type(brain).__name__} runs on this device, so it gets the "
                    f"query unmasked -- nothing is transmitted"
                )
            return _LocalView(text=query)
        masked = guard.mask(query)
        assert_masked_token_invariant(query, masked)
        if masked.vault:
            notes.append(
                f"masked {len(masked.vault)} PII "
                f"entit{'y' if len(masked.vault) == 1 else 'ies'} before calling "
                f"{type(brain).__name__}, which is not on this device"
            )
        return _LocalView(text=masked.masked_text, vault=masked.vault)

    @staticmethod
    def _n(query: str, guard: PIIGuard) -> int:
        return len(guard.detect(query))

    def _ask(
        self,
        brain,
        role: str,
        text: str,
        context: str = "",
        image: Path | None = None,
        crossing: bool = False,
        vault: dict[str, str] | None = None,
    ) -> BrainResponse:
        """Call a brain, tracing what went in and what came back.

        Every `answer()` call in this class goes through here, which is the
        point: a new branch that calls a brain directly would silently drop out
        of the trace, and the trace is how anyone verifies the privacy claim.
        `crossing=True` marks the calls that leave the device -- the trace
        banners off that flag rather than guessing from the brain's type.
        """
        self.tracer.call(role, brain, text, context, crossing=crossing, vault=vault)
        if image is not None:
            response = brain.answer(text, context, image=image)
        else:
            response = brain.answer(text, context)
        self.tracer.result(brain, response, vault=None if crossing else vault)
        return response

    def route(
        self,
        query: str,
        context: str = "",
        image: Path | None = None,
        force_tier: Tier2 | None = None,
    ) -> RouteDecision:
        """Route one query. The real work is `_route`; this traces the outcome.

        Split so that *every* return path is traced, including ones added
        later -- `_route` has four `return RouteDecision(...)` sites and a fifth
        would otherwise be silently untraced.

        `force_tier` pins which brain answers, for the chat UI's local/cloud
        switch. **It is honoured only when `UI_TEST=1`**; everywhere else it is
        ignored, a note says so, and the policy decides as usual, so a demo
        affordance cannot become the production routing behaviour by accident.
        """
        decision = self._route(query, context, image, force_tier)
        self.tracer.decision(decision)
        return decision

    def _route(
        self,
        query: str,
        context: str = "",
        image: Path | None = None,
        force_tier: Tier2 | None = None,
    ) -> RouteDecision:
        notes: list[str] = []
        guard = PIIGuard()

        # 1. Detect, do not mask. What the query contained is recorded from the
        #    start; whether any of it gets masked depends on where it goes.
        detected = guard.detect(query)
        self.tracer.request(self.tier, query, context, image, detected)
        n_detected = len(detected)
        if n_detected:
            notes.append(
                f"detected {n_detected} PII "
                f"entit{'y' if n_detected == 1 else 'ies'} in the query"
            )

        if image is not None and not getattr(self.fast_brain, "can_see", False):
            raise ValueError(
                "an image was supplied but the fast brain cannot see -- "
                "enable TWO_BRAIN_GPU_BRAIN=1 and build it with "
                "GpuLocalBrain.for_vision()"
            )
        if image is not None:
            notes.append("image stays on-device: the cloud tier has no VLM")

        request = _Request(
            guard=guard,
            query=query,
            context=context,
            notes=notes,
            pii_detected=n_detected,
            view=self._local_view(guard, self.fast_brain, query, notes),
            image=image,
        )

        # 2. Decide.
        local_latency_est = self.policy.estimate_local_latency_ms(self.local.profile, query)

        # An image-bearing query always takes the heuristic path below, even
        # when the fast brain self-rates. That is not a limitation of Shape B/C
        # so much as the fact that neither shape has anywhere to *put* an image:
        # both call `answer(text)` with no image argument, so routing an image
        # through them would silently drop it -- a latent bug that only became
        # reachable when `GpuLocalBrain` (the one vision-capable brain here)
        # started self-rating. The heuristic path handles images correctly
        # today, so image behaviour is unchanged. Extending Shape C to images is
        # the deliberate next step, not an oversight.
        # A forced tier short-circuits every routing shape below -- the
        # structured and confidence paths included, since "let the brain decide"
        # is exactly what forcing overrides. It changes *which brain answers*
        # and nothing else: the local view was already built above, so masking
        # and the invariant have run, and an escalated image is still described
        # on-device with its description masked.
        if force_tier is not None:
            difficulty = self.difficulty.score(query)
            would_be = (
                "cloud" if self.policy.should_escalate(difficulty, local_latency_est) else "local"
            )
            if ui_test_enabled():
                notes.append(
                    f"tier forced to {force_tier} (UI_TEST=1; policy would have said {would_be})"
                )
                if force_tier == "cloud":
                    return self._escalate(request, difficulty)
                return self._answer_locally(request, difficulty)
            notes.append(
                f"ignored force_tier={force_tier}: UI_TEST is not enabled, so the policy decides"
            )

        if image is None:
            if getattr(self.fast_brain, "reports_gaps", False):
                return self._route_on_structured_answer(request, local_latency_est)
            if self.fast_brain.reports_confidence:
                return self._route_on_confidence(request, local_latency_est)

        difficulty = self.difficulty.score(query)
        if self.policy.should_escalate(difficulty, local_latency_est):
            notes.append(self.policy.escalation_note(difficulty, local_latency_est))
            return self._escalate(request, difficulty)

        notes.append(self.policy.local_note(difficulty, local_latency_est))
        return self._answer_locally(request, difficulty)

    def _route_on_confidence(self, request: _Request, local_latency_est: float) -> RouteDecision:
        """Ask the fast brain first, then route on the confidence it returns.

        Used when the tier's brain self-rates but cannot name gaps -- today
        that is `PhoneFastBrain` alone. The answer and the confidence arrive
        together (one inference, not two -- see `signals/confidence.py`), which
        means the brain is asked *before* the local-vs-cloud decision and its
        answer is discarded if the decision goes elsewhere. That cost is the
        accepted trade for a real signal instead of a keyword heuristic -- and
        it is paid twice on the "not confident" path when an escalation brain
        is configured (see `_route_away_from_fast_brain`): two real local
        inferences on one query, deliberately not optimized for latency.

        Two details that are easy to get wrong, both load-bearing:

        - **The latency budget is checked before the call, not after.** Its job
          is to avoid *starting* a local inference that cannot finish in time.
          Re-applying it once the answer is already in hand would be actively
          harmful: escalating at that point adds the next brain's latency on
          top of the local time already spent, so it can only make the total
          worse.
        - **This brain is off-device, so it was given masked text.**
          `request.view` was built with `trusted_with_raw_pii = False` for
          `PhoneFastBrain`, so its answer comes back in masked space and
          `_answer_locally` rehydrates it.
        """
        if self.policy.should_escalate(_NO_DIFFICULTY_SIGNAL_YET, local_latency_est):
            difficulty = 1.0  # no signal was even attempted -- not a heuristic score
            request.notes.append(
                f"skipped the fast brain: its profiled latency estimate "
                f"({local_latency_est:.0f}ms) already exceeds the "
                f"{self.policy.local_latency_budget_ms:.0f}ms budget, so a local "
                f"answer would have been discarded anyway"
            )
            request.notes.append(self.policy.escalation_note(difficulty, local_latency_est))
            return self._route_away_from_fast_brain(request, difficulty)

        local = self._ask(self.fast_brain, "fast brain", request.view.text, vault=request.view.vault)
        difficulty = self._difficulty_from(local, request.notes)

        if self.policy.should_escalate(difficulty, _BUDGET_ALREADY_CHECKED):
            # Not `policy.escalation_note`: that one describes both terms of the
            # OR, and quoting a latency-vs-budget comparison here would be
            # misleading -- the budget was settled before the call and cannot be
            # what fired.
            request.notes.append(
                f"escalating: difficulty={difficulty:.2f} >= threshold "
                f"{self.policy.escalate_threshold} "
                f"(the local answer took {local.latency_ms:.0f}ms and was not used)"
            )
            return self._route_away_from_fast_brain(request, difficulty, discarded=local)

        request.notes.append(self.policy.local_note(difficulty, local.latency_ms))
        return self._answer_locally(request, difficulty, response=local)

    def _route_on_structured_answer(self, request: _Request, local_latency_est: float) -> RouteDecision:
        """Shape C: ask the local brain to solve what it can and name what it
        can't, then mask and send only what it named.

        The difference from `_route_on_confidence` is what happens on a "not
        confident" verdict. There, the local answer is *discarded* and some
        other brain redoes the whole query. Here it is kept and shown, and the
        deep brain is asked a narrower question -- the one the local model
        wrote out itself.

        Both brains that take this path are on-device, so the model that
        produced the `solution` and the `unknown` saw the raw query. That makes
        `_answer_hybrid`'s masking of those two strings the load-bearing step in
        the whole feature: they are the only things here that cross, and they
        are raw until it masks them.

        Three outcomes:

        - **local** -- confident, and no gap named. Nothing is masked because
          nothing leaves.
        - **hybrid** -- there is something to hand on *and* a usable partial
          answer to keep.
        - **cloud** -- there is something to hand on but no usable partial (the
          brain errored, or returned an empty solution). Splitting requires two
          halves; with one, this is an ordinary escalation and is reported as
          one.
        """
        # Shape B's budget pre-check is deliberately *not* reused here, and the
        # reason is the premise it rests on: "a local answer would have been
        # discarded anyway". In Shape C it would not be -- a usable local answer
        # is always kept and shown. Reusing the 3000ms budget measurably broke
        # this feature on real hardware: two of the four demo queries skipped
        # the fast brain entirely, including the PII one, which the local model
        # then turned out to answer fully on-device. So the ceiling here is
        # `local_partial_budget_ms`, a runaway guard rather than a preference.
        if local_latency_est > self.policy.local_partial_budget_ms:
            difficulty = 1.0  # no signal was even attempted -- not a heuristic score
            request.notes.append(
                f"skipped the fast brain: its profiled latency estimate "
                f"({local_latency_est:.0f}ms) exceeds even the "
                f"{self.policy.local_partial_budget_ms:.0f}ms ceiling for keeping a "
                f"partial answer"
            )
            request.notes.append(self.policy.escalation_note(difficulty, local_latency_est))
            return self._route_away_from_fast_brain(request, difficulty)

        if local_latency_est > self.policy.local_latency_budget_ms:
            # Worth saying out loud rather than passing silently: this query is
            # over the budget that Shape B would have escalated on, and is being
            # answered locally anyway because the answer will be kept.
            request.notes.append(
                f"over the {self.policy.local_latency_budget_ms:.0f}ms fast-path budget "
                f"(estimate {local_latency_est:.0f}ms) but asking the fast brain anyway -- "
                f"in a split, its answer is kept rather than discarded"
            )

        local = self._ask(self.fast_brain, "fast brain", request.view.text, vault=request.view.vault)
        difficulty = self._difficulty_from(local, request.notes)

        gap = local.unknown.strip()
        if gap:
            request.notes.append(f"fast brain named what it could not answer: {gap!r}")
        if local.model_masked_output:
            # Recorded, never acted on. See signals/structured.py -- masking is
            # this router's job and happens at the boundary, so a model that was
            # handed the raw text is the last thing that should be deciding what
            # a masked version of it looks like.
            request.notes.append(
                "ignored the model's own masked_output field: masking happens at "
                "the cloud boundary, not by delegation to the model"
            )

        if not self.policy.needs_gap_fill(difficulty, gap):
            request.notes.append(self.policy.local_note(difficulty, local.latency_ms))
            return self._answer_locally(request, difficulty, response=local)

        if not local.text.strip():
            request.notes.append(
                "no usable partial answer to keep -- escalating the whole query "
                "rather than reporting a split that only has one half"
            )
            return self._route_away_from_fast_brain(request, difficulty, discarded=local)

        return self._answer_hybrid(request, difficulty, local, gap)

    @staticmethod
    def _difficulty_from(local: BrainResponse, notes: list[str]) -> float:
        """Turn a self-rating brain's response into a difficulty, with notes.

        Shared by Shapes B and C so the two cannot drift on the one rule that
        matters most here: **an unparseable confidence is `1.0`, not a fall-back
        to a different signal.** A brain that formats badly is not the same
        claim as "the surface features say this is hard"; conflating them would
        score the same query two different ways depending on an unrelated
        formatting accident. See docs/ORCHESTRATOR.md.
        """
        if local.error:
            notes.append(f"fast brain reported a problem: {local.error}")
        if local.confidence is None:
            notes.append("fast brain returned no usable confidence signal")
            return 1.0
        difficulty = confidence_to_difficulty(local.confidence)
        notes.append(
            f"fast brain self-reported confidence={local.confidence:.2f} "
            f"-> difficulty={difficulty:.2f}"
        )
        return difficulty

    def _route_away_from_fast_brain(
        self, request: _Request, difficulty: float, discarded: BrainResponse | None = None
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
            return self._answer_via_escalation_brain(request, difficulty, discarded)
        return self._escalate(request, difficulty, discarded=discarded)

    def _answer_via_escalation_brain(
        self, request: _Request, difficulty: float, discarded: BrainResponse | None
    ) -> RouteDecision:
        """A second opinion from `self.escalation_brain` (the AI PC's real
        model) instead of the cloud.

        Still `tier_answered="local"`: the escalation brain runs on this
        machine (docs/npu-deployment.md), so nothing here crosses the boundary
        `CloudDeepBrain` represents -- the privacy invariant this project is
        built around is specifically about *that* boundary, and this path never
        reaches it. Which also means this brain gets its own `_local_view`
        rather than inheriting the phone's: the AI PC is trusted with the raw
        query even though the phone that just failed on it was not.

        `discarded` is `None` when the fast brain was never even called (the
        budget pre-check fired); otherwise it is billed into the reported
        latency, same accounting `_escalate` already does for a discarded local
        answer.
        """
        request.notes.append(
            "not confident enough -- getting a second opinion from the AI PC's "
            "own model instead of falling back to a heuristic or escalating "
            "to the cloud"
        )
        view = self._local_view(request.guard, self.escalation_brain, request.query, request.notes)
        response = self._ask(self.escalation_brain, "escalation brain", view.text, vault=view.vault)

        est_latency_ms = response.latency_ms
        if discarded is not None:
            est_latency_ms += discarded.latency_ms
            request.notes.append(
                f"AI PC answered in {response.latency_ms:.0f}ms; reported latency "
                f"also includes the phone's discarded {discarded.latency_ms:.0f}ms"
            )

        return RouteDecision(
            tier_answered="local",
            difficulty_score=difficulty,
            est_latency_ms=est_latency_ms,
            est_cost_usd=response.cost_usd,
            pii_entities_detected=request.pii_detected,
            pii_entities_masked=len(view.vault),
            answer=request.guard.rehydrate(response.text, view.vault),
            notes=request.notes,
        )

    def _answer_locally(
        self, request: _Request, difficulty: float, response: BrainResponse | None = None
    ) -> RouteDecision:
        # `response` is already populated on the confidence/structured paths --
        # reusing it is what keeps those to a single inference call. An image is
        # only ever present on the heuristic path (see route()), where no
        # response has been produced yet.
        if response is None:
            if request.image is not None:
                # The image never left the device, so the local VLM sees it directly.
                response = self._ask(
                    self.fast_brain, "fast brain", request.view.text,
                    image=request.image, vault=request.view.vault,
                )
            else:
                response = self._ask(
                    self.fast_brain, "fast brain", request.view.text, vault=request.view.vault
                )
        # Rehydration is a no-op for a trusted brain (empty vault): its answer
        # is already in raw space because its input was.
        return RouteDecision(
            tier_answered="local",
            difficulty_score=difficulty,
            est_latency_ms=response.latency_ms,
            est_cost_usd=response.cost_usd,
            pii_entities_detected=request.pii_detected,
            pii_entities_masked=len(request.view.vault),
            answer=request.guard.rehydrate(response.text, request.view.vault),
            notes=request.notes,
        )

    def _answer_hybrid(
        self, request: _Request, difficulty: float, local: BrainResponse, gap: str
    ) -> RouteDecision:
        """Keep the local partial answer, and send the deep brain only the gap.

        The masking here is the part worth reading closely, and it is the whole
        privacy story of Shape C. `gap` and `local.text` were produced by a
        model that was handed the **raw** query, so they can contain the user's
        real email address verbatim -- not a placeholder. They are the only
        strings on this path that cross the boundary besides the query itself,
        and they are raw right up until `mask_for_boundary` here. Getting this
        wrong would leak PII that the original query masking would never have
        caught, because these strings did not exist when the query was read.

        Ordering inside the context is load-bearing too:
        `RoutePolicy.compress_context` keeps the *tail*, so the gap instruction
        goes last and survives truncation. The partial answer is the part that
        gets trimmed when there is too much, which is the right thing to lose --
        it is an optimisation for answer quality, while the gap is the entire
        reason this call is being made.
        """
        masked_query = request.mask_for_boundary(request.query)
        masked_gap = request.mask_for_boundary(gap)

        vault = dict(masked_query.vault)
        vault.update(masked_gap.vault)

        parts: list[str] = []
        if request.context:
            masked_context = request.mask_for_boundary(request.context)
            vault.update(masked_context.vault)
            parts.append(masked_context.masked_text)
        if self.policy.send_partial_to_cloud:
            masked_partial = request.mask_for_boundary(local.text)
            vault.update(masked_partial.vault)
            parts.append(
                "A smaller on-device model has already answered part of this "
                f"question: {masked_partial.masked_text}"
            )
        parts.append(f"Answer only the remaining part it could not: {masked_gap.masked_text}")

        compressed, was_compressed = self.policy.compress_context("\n\n".join(parts))
        if was_compressed:
            request.notes.append(f"compressed the escalated gap context to {len(compressed)} chars")
        if masked_query.vault:
            request.notes.append(f"sent off-device (masked): {masked_query.masked_text!r}")
        request.notes.append(
            "splitting the query: keeping the on-device answer and asking the "
            "deep brain only about the gap"
            + ("" if self.policy.send_partial_to_cloud else " (partial answer withheld)")
        )

        response = self._ask(
            self.deep_brain, "deep brain", masked_query.masked_text, compressed,
            crossing=True, vault=vault,
        )
        self.tracer.rehydrated(vault)

        # Rehydrate only now, on-device, after the answer is back. The local
        # half needs no rehydration at all when the brain was trusted -- it was
        # never masked -- which the empty vault makes a no-op rather than a
        # special case.
        local_answer = request.guard.rehydrate(local.text, request.view.vault)
        cloud_answer = request.guard.rehydrate(response.text, vault)
        return RouteDecision(
            tier_answered="hybrid",
            difficulty_score=difficulty,
            # Both calls really happened and the user waited for both, so both
            # are billed -- same accounting as a discarded local answer.
            est_latency_ms=local.latency_ms + response.latency_ms,
            est_cost_usd=local.cost_usd + response.cost_usd,
            pii_entities_detected=request.pii_detected,
            pii_entities_masked=len(masked_query.vault),
            answer=f"{local_answer}\n\n{cloud_answer}",
            notes=request.notes,
            local_answer=local_answer,
            cloud_answer=cloud_answer,
            gap=request.guard.rehydrate(masked_gap.masked_text, vault),
        )

    def route_stream(
        self,
        query: str,
        context: str = "",
        image: Path | None = None,
        force_tier: Tier2 | None = None,
    ):
        """Route one query, yielding the answer as it is generated.

        Yields `("meta", {...})` once the tier is decided, then `("delta", str)`
        repeatedly, then `("done", {...})`. Used by the chat UI: a local reply
        runs at ~21 tok/s, so a long answer is otherwise a minute of blank
        screen.

        Everything that decides *where* text goes still happens before the first
        delta -- detection, the local view, and, for an escalated image,
        describing it on-device and masking that description. Only generation is
        incremental, and this keeps the same boundary discipline as `_escalate`:
        what crosses is masked from the raw query at the point of crossing.

        Deliberately does not implement the structured or confidence shapes.
        Those ask the brain first and route on what comes back, which cannot be
        streamed without either showing an answer that may be discarded or
        buffering the whole thing and defeating the point. They fall through to
        the heuristic path here; `route()` remains the full implementation.
        """
        notes: list[str] = []
        guard = PIIGuard()
        n_detected = self._n(query, guard)

        if image is not None and not getattr(self.fast_brain, "can_see", False):
            raise ValueError(
                "an image was supplied but the fast brain cannot see -- "
                "enable TWO_BRAIN_GPU_BRAIN=1 and build it with GpuLocalBrain.for_vision()"
            )
        if image is not None:
            notes.append("image stays on-device: the cloud tier has no VLM")

        request = _Request(
            guard=guard,
            query=query,
            context=context,
            notes=notes,
            pii_detected=n_detected,
            view=self._local_view(guard, self.fast_brain, query, notes),
            image=image,
        )

        difficulty = self.difficulty.score(query)
        local_latency_est = self.policy.estimate_local_latency_ms(self.local.profile, query)
        would_be = "cloud" if self.policy.should_escalate(difficulty, local_latency_est) else "local"

        if force_tier is not None and ui_test_enabled():
            notes.append(f"tier forced to {force_tier} (UI_TEST=1; policy would have said {would_be})")
            escalate = force_tier == "cloud"
        else:
            if force_tier is not None:
                notes.append(
                    f"ignored force_tier={force_tier}: UI_TEST is not enabled, so the policy decides"
                )
            escalate = would_be == "cloud"
            notes.append(
                self.policy.escalation_note(difficulty, local_latency_est)
                if escalate
                else self.policy.local_note(difficulty, local_latency_est)
            )

        if escalate:
            masked_query = request.mask_for_boundary(request.query)
            vault = dict(masked_query.vault)
            ctx = request.context
            if image is not None:
                described = self.fast_brain.describe_image(image, request.view.text)
                masked_description = request.mask_for_boundary(described.text)
                vault.update(masked_description.vault)
                ctx = f"{ctx}\n\n{masked_description.masked_text}".strip() if ctx else masked_description.masked_text
                notes.append(
                    f"image described on-device into {len(masked_description.masked_text)} chars; "
                    f"masked {len(masked_description.vault)} PII entit"
                    f"{'y' if len(masked_description.vault) == 1 else 'ies'} in the description"
                )
                masked_context = MaskResult(masked_text=ctx, vault={})
            else:
                masked_context = (
                    request.mask_for_boundary(ctx) if ctx else MaskResult(masked_text="", vault={})
                )
                vault.update(masked_context.vault)
            compressed, was_compressed = self.policy.compress_context(masked_context.masked_text)
            if was_compressed:
                notes.append(f"compressed escalated context to {len(compressed)} chars")
            if masked_query.vault:
                notes.append(f"sent off-device (masked): {masked_query.masked_text!r}")
            stream = self.deep_brain.answer_stream(masked_query.masked_text, compressed)
        else:
            # A trusted on-device brain gets the raw query and an empty vault,
            # so the rehydrate below is a no-op for it. An untrusted one gets
            # masked text and its vault, exactly as `_answer_locally` does.
            vault = dict(request.view.vault)
            kwargs = {"image": image} if image is not None else {}
            stream = self.fast_brain.answer_stream(request.view.text, **kwargs)

        tier = "cloud" if escalate else "local"
        yield ("meta", {
            "tier_answered": tier,
            "difficulty_score": difficulty,
            "pii_entities_masked": n_detected if escalate else len(request.view.vault),
            "notes": notes,
        })

        rehydrator = _IncrementalRehydrator(guard, vault)
        started = time.perf_counter()
        for piece in stream:
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
        })

    def _escalate(
        self, request: _Request, difficulty: float, discarded: BrainResponse | None = None
    ) -> RouteDecision:
        # 3. The boundary. Everything below is masked from the raw query at the
        # point of crossing, rather than having been masked on the way in.
        masked_query = request.mask_for_boundary(request.query)
        vault = dict(masked_query.vault)
        context = request.context

        # 3b. An image cannot cross the boundary -- the deep brain is a text-only
        # LLM with no vision support at all. So the local VLM converts it to
        # words here, on-device, and only those words are eligible to leave.
        # The description is steered by the query: a physics diagram needs the
        # mechanical arrangement, "what is this?" needs identification.
        if request.image is not None:
            described = self.fast_brain.describe_image(request.image, request.view.text)
            # The description is newly generated text that has never been
            # masked. It can easily contain PII the query did not -- a name on
            # a document, an address on a sign, a face described in words -- so
            # it is masked exactly like any other text before it can escalate,
            # and the invariant is asserted on it too.
            masked_description = request.mask_for_boundary(described.text)
            vault.update(masked_description.vault)
            context = (
                f"{context}\n\n{masked_description.masked_text}".strip()
                if context
                else masked_description.masked_text
            )
            request.notes.append(
                f"image described on-device into {len(masked_description.masked_text)} chars; "
                f"masked {len(masked_description.vault)} PII entit"
                f"{'y' if len(masked_description.vault) == 1 else 'ies'} in the description"
            )
            # Already masked above; re-masking placeholders is a no-op, and
            # skipping it here keeps a single masked string rather than two.
            masked_context = MaskResult(masked_text=context, vault={})
        else:
            masked_context = (
                request.mask_for_boundary(context) if context else MaskResult(masked_text="", vault={})
            )
            vault.update(masked_context.vault)

        # 4. Context crosses the boundary too, so it is masked and compressed.
        compressed, was_compressed = self.policy.compress_context(masked_context.masked_text)
        if was_compressed:
            request.notes.append(f"compressed escalated context to {len(compressed)} chars")
        if masked_query.vault:
            request.notes.append(f"sent off-device (masked): {masked_query.masked_text!r}")

        response = self._ask(
            self.deep_brain, "deep brain", masked_query.masked_text, compressed,
            crossing=True, vault=vault,
        )
        self.tracer.rehydrated(vault)

        # A speculative local answer that lost is still time the user waited
        # for, so the reported latency includes it. Hiding it would make the
        # confidence path look free when it is not.
        est_latency_ms = response.latency_ms
        if discarded is not None:
            est_latency_ms += discarded.latency_ms
            request.notes.append(
                f"discarded the local answer after {discarded.latency_ms:.0f}ms; "
                f"reported latency includes it"
            )

        # 5. Rehydrate only now, on-device, after the answer is back.
        return RouteDecision(
            tier_answered="cloud",
            difficulty_score=difficulty,
            est_latency_ms=est_latency_ms,
            est_cost_usd=response.cost_usd,
            pii_entities_detected=request.pii_detected,
            pii_entities_masked=len(masked_query.vault),
            answer=request.guard.rehydrate(response.text, vault),
            notes=request.notes,
        )
