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

import dataclasses
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from two_brain_router.privacy import MaskResult, PIIGuard, assert_masked_token_invariant
from two_brain_router.routing.brains import (
    Brain,
    _to_brain_response,
    BrainResponse,
    CirrascaleDeepBrain,
    CloudDeepBrain,
    GpuLocalBrain,
    LocalFastBrain,
    NpuFastBrain,
    PhoneFastBrain,
)
from two_brain_router.routing.policy import RouteDecision, RoutePolicy, RouteProgress
from two_brain_router.signals import DifficultyEstimator, TierSignals
from two_brain_router.signals.confidence import confidence_to_difficulty
from two_brain_router.trace import Tracer, trace_enabled

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


def _unique_entities(hits: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """De-duplicate `(type, value)` detections by value, keeping first order.

    Detection counts *occurrences*; the vault keys on *values*, so one address
    written twice masks to one placeholder. Reporting raw occurrences against a
    vault-derived masked count would make a request that mentions the same
    email in both the query and the history read as "2 detected, 1 masked" --
    which looks like a leak and isn't one.
    """
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for entity_type, value in hits:
        if value not in seen:
            seen.add(value)
            unique.append((entity_type, value))
    return unique


def _deltas_for(decision: RouteDecision):
    """Emit a finished decision as tier-attributed deltas.

    Used by `route_stream`'s non-streaming fallbacks. A split still yields
    **two** deltas, one per brain, rather than one tagged `"hybrid"` -- the UI
    opens a bubble per tier it hears from, so a single `"hybrid"` delta
    collapses two brains' answers into one anonymous bubble and loses exactly
    the attribution the split exists to show. The rendering path is then
    identical whether or not the configured brains happen to stream.
    """
    if decision.tier_answered == "hybrid":
        if decision.local_answer:
            yield ("delta", {"text": decision.local_answer, "tier": "local"})
        if decision.gap:
            yield ("tier", {"tier": "cloud", "gap": decision.gap})
        if decision.cloud_answer:
            yield ("delta", {"text": decision.cloud_answer, "tier": "cloud"})
        return
    yield ("delta", {"text": decision.answer, "tier": decision.tier_answered})


def _crossing_payload(masked_query: str, context: str, vault: dict[str, str]) -> dict:
    """What crossed the boundary, shaped for display.

    `vault` maps placeholder -> original value, so this inverts it into the
    direction a person reads: the thing they typed, and what it became. Typed
    from the placeholder rather than re-detected, so the record can only ever
    describe substitutions that actually happened.
    """
    subs = []
    for placeholder, value in vault.items():
        kind = placeholder.strip("[]").removeprefix("PII_").rsplit("_", 1)[0]
        subs.append({"type": kind, "value": value, "placeholder": placeholder})
    return {"query": masked_query, "context": context, "substitutions": subs}


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

    text: str = ""
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
    #: The same treatment for `context` (conversation history, usually) --
    #: raw for a trusted brain, masked for one that isn't. Separate from `view`
    #: because the two are masked independently but with the *same* guard, so
    #: their vaults share placeholder numbering and can be merged safely.
    context_view: _LocalView = field(default_factory=_LocalView)
    image: Path | None = None
    #: Called with a `RouteProgress` when the local half is settled and the
    #: cloud call is about to start, so a UI can show that half immediately
    #: instead of waiting out a 15s+ cloud leg. Optional; `route()` behaves
    #: identically without it.
    on_progress: Callable[[RouteProgress], None] | None = None

    @property
    def local_vault(self) -> dict[str, str]:
        """Everything the fast brain's answer might need rehydrating from.

        A brain given masked query *and* masked context can echo a placeholder
        from either, so rehydration has to consider both. Safe to merge: one
        `PIIGuard` masked both, and it maps a repeated value to the same
        placeholder, so the two dicts cannot disagree.
        """
        if not self.context_view.vault:
            return self.view.vault
        return {**self.view.vault, **self.context_view.vault}

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
        #: Which catalogue entry `fast_brain` is, when it came from one.
        #: None for an env-var-built brain -- `use_model` sets it, and the
        #: no-op-if-unchanged check reads it.
        self.model_id: str | None = None
        self.escalation_brain = _build_escalation_brain(tier, self.fast_brain)
        self.deep_brain = _build_deep_brain(self.cloud)

    def use_model(self, model_id: str) -> None:
        """Swap the fast brain to the given catalogue model.

        **Closes the current brain before opening the new one, always.** Not an
        optimisation -- a requirement: this hardware refuses a second live Genie
        dialog session (`GENIE_STATUS_ERROR... err 1002`, see
        `tests/test_npu_brain.py`'s fixture), and two `llama-server` children
        would each hold GBs of VRAM. So a switch is genuinely serial, and costs
        a cold load: ~12s for the NPU, ~17s for the 8B GPU model.

        On failure the router is left with **no** fast brain rather than a
        half-swapped one, and the exception propagates -- a caller that asked
        for a model it cannot have should hear about it, not silently keep
        answering from the previous one and wonder why the numbers look
        familiar.
        """
        from two_brain_router.routing.models import BY_ID, build_brain

        if model_id == self.model_id:
            return
        model = BY_ID.get(model_id)
        if model is None:
            raise ValueError(f"unknown model {model_id!r}")

        old, self.fast_brain, self.model_id = self.fast_brain, None, None
        close = getattr(old, "close", None)
        if callable(close):
            close()
        self.fast_brain = build_brain(model, self.tier, self.local)
        self.model_id = model_id

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

    def _local_context_view(self, guard: PIIGuard, brain: object, context: str) -> _LocalView:
        """`_local_view` for the context (conversation history), minus the note.

        Same trust rule as the query -- raw for an on-device brain, masked for
        one across a boundary -- and masked with the *same* guard, so a value
        appearing in both the history and the query gets one placeholder and the
        vaults merge cleanly (`_Request.local_vault`).

        No audit note of its own: `_local_view` already recorded which side of
        the boundary this brain sits on, and repeating it per-field would bury
        the routing decision in bookkeeping.
        """
        if not context:
            return _LocalView()
        if _trusted_with_raw_pii(brain):
            return _LocalView(text=context)
        masked = guard.mask(context)
        assert_masked_token_invariant(context, masked)
        return _LocalView(text=masked.masked_text, vault=masked.vault)

    @staticmethod
    def _n(query: str, guard: PIIGuard) -> int:
        return len(guard.detect(query))

    @staticmethod
    def _emit_progress(request: _Request, progress: RouteProgress) -> None:
        """Hand a partial result to the caller, if one asked for partials.

        Exceptions from the callback are swallowed on purpose. This is a
        *notification*, and the commonest way it fails is a streaming client
        disconnecting mid-request -- which must not change the routing outcome,
        corrupt the audit trail, or turn a working answer into a 500. The
        request finishes normally and the final `RouteDecision` is still
        correct; nobody is just listening any more.
        """
        if request.on_progress is None:
            return
        try:
            request.on_progress(progress)
        except Exception:  # noqa: BLE001 -- a dead listener is not a routing failure
            pass

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
        on_progress: Callable[[RouteProgress], None] | None = None,
    ) -> RouteDecision:
        """Route one query. The real work is `_route`; this traces the outcome.

        Split so that *every* return path is traced, including ones added
        later -- `_route` has four `return RouteDecision(...)` sites and a fifth
        would otherwise be silently untraced.

        `on_progress`, when given, is called once with a `RouteProgress` at the
        moment the local half is settled and the deep brain is about to be
        asked. Purely additive: the returned `RouteDecision` is identical
        either way, and callers that don't pass it see no change at all.
        """
        decision = self._route(query, context, image, on_progress)
        self.tracer.decision(decision)
        return decision

    def route_stream(self, query: str, context: str = "", image: Path | None = None):
        """Route one query, yielding text as each brain generates it.

        Yields `(kind, payload)` pairs:

            ("meta",  {...})            once, before any generation
            ("delta", {"text", "tier"}) repeatedly, tier-attributed
            ("done",  {...})            once, the full RouteDecision as a dict

        **Only the `solution` field is streamed, never the JSON around it.**
        A Shape C brain emits `{"solution": "...", "confidence": 95, "unknown":
        "..."}` token by token; forwarding that raw would show the user a brace,
        a quoted key and a colon before any answer, then routing metadata after
        it. `SolutionStreamer` walks the buffer and releases only the decoded
        contents of `solution`, while the raw text is kept intact for the real
        parse at the end. (The `(hollowbyte)-feat/chat_app` branch declined to
        stream the local half at all for exactly this reason.)

        **Every delta is tier-attributed**, so a caller renders two bubbles from
        the data rather than inferring attribution from ordering.

        The privacy ordering is unchanged and still comes first: detection, the
        local view, and -- for an escalated image -- describing it on-device and
        masking that description, all happen before the first delta. Only
        *generation* is incremental. What crosses is still masked from the raw
        query at the point of crossing, and the deep brain is still asked the
        gap rather than the whole query.

        Falls back to `route()` and emits its answer as a single delta when the
        tier's brain cannot stream (`streams_tokens`) or takes a shape this does
        not implement -- Shape B, which discards its local answer when it routes
        away, so streaming it would mean showing text about to be retracted.
        """
        from two_brain_router.signals.structured import SolutionStreamer

        fast = self.fast_brain
        can_stream = (
            getattr(fast, "streams_tokens", False)
            and getattr(fast, "reports_gaps", False)
            and image is None
        )
        if not can_stream:
            # One delta, then done: the caller's rendering path stays identical
            # whether or not the configured brains happen to support streaming.
            decision = self.route(query, context, image)
            yield ("meta", {"tier": decision.tier_answered, "streaming": False})
            if decision.crossed_to_cloud:
                yield ("crossing", decision.crossed_to_cloud)
            yield from _deltas_for(decision)
            yield ("done", dataclasses.asdict(decision))
            return

        notes: list[str] = []
        guard = PIIGuard()
        detected = _unique_entities(
            guard.detect(query) + (guard.detect(context) if context else [])
        )
        self.tracer.request(self.tier, query, context, image, detected)
        if detected:
            notes.append(
                f"detected {len(detected)} PII "
                f"entit{'y' if len(detected) == 1 else 'ies'} in the query"
            )
        request = _Request(
            guard=guard,
            query=query,
            context=context,
            notes=notes,
            pii_detected=len(detected),
            view=self._local_view(guard, fast, query, notes),
            context_view=self._local_context_view(guard, fast, context),
        )
        if context:
            notes.append(f"carrying {len(context)} chars of conversation context")

        local_latency_est = self.policy.estimate_local_latency_ms(self.local.profile, query)
        if local_latency_est > self.policy.local_partial_budget_ms:
            # Same ceiling as `_route_on_structured_answer`; nothing local is
            # worth starting past it, so there is nothing to stream.
            decision = self.route(query, context, image)
            yield ("meta", {"tier": decision.tier_answered, "streaming": False})
            if decision.crossed_to_cloud:
                yield ("crossing", decision.crossed_to_cloud)
            yield from _deltas_for(decision)
            yield ("done", dataclasses.asdict(decision))
            return

        yield ("meta", {"tier": "local", "streaming": True, "pii_detected": len(detected)})

        streamer = SolutionStreamer()
        self.tracer.call("fast brain", fast, request.view.text, request.context_view.text,
                         crossing=False, vault=request.local_vault)
        start = time.perf_counter()
        for piece in fast.answer_stream(request.view.text, request.context_view.text):
            visible = streamer.feed(piece)
            if visible:
                # Rehydrated per-delta: for an untrusted brain the text is in
                # masked space, and a placeholder must never reach the screen.
                yield ("delta", {
                    "text": request.guard.rehydrate(visible, request.local_vault),
                    "tier": "local",
                })
        tail = streamer.finish()
        if tail:
            yield ("delta", {
                "text": request.guard.rehydrate(tail, request.local_vault),
                "tier": "local",
            })
        local_latency_ms = (time.perf_counter() - start) * 1000

        local = _to_brain_response(streamer.raw.strip(), local_latency_ms, structured=True)
        self.tracer.result(fast, local, vault=request.local_vault)
        difficulty = self._difficulty_from(local, notes)
        gap = local.unknown.strip()
        if gap:
            notes.append(f"fast brain named what it could not answer: {gap!r}")

        if not self.policy.needs_gap_fill(difficulty, gap) or not gap or not local.text.strip():
            # Stayed local, or has no gap to hand on. Either way nothing more is
            # generated -- and when there is no usable split, `route()` would
            # have escalated the *whole* query, which cannot reuse the text
            # already streamed. Fall back so the audit trail stays truthful.
            if self.policy.needs_gap_fill(difficulty, gap):
                decision = self.route(query, context, image)
                yield ("meta", {"tier": decision.tier_answered, "restarted": True})
                if decision.crossed_to_cloud:
                    yield ("crossing", decision.crossed_to_cloud)
                yield from _deltas_for(decision)
                yield ("done", dataclasses.asdict(decision))
                return
            notes.append(self.policy.local_note(difficulty, local.latency_ms))
            decision = self._answer_locally(request, difficulty, response=local)
            self.tracer.decision(decision)
            yield ("done", dataclasses.asdict(decision))
            return

        # A split: the cloud answers the gap, and its tokens open a second
        # bubble. `_answer_hybrid` does the masking, the boundary assert and the
        # cloud call; streaming its half is the only difference here.
        yield ("tier", {"tier": "cloud", "gap": gap})
        # `yield from` on a generator that `return`s: the deltas flow to the
        # caller and the finished decision comes back here (PEP 380), so the
        # streaming half needs no out-parameter or mutable holder.
        decision = yield from self._answer_hybrid_streaming(request, difficulty, local, gap)
        self.tracer.decision(decision)
        yield ("done", dataclasses.asdict(decision))

    def _route(
        self,
        query: str,
        context: str = "",
        image: Path | None = None,
        on_progress: Callable[[RouteProgress], None] | None = None,
    ) -> RouteDecision:
        notes: list[str] = []
        guard = PIIGuard()

        # 1. Detect, do not mask. What the request contained is recorded from
        #    the start; whether any of it gets masked depends on where it goes.
        #
        #    Context is scanned too, not just the query. Conversation history is
        #    real user text and can carry PII the current query does not -- and
        #    it crosses the boundary on an escalation exactly like the query
        #    does. Counting only the query reported "0 detected" for a request
        #    that then masked an email out of the history, which understates the
        #    thing this sample exists to show.
        detected = _unique_entities(guard.detect(query) + (guard.detect(context) if context else []))
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
            context_view=self._local_context_view(guard, self.fast_brain, context),
            image=image,
            on_progress=on_progress,
        )
        if context:
            notes.append(f"carrying {len(context)} chars of conversation context")

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

        local = self._ask(
            self.fast_brain, "fast brain", request.view.text,
            request.context_view.text, vault=request.local_vault,
        )
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

        local = self._ask(
            self.fast_brain, "fast brain", request.view.text,
            request.context_view.text, vault=request.local_vault,
        )
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

        if not gap:
            # Escalating on low confidence alone, with no gap named. There is
            # nothing gap-shaped to ask about, so this is an ordinary
            # escalation: the whole query goes and the local answer is
            # discarded, rather than shown beside a full cloud answer that
            # covers the same ground.
            #
            # Load-bearing, not tidiness. `_answer_hybrid` sends the gap as the
            # deep brain's *query*, so reaching it with an empty gap asks the
            # cloud an empty question. Observed on real hardware: the NPU
            # returned `confidence 0.00` with `unknown` empty for a query it
            # simply declined, which lands exactly here. The pre-fix code hid
            # this because it sent the whole query as the ask regardless.
            request.notes.append(
                "not confident, and no specific gap named -- escalating the "
                "whole query rather than splitting on nothing"
            )
            return self._route_away_from_fast_brain(request, difficulty, discarded=local)

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
        # Its own views, not the fast brain's: the AI PC is trusted with the raw
        # query even though the phone that just failed on it was not.
        view = self._local_view(request.guard, self.escalation_brain, request.query, request.notes)
        context_view = self._local_context_view(
            request.guard, self.escalation_brain, request.context
        )
        vault = {**view.vault, **context_view.vault}
        response = self._ask(
            self.escalation_brain, "escalation brain", view.text, context_view.text, vault=vault
        )

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
            pii_entities_masked=len(vault),
            answer=request.guard.rehydrate(response.text, vault),
            notes=request.notes,
            # No `crossed_to_cloud`: the escalation brain runs on this machine,
            # so nothing crossed the boundary this field describes. Leaving it
            # None is the claim, not an omission.
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
                    request.context_view.text, image=request.image,
                    vault=request.local_vault,
                )
            else:
                response = self._ask(
                    self.fast_brain, "fast brain", request.view.text,
                    request.context_view.text, vault=request.local_vault,
                )
        # Rehydration is a no-op for a trusted brain (empty vault): its answer
        # is already in raw space because its input was.
        return RouteDecision(
            tier_answered="local",
            difficulty_score=difficulty,
            est_latency_ms=response.latency_ms,
            est_cost_usd=response.cost_usd,
            pii_entities_detected=request.pii_detected,
            pii_entities_masked=len(request.local_vault),
            answer=request.guard.rehydrate(response.text, request.local_vault),
            notes=request.notes,
        )

    def _prepare_gap_escalation(
        self, request: _Request, local: BrainResponse, gap: str
    ) -> tuple[MaskResult, str, dict[str, str]]:
        """Mask everything a split sends, and build the deep brain's context.

        Returns `(masked_gap, compressed_context, vault)`.

        Extracted so the blocking and streaming split paths cannot drift: this
        is where every string that crosses the boundary gets masked and the
        invariant asserted, and having that logic in one place is worth more
        than the indirection costs. Two copies of privacy-critical code is how
        one of them ends up a fix behind.
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
        parts.append(
            f"For background, the user originally asked: {masked_query.masked_text}"
        )
        if self.policy.send_partial_to_cloud:
            masked_partial = request.mask_for_boundary(local.text)
            vault.update(masked_partial.vault)
            parts.append(
                "A smaller on-device model has already answered part of it: "
                f"{masked_partial.masked_text}\nDo not repeat that part."
            )

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
        return masked_gap, compressed, vault

    def _answer_hybrid_streaming(
        self, request: _Request, difficulty: float, local: BrainResponse, gap: str
    ):
        """`_answer_hybrid`, but the cloud half is yielded as it generates.

        A generator that yields `("delta", {...})` and **returns** the finished
        `RouteDecision` (PEP 380), so `route_stream` gets both out of one
        `yield from` without an out-parameter.

        Identical masking to the blocking path -- literally the same
        `_prepare_gap_escalation` call -- so the guarantee cannot differ between
        the two. The only difference is that the deep brain is asked to stream.
        """
        masked_gap, compressed, vault = self._prepare_gap_escalation(request, local, gap)
        local_answer = request.guard.rehydrate(local.text, request.local_vault)

        # Announced *before* the call, not after: the point of showing what
        # crossed is to show it while the user is waiting on the answer it
        # bought, not as a footnote once the answer has arrived.
        yield ("crossing", _crossing_payload(masked_gap.masked_text, compressed, vault))
        self.tracer.call(
            "deep brain", self.deep_brain, masked_gap.masked_text, compressed,
            crossing=True, vault=vault,
        )
        start = time.perf_counter()
        pieces: list[str] = []
        for piece in self.deep_brain.answer_stream(masked_gap.masked_text, compressed):
            pieces.append(piece)
            # Rehydrated per-delta: the cloud answers in masked space, and a
            # `[PII_EMAIL_1]` must never reach the screen. Safe piecewise
            # because a placeholder is one token-ish run of text; the final
            # `cloud_answer` below is rehydrated whole regardless, so a
            # placeholder split across two deltas is corrected there.
            yield ("delta", {"text": request.guard.rehydrate(piece, vault), "tier": "cloud"})
        cloud_latency_ms = (time.perf_counter() - start) * 1000

        response = BrainResponse(text="".join(pieces).strip(), latency_ms=cloud_latency_ms)
        self.tracer.result(self.deep_brain, response)
        self.tracer.rehydrated(vault)
        cloud_answer = request.guard.rehydrate(response.text, vault)
        return RouteDecision(
            tier_answered="hybrid",
            difficulty_score=difficulty,
            est_latency_ms=local.latency_ms + cloud_latency_ms,
            est_cost_usd=local.cost_usd + response.cost_usd,
            pii_entities_detected=request.pii_detected,
            pii_entities_masked=len(vault),
            answer=f"{local_answer}\n\n{cloud_answer}",
            notes=request.notes,
            local_answer=local_answer,
            cloud_answer=cloud_answer,
            gap=request.guard.rehydrate(masked_gap.masked_text, vault),
            crossed_to_cloud=_crossing_payload(masked_gap.masked_text, compressed, vault),
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

        **The gap is the question asked, not a note attached to one.** The deep
        brain's `query` argument is the masked *gap*; the original query, the
        history and the partial answer are background in `context`. Getting this
        backwards is not a stylistic difference -- it was a real, observed
        failure. Sending the whole query as the ask and mentioning the gap in
        the context got this, on a real Cirrascale call:

            query : "My email is [PII_EMAIL_1] ... Draft a short reply about the
                     backup breach, and give me the exact ISBN ..."
            reply : "I cannot provide you with a reply that includes your
                     personal information."

        The deep brain answered the *whole* placeholder-laden request and
        refused it on safety grounds, when the only thing actually needed from
        it was "the exact ISBN of the 1813 first edition" -- entirely
        innocuous. Asking the narrow question narrowly is also the behaviour
        the split was specified to have: only what needs addressing goes in the
        second call.

        Ordering inside the context is still load-bearing, just inverted: the
        gap no longer needs to survive truncation (it is the query now and
        cannot be truncated at all), so `compress_context` keeping the *tail*
        means the partial answer goes last -- the most useful background for
        not repeating work -- and the conversation history is trimmed first.
        """
        masked_gap, compressed, vault = self._prepare_gap_escalation(request, local, gap)

        # The local half is finished and rehydrated *before* the cloud call, not
        # after, so it can be handed out now rather than in ~15s. Rehydrating
        # here is safe and not merely convenient: this text came from a brain
        # given `request.view`, so `request.view.vault` is the only vault that
        # can apply to it, and that vault is complete already -- it does not
        # depend on anything the deep brain will return.
        local_answer = request.guard.rehydrate(local.text, request.local_vault)
        self._emit_progress(
            request,
            RouteProgress(
                phase="local_answer",
                local_answer=local_answer,
                gap=gap,
                difficulty_score=difficulty,
                local_latency_ms=local.latency_ms,
                notes=list(request.notes),  # copy: `notes` keeps growing below
            ),
        )

        # The gap is the question. See the docstring for the real refusal this
        # ordering fixes.
        response = self._ask(
            self.deep_brain, "deep brain", masked_gap.masked_text, compressed,
            crossing=True, vault=vault,
        )
        self.tracer.rehydrated(vault)

        # Rehydrate the cloud half only now, on-device, after the answer is
        # back. The local half needed no rehydration at all when the brain was
        # trusted -- it was never masked -- which the empty vault makes a no-op
        # rather than a special case.
        cloud_answer = request.guard.rehydrate(response.text, vault)
        return RouteDecision(
            tier_answered="hybrid",
            difficulty_score=difficulty,
            # Both calls really happened and the user waited for both, so both
            # are billed -- same accounting as a discarded local answer.
            est_latency_ms=local.latency_ms + response.latency_ms,
            est_cost_usd=local.cost_usd + response.cost_usd,
            pii_entities_detected=request.pii_detected,
            pii_entities_masked=len(vault),
            answer=f"{local_answer}\n\n{cloud_answer}",
            notes=request.notes,
            local_answer=local_answer,
            cloud_answer=cloud_answer,
            gap=request.guard.rehydrate(masked_gap.masked_text, vault),
            crossed_to_cloud=_crossing_payload(masked_gap.masked_text, compressed, vault),
        )

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

        # No partial to hand out on this path -- that is the whole difference
        # from `_answer_hybrid`. Emitted anyway so a UI can say *why* it is
        # about to wait: "escalating, nothing usable locally" is a much better
        # thing to show for 15s than an undifferentiated spinner. `local_answer`
        # and `gap` are None rather than empty strings, so "no partial exists"
        # is distinguishable from "the partial was blank".
        self._emit_progress(
            request,
            RouteProgress(
                phase="escalating",
                local_answer=None,
                gap=None,
                difficulty_score=difficulty,
                local_latency_ms=discarded.latency_ms if discarded is not None else 0.0,
                notes=list(request.notes),
            ),
        )

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
            pii_entities_masked=len(vault),
            answer=request.guard.rehydrate(response.text, vault),
            notes=request.notes,
            crossed_to_cloud=_crossing_payload(masked_query.masked_text, compressed, vault),
        )
