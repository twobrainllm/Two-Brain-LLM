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

    #: `"hybrid"` means **both** brains contributed: the local model answered
    #: what it could and named a gap, and the deep brain answered only that gap
    #: (Shape C -- docs/ORCHESTRATOR.md). It is a third value rather than
    #: `"cloud"` because the two are different privacy events *and* different
    #: cost events, and collapsing them would make the audit trail lie in both
    #: directions. For the "did anything cross the boundary?" question, treat
    #: `"hybrid"` exactly like `"cloud"` -- it did.
    tier_answered: Literal["local", "cloud", "hybrid"]
    difficulty_score: float
    est_latency_ms: float
    est_cost_usd: float
    #: How many PII entities were **masked**, because something crossed a
    #: boundary. Zero on a locally-answered query, and that zero is the good
    #: outcome, not a missing measurement -- nothing was masked because nothing
    #: left the device. Compare `pii_entities_detected`, which is what the query
    #: contained regardless.
    #:
    #: These were one field until masking moved from the front door to the
    #: boundary. Keeping them merged would have made the headline privacy claim
    #: unreadable: a query full of PII answered entirely on-device would report
    #: "0", which reads as "no PII here" when it actually means "all of it
    #: stayed".
    pii_entities_masked: int
    #: What to show the user: the merged text on a hybrid decision, the single
    #: brain's answer otherwise. Always rehydrated, always on-device.
    answer: str
    notes: list[str]
    #: The three fields below are populated on a hybrid decision only, so a
    #: caller can show *which* brain said what instead of one opaque blob. All
    #: rehydrated, same as `answer`.
    local_answer: str | None = None
    cloud_answer: str | None = None
    #: What the local model said it could not do, in its own words.
    gap: str | None = None
    #: How many PII entities the query contained, whether or not any were
    #: masked. Always populated. Defaulted so the field could be added without
    #: breaking any existing construction of this dataclass.
    pii_entities_detected: int = 0
    #: Exactly what crossed the boundary, for display. None when nothing did.
    #:
    #:     {"query": "<masked>", "context": "<masked, compressed>",
    #:      "substitutions": [{"type": "EMAIL", "value": "...", "placeholder": "[PII_EMAIL_1]"}]}
    #:
    #: `notes` already carries a one-line version of this, but a UI should not
    #: have to scrape prose to show the user what left their machine. The raw
    #: `value` is included deliberately: it is the user's own text, already on
    #: their screen, and showing `jane.doe@example.com -> [PII_EMAIL_1]` side by
    #: side is the entire point -- a masked string alone proves nothing without
    #: what it replaced.
    crossed_to_cloud: dict | None = None


@dataclass
class RouteProgress:
    """A partial result, handed out *before* `route()` returns.

    Exists because the two halves of a split have very different latencies --
    the local model answers in ~4s and the cloud leg has been measured at 15s+
    -- so waiting for the merge before showing anything wastes an answer that
    was ready the whole time. `route(..., on_progress=...)` emits one of these
    at the moment the local half is settled and the cloud call is about to
    start.

    Deliberately *not* a `RouteDecision`. A decision is final and complete; this
    is explicitly neither, and giving it its own type means a caller cannot
    accidentally treat an in-flight partial as the finished record -- the
    `est_cost_usd`/`est_latency_ms` fields a decision carries are not knowable
    yet, so they are simply absent rather than present-and-wrong.
    """

    #: `"local_answer"` -- the local model produced a usable partial and named a
    #: gap; the deep brain is about to be asked about that gap only.
    #: `"escalating"` -- nothing usable came back locally (or the fast brain was
    #: skipped), so the whole query is going to the deep brain and there is no
    #: partial to show.
    phase: Literal["local_answer", "escalating"]
    #: Rehydrated, ready to display. None on `"escalating"`.
    local_answer: str | None
    #: What the deep brain is being asked. None on `"escalating"`, where it is
    #: the whole query rather than a named gap.
    gap: str | None
    difficulty_score: float
    #: What the local half cost, in wall-clock. The cloud half is still running.
    local_latency_ms: float
    #: The audit trail *so far*. The final `RouteDecision.notes` is a superset.
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
    #: The same ceiling for Shape C, and deliberately a much larger number.
    #:
    #: `local_latency_budget_ms` exists to avoid *starting* a local inference
    #: whose answer would then be thrown away -- "a local answer would have been
    #: discarded anyway". **That premise is false in Shape C.** There, a usable
    #: local answer is always kept and shown to the user; the deep brain is
    #: asked a narrower question alongside it, not instead of it. So the local
    #: call is never waste, and pricing it as if it were skipped the fast brain
    #: on two of the four demo queries -- including the PII one, where skipping
    #: it sent to the cloud a query the local model went on to answer fully
    #: on-device.
    #:
    #: This is therefore a runaway guard, not a target. The value comes from
    #: measurement, not preference: real structured answers on the AI-PC tier
    #: took 2558-6501 ms (see data/npu_model/phi-3.5-mini-instruct/
    #: _real_structured_inference_log.md), so this sits at roughly 2.3x the
    #: slowest observed call -- high enough never to fire on a normal query,
    #: low enough to bail on a pathological one.
    local_partial_budget_ms: float = 15000
    #: Escalated context is trimmed to this many characters before it leaves
    #: the device.
    max_context_chars: int = 800
    #: On a hybrid decision, whether the local model's partial answer is sent
    #: to the deep brain alongside the gap.
    #:
    #: A privacy/quality trade, which is why it is a visible knob and not an
    #: implementation detail. **On:** the deep brain can complete the answer
    #: instead of duplicating it, at the cost of one more piece of
    #: locally-generated text crossing the boundary (masked first, like
    #: everything else). **Off:** strictly less leaves the device, and the two
    #: halves may overlap or contradict, because the deep brain is answering
    #: the gap blind.
    send_partial_to_cloud: bool = True

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

    def needs_gap_fill(self, difficulty: float, gap: str) -> bool:
        """Does this structured local answer need the deep brain at all?

        Two independent triggers, ORed:

        - **A named gap.** The local model explicitly said which part it could
          not do. That is a direct statement about *this* query and it counts
          on its own, even at high confidence -- a model can be entirely sure
          about the half it answered and still be missing the other half.
          Ignoring a stated gap because the overall number looked good would
          throw away the most specific signal in the system.
        - **Low confidence**, via the same `escalate_threshold` every other path
          uses. Still one threshold, not two.

        Deliberately no latency term, unlike `should_escalate`. By the time this
        is asked the local inference has already been paid for, and the budget
        was settled *before* the call (see `router.py::_route_on_confidence`).
        Re-testing it here could only add the deep brain's latency on top of
        time already spent.
        """
        return bool(gap.strip()) or difficulty >= self.escalate_threshold

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
