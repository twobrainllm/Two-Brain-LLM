"""Shape C: the local brain solves what it can and names what it can't.

Two halves, matching the two seams the feature spans:

1. **The parser** (`signals/structured.py`) -- pure, no router, no brain. Most
   of these are transcriptions of ways a small quantized model actually
   misbehaves, not hypotheticals.
2. **The route** (`routing/router.py::_route_on_structured_answer`) -- driven by
   a fake brain that returns a scripted `BrainResponse`, so the *decision* is
   what's under test rather than any model's behaviour. The real-hardware
   counterpart is `tests/test_npu_brain.py`.

The privacy tests here are the ones that matter most. Shape C sends the deep
brain **newly generated local text** -- the gap, and optionally the partial
answer -- which is text the guard has never seen before. That is a genuinely
new way for PII to escape, so it is asserted against the string the deep brain
actually received, not against a mocked call.
"""
from __future__ import annotations

import pytest

from two_brain_router.routing import BrainResponse, RoutePolicy, TwoBrainRouter
from two_brain_router.signals.structured import (
    StructuredAnswer,
    parse_structured,
)

# --------------------------------------------------------------------------
# 1. The parser
# --------------------------------------------------------------------------


def test_clean_single_line_json_is_the_normal_path():
    parsed = parse_structured(
        '{"solution": "Tokyo is UTC+9.", "confidence": 95, "unknown": ""}'
    )
    assert parsed == StructuredAnswer(
        solution="Tokyo is UTC+9.", confidence=0.95, unknown="", source="json"
    )
    assert parsed.has_gap is False


def test_a_code_fence_is_stripped_even_though_the_prompt_forbids_it():
    parsed = parse_structured(
        'Sure!\n```json\n{"solution": "a", "confidence": 50, "unknown": "b"}\n```\nHope that helps!'
    )
    assert parsed.solution == "a"
    assert parsed.unknown == "b"
    assert parsed.source == "json"


def test_braces_inside_the_solution_do_not_truncate_the_object():
    """A solution containing code is the case a naive regex gets wrong."""
    parsed = parse_structured(
        '{"solution": "Use `if (x) { return {a: 1}; }` here", "confidence": 80, "unknown": ""}'
    )
    assert parsed.solution == "Use `if (x) { return {a: 1}; }` here"
    assert parsed.confidence == 0.80


def test_an_unknown_returned_as_a_list_is_joined_not_stringified():
    parsed = parse_structured(
        '{"solution": "a", "confidence": 40, "unknown": ["the proof", "the bound"]}'
    )
    assert parsed.unknown == "the proof; the bound"
    assert "[" not in parsed.unknown  # never the repr of a Python list


@pytest.mark.parametrize("written", ["", "None", "n/a", "null", "nothing", "NONE."])
def test_a_nullish_unknown_is_not_a_gap(written):
    """Models write the *word* "None" in a field asked for an empty string.

    Taking that literally would make every single query report a gap, and every
    query would escalate -- the exact failure this path exists to avoid.
    """
    parsed = parse_structured(
        '{"solution": "a", "confidence": 90, "unknown": %s}' % ('""' if not written else f'"{written}"')
    )
    assert parsed.unknown == ""
    assert parsed.has_gap is False


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("95", 0.95),  # int -> the 0-100 scale the prompt asked for
        ("0.95", 0.95),  # float <= 1 -> already a fraction
        ("95.0", 0.95),  # float > 1 -> the 0-100 scale, written with a point
        ('"95%"', 0.95),  # a string with a unit
        ("1", 0.01),  # int 1 means one percent, NOT fully confident
        ("150", 1.0),  # clamped rather than rejected
        ("-5", 0.0),
        ("true", None),  # a bool is not a score
        ('"unsure"', None),  # no number at all
    ],
)
def test_confidence_scales_are_read_in_the_safe_direction(written, expected):
    parsed = parse_structured('{"solution": "a", "confidence": %s, "unknown": ""}' % written)
    assert parsed.confidence == expected


def test_alias_field_names_are_accepted():
    parsed = parse_structured(
        '{"answer": "a", "certainty": 70, "cannot_answer": "the rest"}'
    )
    assert (parsed.solution, parsed.confidence, parsed.unknown) == ("a", 0.70, "the rest")


def test_no_json_falls_back_to_the_bare_confidence_line():
    """A model that regresses to the older Shape B format still routes.

    This is the rung that keeps `TWO_BRAIN_STRUCTURED=0` and a badly-behaved
    model from being the same event.
    """
    parsed = parse_structured("Tokyo is UTC+9.\nCONFIDENCE: 90")
    assert parsed.source == "self_report"
    assert parsed.solution == "Tokyo is UTC+9."
    assert parsed.confidence == 0.90
    assert parsed.unknown == ""


def test_neither_format_gives_no_signal_rather_than_a_made_up_one():
    parsed = parse_structured("I think it is probably fine.")
    assert parsed.source == "raw"
    assert parsed.confidence is None  # NOT 0.0 -- see signals/confidence.py
    assert parsed.solution == "I think it is probably fine."


def test_a_masked_output_field_is_recorded_but_kept_separate():
    """Parsed so it isn't silently dropped; never merged into the solution.

    Whatever the model calls "masked" has no authority here -- `privacy/guard.py`
    already masked the query before this brain was ever called.
    """
    parsed = parse_structured(
        '{"solution": "a", "confidence": 90, "unknown": "", "masked_output": "REDACTED"}'
    )
    assert parsed.model_masked_output == "REDACTED"
    assert parsed.solution == "a"


# --------------------------------------------------------------------------
# 2. The route
# --------------------------------------------------------------------------


class _ScriptedStructuredBrain:
    """A fast brain that returns exactly what a test tells it to.

    Deliberately not a model: Shape C's routing decision is a function of
    `(confidence, unknown, text)`, so scripting those three directly is what
    isolates the decision from any model's willingness to produce them.
    """

    reports_confidence = True
    reports_gaps = True
    #: Matches the two real brains this stands in for -- both AI-PC brains run
    #: on this machine and are handed the query as typed. Not a detail: with
    #: this False, every privacy test below would receive pre-masked input and
    #: would pass without ever exercising the masking that protects this path.
    trusted_with_raw_pii = True

    def __init__(self, text: str, confidence: float | None, unknown: str = "") -> None:
        self._response = BrainResponse(
            text=text, latency_ms=800.0, confidence=confidence, unknown=unknown
        )
        self.received: list[str] = []
        self.received_context: list[str] = []

    def answer(self, query: str, context: str = "") -> BrainResponse:
        self.received.append(query)
        self.received_context.append(context)
        return self._response


class _RecordingDeepBrain:
    """Records the exact (query, context) pair that crossed the boundary.

    Echoes the masked query back into its answer, the same way `CloudDeepBrain`
    does. That is not decoration: a reply containing no placeholder would make
    every rehydration assertion below vacuously true.
    """

    reports_confidence = False
    reports_gaps = False
    trusted_with_raw_pii = False  # the boundary itself

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def answer(self, query: str, context: str = "") -> BrainResponse:
        self.calls.append((query, context))
        return BrainResponse(
            text=f"[cloud filled the gap re: {query}]",
            latency_ms=1500.0,
            cost_usd=0.004,
        )


def _router(brain, policy: RoutePolicy | None = None) -> tuple[TwoBrainRouter, _RecordingDeepBrain]:
    router = TwoBrainRouter(tier="pc", policy=policy)
    router.fast_brain = brain
    deep = _RecordingDeepBrain()
    router.deep_brain = deep
    return router, deep


def test_confident_and_no_gap_stays_local():
    brain = _ScriptedStructuredBrain("Tokyo is UTC+9.", confidence=0.95)
    router, deep = _router(brain)

    decision = router.route("What time zone is Tokyo in?")

    assert decision.tier_answered == "local"
    assert decision.answer == "Tokyo is UTC+9."
    assert deep.calls == []
    assert decision.local_answer is None  # only populated on a split


def test_a_named_gap_splits_the_query_even_at_high_confidence():
    """The headline behaviour, and the one that is *not* just Shape B.

    Confidence 0.90 inverts to difficulty 0.10, far below the 0.55 threshold --
    under Shape B this query would have stayed local and the missing half would
    simply never have been answered. The model's own statement that something
    is missing is a more specific signal than its overall score, so it wins.
    """
    brain = _ScriptedStructuredBrain(
        "Merge sort is O(n log n) in all cases.",
        confidence=0.90,
        unknown="quicksort's worst case and the trade-offs",
    )
    router, deep = _router(brain)

    decision = router.route("Compare merge sort and quicksort complexity.")

    assert decision.difficulty_score == pytest.approx(0.10)
    assert decision.difficulty_score < router.policy.escalate_threshold
    assert decision.tier_answered == "hybrid"
    assert decision.local_answer == "Merge sort is O(n log n) in all cases."
    assert decision.cloud_answer.startswith("[cloud filled the gap")
    assert decision.gap == "quicksort's worst case and the trade-offs"
    # Both halves reach the user, local first, separated by a blank line.
    assert decision.answer == f"{decision.local_answer}\n\n{decision.cloud_answer}"
    assert len(deep.calls) == 1


def test_the_gap_is_the_question_the_deep_brain_is_asked():
    """The gap goes in the `query` slot, not buried in the context.

    This is a regression test for a real refusal. With the whole original query
    as the ask and the gap merely mentioned in context, a live Cirrascale call
    answered the *whole* placeholder-laden request and refused it:

        "I cannot provide you with a reply that includes your personal
         information."

    ...when the only thing needed was an ISBN. Asking narrowly is both what
    makes the split work and what the split was specified to do.
    """
    brain = _ScriptedStructuredBrain(
        "Half of it.", confidence=0.9, unknown="the other half"
    )
    router, deep = _router(brain)

    router.route("Do both halves.")

    query, context = deep.calls[0]
    assert query == "the other half", f"the gap must be the question asked, got {query!r}"
    # The original query is still available, but as background rather than as
    # the instruction.
    assert "Do both halves." in context
    assert context.index("Do both halves.") >= 0


def test_low_confidence_with_no_named_gap_is_a_plain_escalation():
    """Unconfident overall, but unable to say *which part* is missing.

    There is nothing gap-shaped to ask about, so this is an ordinary
    escalation: the whole query goes and the rough local attempt is discarded,
    rather than displayed beside a full cloud answer covering the same ground.

    This is also a correctness guard, not only a UX call -- see
    `test_an_empty_gap_never_becomes_an_empty_question_to_the_cloud`.
    """
    brain = _ScriptedStructuredBrain("A rough attempt.", confidence=0.2)
    router, deep = _router(brain)

    decision = router.route("Something hard.")

    assert decision.difficulty_score == pytest.approx(0.8)
    assert decision.tier_answered == "cloud"
    assert decision.local_answer is None
    assert any("no specific gap named" in note for note in decision.notes)
    # The cloud is asked the real question, not an empty one.
    assert deep.calls[0][0].strip()


def test_an_empty_gap_never_becomes_an_empty_question_to_the_cloud():
    """Regression: the deep brain must never be asked "".

    Shape C sends the *gap* as the deep brain's query. A low-confidence answer
    with no gap named would otherwise reach `_answer_hybrid` with `gap == ""`
    and ask the cloud an empty question. Observed on real hardware -- the NPU
    returned `confidence 0.00` with `unknown` empty for a query it declined --
    and invisible before the gap became the query, because the old code sent
    the whole query as the ask regardless of the gap.
    """
    brain = _ScriptedStructuredBrain(
        "I am unable to provide that.", confidence=0.0, unknown=""
    )
    router, deep = _router(brain)

    decision = router.route("Draft a reply and give me the exact ISBN of the 1813 edition.")

    assert len(deep.calls) == 1
    asked, _context = deep.calls[0]
    assert asked.strip(), "the deep brain was asked an empty question"
    assert "ISBN" in asked, "the whole query should go when there is no gap to narrow to"
    assert decision.tier_answered == "cloud"


def test_no_usable_partial_is_a_plain_escalation_not_a_split():
    """A split needs two halves. With one, say "cloud" and mean it."""
    brain = _ScriptedStructuredBrain("   ", confidence=0.1, unknown="all of it")
    router, deep = _router(brain)

    decision = router.route("Something hard.")

    assert decision.tier_answered == "cloud"
    assert decision.local_answer is None
    assert len(deep.calls) == 1
    assert any("no usable partial answer" in note for note in decision.notes)


def test_an_unparseable_confidence_is_maximally_uncertain_not_zero_signal():
    """The point of this test is the *difficulty*, not the destination.

    `None` means the model ignored the output format entirely, so it is read as
    maximally uncertain (1.0) rather than scored by some other signal. With no
    gap named either, that lands on a plain escalation.
    """
    brain = _ScriptedStructuredBrain("Something.", confidence=None)
    router, _deep = _router(brain)

    decision = router.route("A question.")

    assert decision.difficulty_score == 1.0
    assert decision.tier_answered == "cloud"
    assert any("no usable confidence signal" in note for note in decision.notes)


def test_a_named_gap_still_splits_even_at_zero_confidence():
    """A gap is a more specific claim than the confidence number, so it wins.

    Guards the fix above from over-reaching: "no gap named" is what routes to a
    plain escalation, not "low confidence". A model that says *what* it is
    missing is still worth splitting on, however unsure it is overall.
    """
    brain = _ScriptedStructuredBrain(
        "Half of it.", confidence=0.0, unknown="the other half"
    )
    router, deep = _router(brain)

    decision = router.route("Do both halves.")

    assert decision.tier_answered == "hybrid"
    assert decision.local_answer == "Half of it."
    assert deep.calls[0][0] == "the other half"


def test_both_calls_are_billed_to_the_user():
    """Speculation is not free, and neither is a split -- the user waited for
    both inferences, so `est_latency_ms` reports both."""
    brain = _ScriptedStructuredBrain("Part.", confidence=0.9, unknown="rest")
    router, _deep = _router(brain)

    decision = router.route("A question.")

    assert decision.est_latency_ms == pytest.approx(800.0 + 1500.0)
    assert decision.est_cost_usd == pytest.approx(0.004)


# --------------------------------------------------------------------------
# 2b. Privacy: the trust boundary, and masking at it
# --------------------------------------------------------------------------

_PII_QUERY = "Email jane.doe@example.com about SSN 123-45-6789 and summarise the policy."


def test_an_on_device_brain_is_given_the_query_exactly_as_typed():
    """The AI-PC model runs here, so masking it protects nothing and costs
    answer quality -- a model asked to draft a reply to `[PII_EMAIL_1]` writes
    a worse reply than one that can see the address.

    This is the assertion that would have failed under the old mask-first
    ordering, and it is the whole point of the change.
    """
    brain = _ScriptedStructuredBrain("Drafted.", confidence=0.95)
    router, deep = _router(brain)

    decision = router.route(_PII_QUERY)

    assert brain.received == [_PII_QUERY]
    assert "jane.doe@example.com" in brain.received[0]
    assert "[PII_" not in brain.received[0]
    # ...and because it answered, nothing was masked and nothing left.
    assert decision.tier_answered == "local"
    assert deep.calls == []
    assert decision.pii_entities_masked == 0
    assert decision.pii_entities_detected == 2


def test_an_off_device_brain_is_still_given_masked_text():
    """`trusted_with_raw_pii` is per-brain, not per-tier.

    `PhoneFastBrain` is the real case: it runs on a physically separate device,
    and `adb reverse` makes that hop *look* like loopback, so the distinction
    has to be declared rather than inferred. A brain that simply forgets to
    declare the flag lands here too, which is the safe direction.
    """
    brain = _ScriptedStructuredBrain("Drafted.", confidence=0.95)
    brain.trusted_with_raw_pii = False
    router, _deep = _router(brain)

    decision = router.route(_PII_QUERY)

    assert "jane.doe@example.com" not in brain.received[0]
    assert "[PII_EMAIL_1]" in brain.received[0]
    assert decision.pii_entities_masked == 2
    assert decision.pii_entities_detected == 2


def test_detected_and_masked_are_counted_separately():
    """Zero masked on a local answer is the *good* outcome, not a missing
    measurement -- so the count of what was present has to survive alongside
    it, or the headline privacy claim reads as "no PII here"."""
    brain = _ScriptedStructuredBrain("Answered.", confidence=0.95)
    router, _deep = _router(brain)

    decision = router.route(_PII_QUERY)

    assert decision.pii_entities_detected == 2
    assert decision.pii_entities_masked == 0


def test_pii_from_the_query_never_reaches_the_cloud_unmasked_on_a_split():
    brain = _ScriptedStructuredBrain(
        "I drafted the email.", confidence=0.9, unknown="the policy summary"
    )
    router, deep = _router(brain)

    decision = router.route(_PII_QUERY)

    assert decision.tier_answered == "hybrid"
    sent = " ".join(deep.calls[0])
    assert "jane.doe@example.com" not in sent
    assert "123-45-6789" not in sent
    # The original query still crosses -- as masked background, not as the ask.
    assert "[PII_EMAIL_1]" in sent
    # ...and the user still sees the real values in the half that never left.
    assert "jane.doe@example.com" in _PII_QUERY  # sanity: the fixture has PII
    assert decision.local_answer == "I drafted the email."


def test_the_raw_partial_answer_is_masked_before_it_crosses():
    """The single most important test in this file.

    The local model was handed the query *unmasked*, so its `solution` can
    quote the user's real email address verbatim. That string is then offered
    to the deep brain as context. If `_answer_hybrid` did not mask it, the
    query-level masking would not save us -- this text did not exist when the
    query was read.
    """
    brain = _ScriptedStructuredBrain(
        "I drafted a note to jane.doe@example.com about SSN 123-45-6789.",
        confidence=0.9,
        unknown="the policy summary",
    )
    router, deep = _router(brain)

    decision = router.route(_PII_QUERY)

    assert decision.tier_answered == "hybrid"
    _query, context = deep.calls[0]
    assert "jane.doe@example.com" not in context
    assert "123-45-6789" not in context
    assert "[PII_EMAIL_1]" in context
    # The user still sees the local half exactly as the model wrote it -- it
    # never left, so it was never masked.
    assert "jane.doe@example.com" in decision.local_answer


def test_pii_the_local_model_invented_in_the_gap_is_masked_before_it_crosses():
    """The other half of the same hazard.

    `unknown` is text the *model* wrote. It never passed through the guard on
    the way in, so a model that restates the question -- or hallucinates an
    example address -- can put an identifier into the escalation payload that
    the original query never contained. If this test fails, PII is leaving the
    device through a field nothing else checks.
    """
    brain = _ScriptedStructuredBrain(
        "Partial.",
        confidence=0.9,
        unknown="I could not verify the address at 555-987-6543 or admin@internal.example",
    )
    router, deep = _router(brain)

    decision = router.route("A question with no PII in it at all.")

    sent = " ".join(deep.calls[0])
    assert "555-987-6543" not in sent
    assert "admin@internal.example" not in sent
    # The gap the caller sees is rehydrated, so nothing is lost -- only what
    # crossed the boundary was masked.
    assert "555-987-6543" in decision.gap


def test_pii_the_local_model_invented_in_the_partial_is_masked_too():
    brain = _ScriptedStructuredBrain(
        "Reply to leaked@example.com.", confidence=0.9, unknown="the rest"
    )
    router, deep = _router(brain)

    decision = router.route("A question with no PII in it at all.")

    _query, context = deep.calls[0]
    assert "leaked@example.com" not in context
    # The local half shown to the user is untouched -- it never left.
    assert decision.local_answer == "Reply to leaked@example.com."


def test_withholding_the_partial_sends_strictly_less_off_device():
    brain = _ScriptedStructuredBrain(
        "The on-device half.", confidence=0.9, unknown="the missing half"
    )
    router, deep = _router(brain, policy=RoutePolicy(send_partial_to_cloud=False))

    decision = router.route("A question.")

    query, context = deep.calls[0]
    assert "The on-device half." not in context, "the partial should have been withheld"
    assert query == "the missing half", "the gap is still asked -- only the partial is withheld"
    # ...and the user still gets both halves back.
    assert decision.local_answer == "The on-device half."
    assert any("partial answer withheld" in note for note in decision.notes)


def test_the_gap_cannot_be_lost_to_context_compression():
    """Compression can never cost us the thing we are asking about.

    It used to be able to: with the gap living in the context, a long partial
    answer and a small budget could trim the very instruction that said what
    the deep brain was for. Now the gap is the `query`, which
    `compress_context` never touches -- so the guarantee is structural rather
    than a matter of keeping it last in a list.
    """
    brain = _ScriptedStructuredBrain(
        "word " * 500, confidence=0.9, unknown="THE ACTUAL GAP"
    )
    router, deep = _router(brain, policy=RoutePolicy(max_context_chars=200))

    decision = router.route("A question.")

    query, context = deep.calls[0]
    assert query == "THE ACTUAL GAP", "the gap must be unaffected by compression"
    assert len(context) <= 200
    assert any("compressed the escalated gap context" in note for note in decision.notes)


# --------------------------------------------------------------------------
# 2c. Interaction with the shapes that already existed
# --------------------------------------------------------------------------


def test_a_brain_that_only_self_rates_still_takes_shape_b():
    """`reports_gaps` is what selects Shape C, not `reports_confidence`.

    `PhoneFastBrain` is exactly this case, which is why the mobile tier is
    untouched by any of the above.
    """
    brain = _ScriptedStructuredBrain("Discarded.", confidence=0.1)
    brain.reports_gaps = False
    router, deep = _router(brain)

    decision = router.route("Something hard.")

    assert decision.tier_answered == "cloud"  # Shape B discards, never splits
    assert decision.local_answer is None


def test_an_image_bearing_query_bypasses_shape_c_rather_than_dropping_the_image():
    """Shape C's brain call has nowhere to put an image.

    Routing one through it would silently drop the image and answer the text
    alone -- a wrong answer with no error. Until Shape C handles images, an
    image forces the heuristic path, which does.
    """
    brain = _ScriptedStructuredBrain("x", confidence=0.9, unknown="y")
    router, _deep = _router(brain)

    with pytest.raises(ValueError, match="cannot see"):
        router.route("What is in this picture?", image=__import__("pathlib").Path("nope.png"))


# --------------------------------------------------------------------------
# 2d. Progressive results: the local half, before the cloud half exists
# --------------------------------------------------------------------------


def test_the_local_half_is_handed_out_before_the_cloud_is_called():
    """`on_progress` fires while the deep brain has not answered yet.

    Asserted by recording the deep brain's call log *at the moment the callback
    runs*: if the progress event were emitted after the cloud call, the log
    would already be non-empty. Ordering is the entire feature, so ordering is
    what's checked -- not merely that the callback fired at all.
    """
    brain = _ScriptedStructuredBrain("The local half.", confidence=0.9, unknown="the rest")
    router, deep = _router(brain)

    seen = []
    router.route("A question.", on_progress=lambda p: seen.append((p, len(deep.calls))))

    assert len(seen) == 1
    progress, cloud_calls_so_far = seen[0]
    assert cloud_calls_so_far == 0, "the cloud was already called before the partial was emitted"
    assert progress.phase == "local_answer"
    assert progress.local_answer == "The local half."
    assert progress.gap == "the rest"


def test_progress_carries_the_rehydrated_local_answer_not_placeholders():
    """What the callback hands out is display-ready.

    A UI paints this string directly, so a `[PII_EMAIL_1]` reaching it would be
    user-visible. Only relevant when the fast brain was untrusted (the phone) --
    a trusted brain's answer was never masked -- so that is the case tested.
    """
    brain = _ScriptedStructuredBrain(
        "Replying to [PII_EMAIL_1] now.", confidence=0.9, unknown="the rest"
    )
    brain.trusted_with_raw_pii = False
    router, _deep = _router(brain)

    seen = []
    router.route("Email jane.doe@example.com about this.", on_progress=seen.append)

    assert seen[0].local_answer == "Replying to jane.doe@example.com now."
    assert "[PII_" not in seen[0].local_answer


def test_no_progress_event_when_the_query_never_leaves():
    """A locally-answered query has no partial -- there is nothing to wait for,
    so there is nothing to announce."""
    brain = _ScriptedStructuredBrain("Answered fully.", confidence=0.95)
    router, _deep = _router(brain)

    seen = []
    decision = router.route("An easy question.", on_progress=seen.append)

    assert decision.tier_answered == "local"
    assert seen == []


def test_a_broken_progress_callback_cannot_break_the_request():
    """A listener that raises is a dead listener, not a failed route.

    The realistic cause is a streaming client disconnecting mid-request. That
    must not turn a working answer into a 500, corrupt the audit trail, or
    abandon a cloud call already paid for.
    """
    brain = _ScriptedStructuredBrain("The local half.", confidence=0.9, unknown="the rest")
    router, deep = _router(brain)

    def _explode(_progress):
        raise ConnectionResetError("client went away")

    decision = router.route("A question.", on_progress=_explode)

    assert decision.tier_answered == "hybrid"
    assert decision.local_answer == "The local half."
    assert len(deep.calls) == 1


# --------------------------------------------------------------------------
# 2e. Conversation history (the `context` argument)
# --------------------------------------------------------------------------

_HISTORY = "Earlier in this conversation:\nUser: what is a CT scan?\nAssistant: It uses X-rays."


def test_an_on_device_brain_is_given_the_history_raw():
    """Multi-turn only works if the local model actually receives the history.

    It never did before: every `_ask(self.fast_brain, ...)` call omitted the
    context argument, so `route(query, context)` fed the cloud and nothing
    else, and a follow-up like "explain in more detail" reached the model as a
    standalone sentence with no referent.
    """
    brain = _ScriptedStructuredBrain("Expanded.", confidence=0.95)
    router, _deep = _router(brain)

    router.route("Explain in more detail", context=_HISTORY)

    assert brain.received_context == [_HISTORY]


def test_history_for_an_off_device_brain_is_masked_like_everything_else():
    """`trusted_with_raw_pii` governs the context too, not just the query.

    Otherwise the phone tier would receive masked queries and raw history --
    which is the same leak, one field over.
    """
    brain = _ScriptedStructuredBrain("Expanded.", confidence=0.95)
    brain.trusted_with_raw_pii = False
    router, _deep = _router(brain)

    router.route(
        "Explain in more detail",
        context="Earlier: my email is jane.doe@example.com",
    )

    sent = brain.received_context[0]
    assert "jane.doe@example.com" not in sent
    assert "[PII_EMAIL_1]" in sent


def test_history_reaches_the_cloud_masked_on_a_split():
    """The chosen policy: history *does* cross, but never in the clear."""
    brain = _ScriptedStructuredBrain("Half.", confidence=0.9, unknown="the rest")
    router, deep = _router(brain)

    decision = router.route(
        "Explain in more detail",
        context="Earlier: contact me at jane.doe@example.com",
    )

    assert decision.tier_answered == "hybrid"
    _query, context = deep.calls[0]
    assert "jane.doe@example.com" not in context
    assert "[PII_EMAIL_1]" in context


def test_a_placeholder_that_came_from_the_history_still_rehydrates():
    """The subtle one: an answer can echo a placeholder minted from the
    *context*, not the query, so rehydration has to consider both vaults.

    With only the query's vault, `[PII_EMAIL_1]` would reach the user as a
    literal token -- not a leak, but a visibly broken answer, and exactly the
    class of bug `_Request.local_vault` exists to prevent.
    """
    brain = _ScriptedStructuredBrain("I will write to [PII_EMAIL_1].", confidence=0.95)
    brain.trusted_with_raw_pii = False  # so masking happens at all
    router, _deep = _router(brain)

    decision = router.route(
        "Reply to them",  # no PII in the query itself
        context="Earlier: my email is jane.doe@example.com",
    )

    assert decision.answer == "I will write to jane.doe@example.com."
    assert "[PII_" not in decision.answer


def test_no_context_is_still_the_default_and_changes_nothing():
    brain = _ScriptedStructuredBrain("Answered.", confidence=0.95)
    router, _deep = _router(brain)

    decision = router.route("A standalone question.")

    assert brain.received_context == [""]
    assert decision.tier_answered == "local"
    assert not any("conversation context" in n for n in decision.notes)


def test_pii_that_only_appears_in_the_history_is_still_counted():
    """The audit trail must count what crossed, not just what was typed.

    Found by testing, not by reading: with PII only in the history, the
    response reported `0 detected / 0 masked` while the trace showed an email
    being masked at the boundary. The masking was correct -- the *reporting*
    understated it, which on a privacy demo is the failure that matters, since
    the profiler would have said "No PII detected" about a request that sent a
    masked address to the cloud.
    """
    brain = _ScriptedStructuredBrain("Half.", confidence=0.9, unknown="the rest")
    router, deep = _router(brain)

    decision = router.route(
        "Explain in more detail",  # no PII in the query at all
        context="Earlier: email me at jane.doe@example.com",
    )

    assert decision.tier_answered == "hybrid"
    assert decision.pii_entities_detected == 1, "history PII was not detected"
    assert decision.pii_entities_masked >= 1, "history PII was masked but not reported as masked"
    assert "jane.doe@example.com" not in " ".join(deep.calls[0])


def test_the_same_entity_in_query_and_history_counts_once():
    """Detection counts occurrences; the vault keys on values. Reporting the
    raw occurrence count against a vault-derived masked count would read as
    "2 detected, 1 masked" -- which looks like a leak and isn't one."""
    brain = _ScriptedStructuredBrain("Half.", confidence=0.9, unknown="the rest")
    router, _deep = _router(brain)

    decision = router.route(
        "Resend to jane.doe@example.com",
        context="Earlier: email me at jane.doe@example.com",
    )

    assert decision.pii_entities_detected == 1
    assert decision.pii_entities_masked == 1
