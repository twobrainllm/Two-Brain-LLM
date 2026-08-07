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

    def answer(self, query: str, context: str = "") -> BrainResponse:
        self.received.append(query)
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


def test_the_deep_brain_is_asked_only_about_the_gap():
    brain = _ScriptedStructuredBrain(
        "Half of it.", confidence=0.9, unknown="the other half"
    )
    router, deep = _router(brain)

    router.route("Do both halves.")

    _query, context = deep.calls[0]
    assert "Answer only the remaining part it could not: the other half" in context


def test_low_confidence_with_a_usable_partial_still_splits():
    """No gap named, but the number alone is below threshold.

    The partial answer is kept rather than discarded, which is the difference
    from `_route_on_confidence`.
    """
    brain = _ScriptedStructuredBrain("A rough attempt.", confidence=0.2)
    router, _deep = _router(brain)

    decision = router.route("Something hard.")

    assert decision.difficulty_score == pytest.approx(0.8)
    assert decision.tier_answered == "hybrid"
    assert decision.local_answer == "A rough attempt."


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
    brain = _ScriptedStructuredBrain("Something.", confidence=None)
    router, _deep = _router(brain)

    decision = router.route("A question.")

    assert decision.difficulty_score == 1.0
    assert decision.tier_answered == "hybrid"
    assert any("no usable confidence signal" in note for note in decision.notes)


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
    assert "[PII_EMAIL_1]" in sent
    # ...and the user still gets the real values back, rehydrated on-device.
    assert "jane.doe@example.com" in decision.answer


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

    _query, context = deep.calls[0]
    assert "The on-device half." not in context
    assert "the missing half" in context
    # ...and the user still gets both halves back.
    assert decision.local_answer == "The on-device half."
    assert any("partial answer withheld" in note for note in decision.notes)


def test_the_gap_survives_context_compression():
    """`compress_context` keeps the tail, so the gap must be last.

    With a long partial answer and a small budget, the partial is what gets
    trimmed. Losing the gap instead would send the deep brain a context that
    never says what it is being asked for.
    """
    brain = _ScriptedStructuredBrain(
        "word " * 500, confidence=0.9, unknown="THE ACTUAL GAP"
    )
    router, deep = _router(brain, policy=RoutePolicy(max_context_chars=200))

    decision = router.route("A question.")

    _query, context = deep.calls[0]
    assert "THE ACTUAL GAP" in context
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
