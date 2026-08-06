"""Self-reported confidence: parsing it, and converting it to a difficulty.

Per `src/phone_brain/L_INTERFACE_CONTRACT.md` the decided signal is
**self-reported confidence** -- the fast brain answers and rates its own
confidence in a single call, and the *orchestrator*, never the brain, applies
the threshold and owns the routing decision. That contract is explicit that L
"never decides to escalate itself -- it only ever produces a number". So this
module deliberately does not export a `should_escalate`: it converts the
brain's number into the difficulty scale `routing/policy.py` already thresholds
on, and stops there.

That conversion is the whole answer to "the threshold lives in two places"
(`docs/PHONE_BRAIN.md` reconciliation point 3): rather than carry a second
`confidence_threshold` alongside `RoutePolicy.escalate_threshold`, confidence
is inverted into a difficulty and the *existing* single threshold decides.

Kept pure -- prompt text in, parsed values out, no I/O and no reference to a
brain -- for the same reason `routing/policy.py` is pure: it can be
unit-tested on its own. `routing/brains.py` calls into this module; nothing
here calls back.
"""
from __future__ import annotations

import re

#: Appended to the prompt to ask the brain to self-rate. Kept byte-identical to
#: `src/phone_brain/confidence_estimator.py`'s `SELF_REPORT_SUFFIX`, which is the
#: wording already exercised against the mock server -- changing it changes what
#: the model is asked, so it is a re-validation, not a refactor.
SELF_REPORT_SUFFIX = (
    "\n\nAfter answering, on a new line output exactly: "
    "CONFIDENCE: <a number from 0 to 100>"
)

#: Same expression as confidence_estimator.py's `_CONF_RE`, for the same reason.
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\d{1,3})", re.IGNORECASE)


def parse_self_reported(text: str) -> tuple[str, float | None]:
    """Split a self-rated completion into `(answer, confidence)`.

    `confidence` is normalised to `[0.0, 1.0]`, or **None** when the model did
    not emit a parseable `CONFIDENCE:` line at all.

    None is not the same as 0.0, and the distinction is load-bearing:

    - **None** means *no signal* -- the model ignored the instruction format.
      `src/phone_brain/PHONE_DEPLOYMENT_GUIDE.md` Part 8 flags this as a real,
      expected failure mode of a small quantized model, so the router falls
      back to the surface-feature heuristic rather than inventing a number.
    - **0.0** means the brain (or its transport) is telling us it cannot answer
      this -- a definite escalate.

    Collapsing the two would either silently force-escalate every query a model
    formats badly, or silently trust a value that was never reported.
    """
    match = CONFIDENCE_RE.search(text)
    if match is None:
        return text.strip(), None
    # Clamp rather than reject: a model that says "CONFIDENCE: 150" has still
    # signalled "very confident", and the regex already bounds this to 3 digits.
    reported = min(max(int(match.group(1)), 0), 100)
    answer = CONFIDENCE_RE.sub("", text).strip()
    return answer, reported / 100.0


def confidence_to_difficulty(confidence: float) -> float:
    """Invert a confidence into the difficulty scale `RoutePolicy` thresholds.

    `signals/difficulty.py` scores 0.0 (easy) -> 1.0 (hard); a self-report is
    the opposite polarity. Inverting here -- rather than teaching the policy a
    second scale -- is what keeps `routing/policy.py` untouched and keeps a
    single escalation threshold in the system.
    """
    return 1.0 - min(max(confidence, 0.0), 1.0)
