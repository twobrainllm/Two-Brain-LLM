"""The structured self-report: solve what you can, and name what you can't.

`signals/confidence.py` asks the fast brain one question -- *how sure are
you?* -- and gets one number back. That number is enough to pick a brain, but
not enough to **split the work**: a query the local model half-knows still
escalates whole, and the local model's partial answer is discarded.

This module asks for more, in the same single call:

    {"solution": "...", "confidence": 0-100, "unknown": "..."}

`unknown` is the load-bearing addition. It is the sub-question the local model
is telling us it cannot finish, in its own words, and it is what the *second*
call -- to the deep brain -- is actually about. The `solution` is kept and
shown either way. See `docs/ORCHESTRATOR.md`, "Shape C".

Kept pure for the same reason `confidence.py` and `routing/policy.py` are:
text in, parsed values out, no I/O and no reference to a brain, so it can be
unit-tested on its own. `routing/brains.py` calls into this module; nothing
here calls back.

**On `masked_output`.** The schema below accepts a `masked_output` field
because a model asked to be privacy-aware will sometimes volunteer one, and
silently dropping part of a model's reply is worse than recording it. It is
*advisory only* and nothing routes on it. Masking in this system is done by
`privacy/guard.py` before any brain is called (invariant #1), so by the time a
brain sees the query it is already masked -- a model-produced "masked" string
can therefore only be redundant, or wrong. `StructuredAnswer.model_masked_output`
exists for the audit trail; it never decides what crosses the boundary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from two_brain_router.signals.confidence import parse_self_reported

#: Appended to the user turn to ask for the structured reply.
#:
#: **Single-line JSON is required, not preferred.** `NpuFastBrain` stops
#: generation on `"\n\n"` to bound a model that otherwise runs to its token cap
#: (see that class's `_STOP_SEQUENCES`), so a pretty-printed object would be cut
#: off mid-brace and parse as nothing. Asking for one line keeps that guard
#: usable instead of forcing a choice between structure and runaway output.
#:
#: The "empty string if none" clause on `unknown` is also deliberate: left
#: unspecified, small models write the *word* "None" or "N/A" there, which is a
#: non-empty gap and would escalate every single query. `parse_structured`
#: defends against that anyway (`_NULLISH`), but asking correctly is cheaper
#: than parsing around it.
#:
#: The explicit ban on caveats is measured, not defensive boilerplate. Without
#: it, a real run put this in `unknown` for the PII demo query: *"This response
#: assumes the user's authority to address the security matter and does not
#: involve retrieving or handling personal identification numbers directly."*
#: That is a disclaimer about an answer the model **did** give, not a part it
#: failed to give -- and treating it as a gap sent a query the local model had
#: fully answered to the cloud, which on a PII query is precisely the round trip
#: this project exists to avoid. See
#: `data/npu_model/phi-3.5-mini-instruct/_real_structured_inference_log.md`.
STRUCTURED_SUFFIX = (
    "\n\nReply with ONLY a single-line JSON object and nothing else -- no code "
    "fence, no commentary before or after:\n"
    '{"solution": "<answer the parts you are sure about>", '
    '"confidence": <integer 0-100>, '
    '"unknown": "<a specific sub-question you could NOT answer, or an empty '
    'string if you answered all of it>"}\n'
    "Put something in \"unknown\" only if part of the question is still "
    "unanswered. Caveats, assumptions, disclaimers and notes about the answer "
    "you did give do not belong there -- use an empty string for those."
)

#: System-message half of the same instruction, for brains that send one.
STRUCTURED_SYSTEM_PROMPT = (
    "You are a careful, concise on-device assistant. Answer as much of the "
    "question as you genuinely can. If part of it needs knowledge or reasoning "
    "you are not confident about, still answer the rest, and name that missing "
    "part in the 'unknown' field so a larger model can finish it. 'unknown' is "
    "only for a part of the question you left unanswered -- never for caveats "
    "or assumptions about the answer you did give. "
    "Reply with one single-line JSON object and nothing else. Never use blank "
    "lines."
)

#: Field aliases seen from real small models. `solution`/`confidence`/`unknown`
#: is what STRUCTURED_SUFFIX asks for; the rest is what models produce anyway.
_SOLUTION_KEYS = ("solution", "answer", "response", "partial_solution")
_CONFIDENCE_KEYS = ("confidence", "confidence_score", "certainty")
_UNKNOWN_KEYS = (
    "unknown",
    "unknown_part",
    "unknown_parts",
    "cannot_answer",
    "gap",
    "missing",
    "needs_help_with",
)
_MASKED_KEYS = ("masked_output", "masked", "masked_query")

#: Strings a model writes when it means "nothing here". Treating any of these
#: as a real gap would escalate every query, which is the failure mode this
#: whole path exists to avoid.
_NULLISH = frozenset(
    {"", "none", "n/a", "na", "null", "nil", "nothing", "-", "no gaps", "none.", "nothing."}
)

#: Matches a ```json ... ``` fence, which models add despite being told not to.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class StructuredAnswer:
    """One parsed structured reply from a fast brain."""

    #: What the model could answer. May be partial; may be empty if it could
    #: answer nothing at all.
    solution: str
    #: `[0.0, 1.0]`, or None when the model reported no usable number. None and
    #: 0.0 mean different things -- see `confidence.parse_self_reported`.
    confidence: float | None
    #: What the model says it *cannot* answer, in its own words. Empty string
    #: means "nothing left over", which is the common case for an easy query.
    unknown: str = ""
    #: Advisory only, never routed on -- see the module docstring.
    model_masked_output: str = ""
    #: Which rung of the degradation ladder this came off. Reported in the
    #: router's notes so a badly-formatted model is visible rather than
    #: silently downgraded.
    source: Literal["json", "self_report", "raw"] = "json"

    @property
    def has_gap(self) -> bool:
        return bool(self.unknown.strip())


def _coerce_text(value: Any) -> str:
    """Flatten whatever a model put in a string-shaped field.

    Models return lists for `unknown` (one entry per gap) often enough that
    `str(["a", "b"])` -- which would send the literal `['a', 'b']` to the deep
    brain -- is a real outcome worth handling rather than tolerating.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_coerce_text(item) for item in value]
        return "; ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [_coerce_text(item) for item in value.values()]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


def _coerce_confidence(value: Any) -> float | None:
    """Normalise a reported confidence to `[0.0, 1.0]`, or None if unusable.

    The int/float split is not pedantry, it resolves a genuine ambiguity in the
    safe direction. `STRUCTURED_SUFFIX` asks for an integer 0-100, so a bare
    `1` means *one percent* -- but models also ignore the instruction and reply
    on a 0-1 scale, where `0.95` obviously means 95%. Reading the type keeps
    both right:

    - `int` -> always the 0-100 scale. `1` becomes 0.01 (barely confident), and
      that is the safe reading: over-reading it as 1.00 would keep a query
      local that the model just told us it could not do.
    - `float` -> `<= 1.0` is already a fraction (0.95 -> 0.95); above that it is
      the 0-100 scale written with a decimal point (95.0 -> 0.95).

    A string is parsed for the first number it contains, so `"95%"` and
    `"about 90"` still land somewhere sensible, and takes the float branch only
    if it actually wrote a decimal point.
    """
    if isinstance(value, bool):  # bool is an int subclass; a bool is not a score.
        return None
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match is None:
            return None
        raw = match.group(0)
        value = float(raw) if "." in raw else int(raw)
    if isinstance(value, int):
        return min(max(value / 100.0, 0.0), 1.0)
    if isinstance(value, float):
        scaled = value if value <= 1.0 else value / 100.0
        return min(max(scaled, 0.0), 1.0)
    return None


def _first_key(obj: dict, keys: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def _extract_json_object(text: str) -> dict | None:
    """Pull the first complete JSON object out of a model reply.

    Scans for balanced braces rather than regexing, because a `solution` value
    routinely contains braces of its own (code, sets, LaTeX) and a greedy or
    lazy regex gets both cases wrong. String literals are tracked so a `}`
    inside a quoted value doesn't close the object early.
    """
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break  # malformed -- try the next `{`
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def parse_structured(text: str) -> StructuredAnswer:
    """Parse a fast brain's reply, degrading gracefully rather than failing.

    Three rungs, in order, because a 3B quantized model does not always do what
    it was asked and a failure to parse must never be mistaken for a confident
    answer:

    1. **`source="json"`** -- a JSON object was found. The normal path.
    2. **`source="self_report"`** -- no JSON, but a bare `CONFIDENCE: <n>` line
       is there, so `signals/confidence.py` can still read it. The whole reply
       becomes the solution and there is no gap, which makes this exactly the
       Shape B behaviour that predates this module. A model that regresses to
       the older format therefore still routes correctly instead of falling off
       a cliff.
    3. **`source="raw"`** -- neither. `confidence` is None, which the router
       reads as `difficulty = 1.0` (maximally uncertain), *not* as zero
       confidence and not as a reason to reach for a different signal. Same
       rule `docs/ORCHESTRATOR.md` already sets for an unparseable
       `CONFIDENCE:` line.

    Never raises: every rung produces a `StructuredAnswer`, so a brain calling
    this cannot turn a badly-formatted reply into an exception the router would
    have to special-case.
    """
    obj = _extract_json_object(text)
    if obj is None:
        solution, confidence = parse_self_reported(text)
        return StructuredAnswer(
            solution=solution,
            confidence=confidence,
            unknown="",
            source="self_report" if confidence is not None else "raw",
        )

    solution = _coerce_text(_first_key(obj, _SOLUTION_KEYS))
    confidence = _coerce_confidence(_first_key(obj, _CONFIDENCE_KEYS))
    unknown = _coerce_text(_first_key(obj, _UNKNOWN_KEYS))
    if unknown.lower() in _NULLISH:
        unknown = ""

    # A model that emitted valid JSON but no solution key at all told us
    # nothing usable; prefer the raw reply over an empty answer.
    if not solution and not unknown:
        solution = text.strip()

    return StructuredAnswer(
        solution=solution,
        confidence=confidence,
        unknown=unknown,
        model_masked_output=_coerce_text(_first_key(obj, _MASKED_KEYS)),
        source="json",
    )


class SolutionStreamFilter:
    """Emit the `solution` field's text from a structured reply, as it arrives.

    Shape C's local half is JSON, so streaming it raw would show the user
    `{"solution": "` and the escaping around their answer. Buffering instead
    means the blue bubble appears all at once after several seconds, which
    throws away the point of streaming for the half that is usually slower.

    This pulls the one field out mid-flight: feed it raw chunks, get back only
    the characters that belong inside `solution`, unescaped. Everything else --
    the opening brace, the key, the later `confidence` and `unknown` fields --
    is swallowed.

    Deliberately a small hand-rolled scanner rather than an incremental JSON
    parser: it only has to find one string value and stop, and it must tolerate
    output that never becomes valid JSON at all. `feed` returns "" forever in
    that case rather than raising, and `parse_structured` still runs on the
    complete text afterwards, so the authoritative parse is unchanged. This is
    a display optimisation and is never the source of truth.

    Accepts the same key aliases as `parse_structured`, since a model that
    writes `answer` instead of `solution` should still stream.
    """

    def __init__(self, keys: tuple[str, ...] = _SOLUTION_KEYS) -> None:
        self._keys = keys
        self._buf = ""
        self._state = "seek"  # seek -> in_string -> done
        self._escape = False

    def feed(self, chunk: str) -> str:
        if self._state == "done":
            return ""
        self._buf += chunk
        if self._state == "seek":
            start = self._find_value_start()
            if start is None:
                # Keep only a tail long enough to still match a split key.
                if len(self._buf) > 200:
                    self._buf = self._buf[-200:]
                return ""
            self._state = "in_string"
            self._buf = self._buf[start:]
        return self._drain()

    def _find_value_start(self) -> int | None:
        """Index just past the opening quote of the first matching key's value."""
        best: int | None = None
        for key in self._keys:
            for quote in ('"', "'"):
                marker = f"{quote}{key}{quote}"
                at = self._buf.find(marker)
                if at == -1:
                    continue
                rest = self._buf[at + len(marker):]
                stripped = rest.lstrip()
                if not stripped.startswith(":"):
                    continue
                after_colon = rest[rest.index(":") + 1:]
                value = after_colon.lstrip()
                if not value.startswith('"'):
                    continue  # value not started yet, or not a string
                index = at + len(marker) + rest.index(":") + 1 + (len(after_colon) - len(value)) + 1
                if best is None or index < best:
                    best = index
        return best

    def _drain(self) -> str:
        out: list[str] = []
        consumed = 0
        for ch in self._buf:
            consumed += 1
            if self._escape:
                out.append({"n": "\n", "t": "\t", "r": "\r"}.get(ch, ch))
                self._escape = False
                continue
            if ch == "\\":
                self._escape = True
                continue
            if ch == '"':
                self._state = "done"
                break
            out.append(ch)
        self._buf = self._buf[consumed:]
        return "".join(out)
