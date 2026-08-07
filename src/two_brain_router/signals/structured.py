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
#: Three failure modes have earned a line in this prompt so far, in order
#: found -- each is a real run, not a hypothetical, and each fix was checked
#: against the *other* two before landing, because they pull in different
#: directions and an early attempt at #2 broke #1's counterpart:
#:
#: 1. **Caveats are not gaps.** Without the ban, a real run put this in
#:    `unknown` for the PII demo query: *"This response assumes the user's
#:    authority to address the security matter and does not involve retrieving
#:    or handling personal identification numbers directly."* A disclaimer
#:    about an answer the model **did** give, not a part it failed to give --
#:    and treating it as a gap sent a fully-answered PII query to the cloud,
#:    precisely the round trip this project exists to avoid.
#: 2. **Self-invented follow-ups are not gaps either.** For "explain about
#:    stable diffusion" a real run answered a general definition, then put
#:    this in `unknown`: *"How does stable diffusion specifically apply to
#:    environmental science or economic models?"* -- neither field was
#:    mentioned anywhere in the query. The model *widened the question*
#:    rather than reporting a hole in its own answer, and that escalated a
#:    request it considered fully answered.
#: 3. **A fact the model doesn't know must be named, not faked.** The first
#:    fix for #2 was worded as "only a literal part of the question, nothing
#:    invented" -- and on the very next real run, that collapsed a genuine
#:    gap into a false positive: for the population half of the France
#:    question above, the model no longer named a gap *or* answered it. It
#:    emitted `{"solution": "Paris, France's population on 3 March 2019",
#:    "confidence": 100, "unknown": ""}` -- echoing the question fragment
#:    back as if it were the number, at full claimed confidence. That is
#:    worse than #2: a silently wrong "complete" local answer instead of an
#:    honest escalation. The instruction below says explicitly not to do
#:    this -- guessing or restating is banned, and not-knowing has exactly
#:    one correct expression, naming it in `unknown`.
#:
#: #2 and #3 are opposite-direction failures of the same instruction and both
#: have to hold at once: don't invent a gap that isn't there (#2), don't erase
#: a gap that is (#3). Any future wording change must be checked against a
#: query with a real gap (the France/population query below) *and* a query
#: with none (the stable-diffusion one) before it ships -- one without the
#: other is how #3 happened.
#:
#: See `data/npu_model/phi-3.5-mini-instruct/_real_structured_inference_log.md`
#: for the measurements behind all three.
STRUCTURED_SUFFIX = (
    "\n\nReply with ONLY a single-line JSON object and nothing else -- no code "
    "fence, no commentary before or after:\n"
    '{"solution": "<answer the parts you are sure about>", '
    '"confidence": <integer 0-100>, '
    '"unknown": "<a part of THIS question you could not actually answer, in '
    'your own words -- or an empty string if you answered everything>"}\n'
    "If you do not actually know a fact the user asked for, put it in "
    "\"unknown\" -- do not guess, and do not repeat the question back as if it "
    "were the answer. Never put a caveat or assumption about your own answer "
    "in \"unknown\". Never put a new topic or follow-up question there that "
    "the user did not ask, even a reasonable one -- if it was not in the "
    "question, it is not a gap.\n"
    'Example: "What is the capital of France, and its population on 3 March '
    '2019?" -- if you do not know that exact figure, unknown = "the population '
    'on 3 March 2019", not empty and not a guess. "Explain about stable '
    'diffusion" -- once answered, unknown = "" even though you could also '
    "cover its use in economics, because that was never asked."
)

#: System-message half of the same instruction, for brains that send one.
STRUCTURED_SYSTEM_PROMPT = (
    "You are a careful, concise on-device assistant. Answer exactly what the "
    "user asked. If you genuinely do not know or cannot determine a specific "
    "fact the user asked for, say so honestly by naming it in 'unknown' -- do "
    "not guess, and do not restate the question as if it were an answer. "
    "'unknown' is ONLY for a part of the user's own question that you could "
    "not answer -- never for a caveat about the answer you did give, and "
    "never for a new topic, application, or follow-up the user did not ask "
    "about, no matter how relevant it seems. If every part of the question is "
    "genuinely answered, 'unknown' is an empty string. "
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


class SolutionStreamer:
    r"""Emits only the `solution` field's text as a structured reply arrives.

    The problem this exists for: a Shape C brain generates
    `{"solution": "Paris is...", "confidence": 95, "unknown": ""}` token by
    token. Forwarding those tokens straight to a UI shows the user the
    scaffolding -- a brace, a quoted key, a colon -- before any answer appears,
    and the `confidence`/`unknown` fields after it. Those are routing metadata;
    they are not the answer and must never be rendered as one.

    So this walks the raw buffer and hands back only the decoded contents of
    `solution`, as they become available. Feed it whatever chunks arrive, in
    order; each call returns the *new* text since the last call, or `""`.

    Three things make this fiddlier than "wait for valid JSON":

    - **Chunk boundaries fall anywhere.** `"solu` / `tion": "Par` / `is"` is a
      normal split, so nothing can be matched against a single chunk. The whole
      buffer is rescanned each feed instead; replies are a few hundred
      characters, so the quadratic cost is irrelevant next to one NPU token.
    - **JSON escapes must be decoded, and may be truncated.** A buffer ending
      mid-escape (`...\u00e9` cut after `\u00`) has to withhold that fragment
      rather than emit a broken character, and pick it up on the next feed.
    - **The model may not emit JSON at all.** Small models sometimes answer in
      plain prose (`parse_structured`'s lower rungs exist for exactly that). If
      the buffer clearly is not a JSON object, everything is streamed verbatim
      -- degrading to "show the user the words" rather than showing nothing.
    """

    #: Enough characters to tell prose from `{"solution": ...`. A model that
    #: has not opened a brace by here is not going to.
    _JSON_SNIFF_CHARS = 24

    def __init__(self) -> None:
        self._raw: list[str] = []
        self._emitted = 0
        self._plain_text: bool | None = None

    @property
    def raw(self) -> str:
        """Everything fed so far, for the caller to parse properly at the end."""
        return "".join(self._raw)

    def feed(self, chunk: str) -> str:
        """Add `chunk`; return whatever new answer text that made available."""
        if not chunk:
            return ""
        self._raw.append(chunk)
        buffer = self.raw

        if self._plain_text is None:
            stripped = buffer.lstrip()
            if stripped.startswith("{") or stripped.startswith("```"):
                self._plain_text = False
            elif len(stripped) >= self._JSON_SNIFF_CHARS:
                # Committed: no object is coming, so stream the prose as-is.
                self._plain_text = True
            else:
                return ""  # too early to tell -- hold rather than guess wrong

        available = buffer if self._plain_text else _partial_solution(buffer)
        if available is None or len(available) <= self._emitted:
            return ""
        new = available[self._emitted :]
        self._emitted = len(available)
        return new

    def finish(self) -> str:
        """Any remaining text, once the stream is known to be complete.

        Covers the case where the sniff never resolved -- a reply shorter than
        `_JSON_SNIFF_CHARS` that turned out to be prose, which would otherwise
        be withheld forever.
        """
        buffer = self.raw
        if self._plain_text is None:
            self._plain_text = not buffer.lstrip().startswith(("{", "```"))
        available = buffer if self._plain_text else _partial_solution(buffer)
        if available is None or len(available) <= self._emitted:
            return ""
        new = available[self._emitted :]
        self._emitted = len(available)
        return new


def _partial_solution(buffer: str) -> str | None:
    """Decode as much of the `solution` string as `buffer` contains.

    Returns None while the key has not been seen yet -- distinct from `""`,
    which means "the key is open and so far it is empty".
    """
    for key in _SOLUTION_KEYS:
        start = _value_start(buffer, key)
        if start is not None:
            return _decode_json_string_prefix(buffer, start)
    return None


def _value_start(buffer: str, key: str) -> int | None:
    """Index just past the opening quote of `"<key>" : "`, or None."""
    needle = f'"{key}"'
    at = buffer.find(needle)
    if at == -1:
        return None
    i = at + len(needle)
    while i < len(buffer) and buffer[i] in " \t\r\n":
        i += 1
    if i >= len(buffer) or buffer[i] != ":":
        return None
    i += 1
    while i < len(buffer) and buffer[i] in " \t\r\n":
        i += 1
    if i >= len(buffer) or buffer[i] != '"':
        return None
    return i + 1


def _decode_json_string_prefix(buffer: str, start: int) -> str:
    r"""Decode a JSON string body from `start` until its close quote or the end.

    A trailing incomplete escape is dropped rather than guessed at: the next
    feed will carry the rest of it, and emitting half of a `\uXXXX` would put
    a broken character on screen that can never be taken back.
    """
    out: list[str] = []
    i = start
    n = len(buffer)
    simple = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    while i < n:
        ch = buffer[i]
        if ch == '"':
            break  # closing quote -- the value is complete
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            break  # dangling backslash; wait for more
        esc = buffer[i + 1]
        if esc == "u":
            if i + 6 > n:
                break  # truncated \uXXXX
            try:
                out.append(chr(int(buffer[i + 2 : i + 6], 16)))
            except ValueError:
                out.append(buffer[i : i + 6])
            i += 6
            continue
        out.append(simple.get(esc, esc))
        i += 2
    return "".join(out)
