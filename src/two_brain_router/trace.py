"""Terminal trace of one routed request: every call in, every call out, and
exactly what was masked before it left the device.

`RouteDecision.notes` already records *what the router decided and why*. This
records **the payloads** -- the strings actually handed to each brain and the
strings each one returned. That is a different question, and the one that
matters for auditing this sample: "did the raw email address reach the cloud?"
is not answerable from a note saying "masked 3 entities", only from seeing the
bytes.

Sample output for a split query::

    ====================================================================
    REQUEST  tier=pc
      query (raw) : What is the capital of France, and its population on 3 March 2019?
      PII detected: none
    - CALL  fast brain | NpuFastBrain   [ON-DEVICE -- raw text]
      query   : What is the capital of France, and its population on 3 March 2019?
    - RETURN  NpuFastBrain  3075ms
      text       : Paris, 2.1 million
      confidence : 0.95
      unknown    : The specific population on 3 March 2019
    - BOUNDARY  -> CloudDeepBrain   [OFF-DEVICE -- masked]
      masked 0 entities
      query   : What is the capital of France, and its population on 3 March 2019?
      context : A smaller on-device model has already answered part of this...
    - RETURN  CloudDeepBrain  1500ms
      text       : ...
    - DECISION  hybrid  difficulty=0.05  4575ms  $0.00002
      PII detected 0 / masked 0
    ====================================================================

**This trace prints raw PII to your terminal.** That is deliberate -- you cannot
verify that PII stayed on-device without seeing that it was there -- but it does
mean the trace is itself a place user data ends up. It is off by default,
prints to stdout and never to a file, and `TWO_BRAIN_TRACE_REDACT=1` shows
placeholders in the `[ON-DEVICE]` sections instead if you want to demo this in
front of an audience.

stdlib only (`sys`, `os`, `textwrap`) -- this package declares zero runtime
dependencies. No `logging` handlers are configured either, so importing this
module cannot change logging behaviour for whatever embeds it.
"""
from __future__ import annotations

import os
import sys
import textwrap
from typing import TextIO

#: Enable the trace. Off by default so the library and the test suite stay
#: silent; `cli.py` and `api.py` turn it on themselves, since a human watching a
#: terminal is exactly who it is for.
TRACE_ENV_VAR = "TWO_BRAIN_TRACE"
#: Show placeholders instead of real values even in the on-device sections.
REDACT_ENV_VAR = "TWO_BRAIN_TRACE_REDACT"
#: Per-field character budget before a value is elided.
MAX_CHARS_ENV_VAR = "TWO_BRAIN_TRACE_MAX"

_RULE = "=" * 74
_DEFAULT_MAX_CHARS = 1200


def trace_enabled(default: bool = False) -> bool:
    raw = os.environ.get(TRACE_ENV_VAR)
    if raw is None:
        return default
    return raw != "0"


class Tracer:
    """Writes the trace, or does nothing at all.

    A disabled `Tracer` is a real object with the same methods rather than
    `None`, so the router never guards a call site with `if self.tracer:` --
    which is how a trace call gets forgotten in a new branch.
    """

    def __init__(
        self,
        enabled: bool = False,
        stream: TextIO | None = None,
        redact: bool | None = None,
        max_chars: int | None = None,
    ) -> None:
        self.enabled = enabled
        self._stream = stream if stream is not None else sys.stdout
        self._redact = (os.environ.get(REDACT_ENV_VAR) == "1") if redact is None else redact
        if max_chars is None:
            try:
                max_chars = int(os.environ.get(MAX_CHARS_ENV_VAR, _DEFAULT_MAX_CHARS))
            except ValueError:
                max_chars = _DEFAULT_MAX_CHARS
        self._max_chars = max_chars

    # -- plumbing ---------------------------------------------------------

    def _write(self, line: str = "") -> None:
        if not self.enabled:
            return
        print(line, file=self._stream, flush=True)

    def _field(self, label: str, value: object, indent: str = "  ") -> None:
        """One `label : value` line, wrapped and elided rather than unbounded.

        A model's answer or a compressed context can run to thousands of
        characters; dumping them whole turns the trace into something nobody
        reads, which defeats the point of having one.
        """
        text = "" if value is None else str(value)
        if len(text) > self._max_chars:
            text = f"{text[: self._max_chars]}... (+{len(text) - self._max_chars} more chars)"
        pad = " " * (len(indent) + 12)
        first = f"{indent}{label:<10} : "
        if not text:
            self._write(f"{first}(empty)")
            return
        wrapped = textwrap.wrap(text, width=120 - len(pad), replace_whitespace=False) or [""]
        self._write(first + wrapped[0])
        for line in wrapped[1:]:
            self._write(pad + line)

    def _maybe_redact(self, text: str, vault: dict[str, str] | None) -> str:
        """Swap real values back out for placeholders, when asked.

        Only meaningful for on-device text, which is raw by construction. Uses
        the vault built for *this* request, so it can only redact what the guard
        already knows how to find -- it is a presentation aid, not a second
        masking implementation, and must never be mistaken for one.
        """
        if not self._redact or not vault:
            return text
        for placeholder, value in vault.items():
            text = text.replace(value, placeholder)
        return text

    # -- events -----------------------------------------------------------

    def request(self, tier: str, query: str, context: str, image: object, detected: list[tuple[str, str]]) -> None:
        if not self.enabled:
            return
        self._write()
        self._write(_RULE)
        self._write(f"REQUEST  tier={tier}")
        vault = {v: f"[PII_{t}_?]" for t, v in detected} if self._redact else None
        self._field("query (raw)", self._maybe_redact(query, vault))
        if context:
            self._field("context", self._maybe_redact(context, vault))
        if image is not None:
            self._field("image", image)
        if detected:
            summary = ", ".join(f"{t}={v!r}" for t, v in detected) if not self._redact else \
                ", ".join(t for t, _ in detected)
            self._field("PII found", f"{len(detected)} -- {summary}")
        else:
            self._field("PII found", "none")

    def call(
        self,
        role: str,
        brain: object,
        text: str,
        context: str = "",
        crossing: bool = False,
        vault: dict[str, str] | None = None,
    ) -> None:
        """One outbound brain call.

        `crossing` is the field to look at: it is True exactly when this call
        leaves the device, and it is what the `[OFF-DEVICE -- masked]` banner is
        driven by. Everything under an `[ON-DEVICE]` banner stayed here.
        """
        if not self.enabled:
            return
        name = type(brain).__name__
        if crossing:
            self._write(f"- BOUNDARY  -> {name}   [OFF-DEVICE -- masked]")
            if vault:
                for placeholder, value in vault.items():
                    self._write(f"    masked  {value!r} -> {placeholder}")
            else:
                self._write("    masked  nothing (no PII in what crosses)")
        else:
            trusted = getattr(brain, "trusted_with_raw_pii", False)
            banner = "[ON-DEVICE -- raw text]" if trusted else "[OFF-DEVICE -- masked]"
            self._write(f"- CALL  {role} | {name}   {banner}")
        self._field("query", self._maybe_redact(text, vault if not crossing else None))
        if context:
            self._field("context", self._maybe_redact(context, vault if not crossing else None))

    def result(self, brain: object, response: object, vault: dict[str, str] | None = None) -> None:
        if not self.enabled:
            return
        name = type(brain).__name__
        latency = getattr(response, "latency_ms", 0.0)
        self._write(f"- RETURN  {name}  {latency:.0f}ms")
        self._field("text", self._maybe_redact(str(getattr(response, "text", "")), vault))
        confidence = getattr(response, "confidence", None)
        if confidence is not None:
            self._field("confidence", f"{confidence:.2f}")
        unknown = getattr(response, "unknown", "")
        if unknown:
            self._field("unknown", self._maybe_redact(unknown, vault))
        error = getattr(response, "error", None)
        if error:
            self._field("error", error)
        cost = getattr(response, "cost_usd", 0.0)
        if cost:
            self._field("cost", f"${cost:.6f}")

    def rehydrated(self, vault: dict[str, str]) -> None:
        if not self.enabled or not vault:
            return
        self._write(f"- REHYDRATE  {len(vault)} placeholder(s) restored on-device")
        for placeholder, value in vault.items():
            self._write(f"    {placeholder} -> {value!r}")

    def decision(self, decision: object) -> None:
        if not self.enabled:
            return
        tier = getattr(decision, "tier_answered", "?")
        self._write(
            f"- DECISION  {tier}  difficulty={getattr(decision, 'difficulty_score', 0.0):.2f}  "
            f"{getattr(decision, 'est_latency_ms', 0.0):.0f}ms  "
            f"${getattr(decision, 'est_cost_usd', 0.0):.5f}"
        )
        detected = getattr(decision, "pii_entities_detected", 0)
        masked = getattr(decision, "pii_entities_masked", 0)
        verdict = "none left the device" if masked == 0 else f"{masked} masked before leaving"
        self._field("PII", f"{detected} detected -- {verdict}")
        if getattr(decision, "gap", None):
            self._field("gap", decision.gap)
            self._field("local", getattr(decision, "local_answer", ""))
            self._field("cloud", getattr(decision, "cloud_answer", ""))
        else:
            self._field("answer", getattr(decision, "answer", ""))
        self._write(_RULE)
