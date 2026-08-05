"""Entity patterns for the mock PII detector.

Regex-only by design: this is the stand-in for `quad.privacy`'s real detector
(gap G8), and a regex table makes the mock's coverage obvious rather than
implied. Add an entry here to widen coverage; the guard picks it up with no
other change.
"""
from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}
