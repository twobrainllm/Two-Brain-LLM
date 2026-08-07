"""Privacy guardrail stub for the Two-Brain router.

Stand-in for `quad.privacy` -- the real PII guardrail (entity detection, masking,
rehydration, masked-token invariant) tracked as gap **G8** in QUAD's private
core-platform repo, marked `[G8 - DELIVERED]` there. That component is wired
into a chat product elsewhere and is not part of this checked-out
`QUAD-Client-main` repo (no `quad.privacy` module exists anywhere in this
tree -- confirmed by a full-repo search). Per the documented workaround for
gaps pending real-hardware validation, this module implements the same
detect/mask/rehydrate contract with a regex-based mock detector so the router
around it is real and testable; swap this module for `quad.privacy` once that
gap closes in this environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from two_brain_router.privacy.patterns import PATTERNS


@dataclass
class MaskResult:
    masked_text: str
    vault: dict[str, str] = field(default_factory=dict)


class PIIGuard:
    """Mock stand-in for quad.privacy's detect/mask/rehydrate contract (gap G8)."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        #: value -> placeholder, so the same entity masks identically across
        #: every call on this guard and the vaults can be merged safely.
        self._seen: dict[str, str] = {}

    def detect(self, text: str) -> list[tuple[str, str]]:
        """Return [(entity_type, matched_value), ...] in match order."""
        hits: list[tuple[int, str, str]] = []
        for entity_type, pattern in PATTERNS.items():
            for m in pattern.finditer(text):
                hits.append((m.start(), entity_type, m.group(0)))
        hits.sort(key=lambda h: h[0])
        return [(t, v) for _, t, v in hits]

    def mask(self, text: str) -> MaskResult:
        """Replace every detected entity with a `[PII_<TYPE>_<n>]` placeholder.

        Numbering continues across calls on the same guard (`self._counters`),
        and the router creates one guard per request. This matters: a request
        masks the query, then separately masks the context and -- for an
        image-bearing query -- the locally generated description. With
        per-call numbering, two different emails in two different calls would
        both become `[PII_EMAIL_1]`, and merging their vaults would silently
        drop one and rehydrate the *wrong* value into the user's answer.

        The same value seen twice within one request maps to the same
        placeholder, which is what makes the vaults safe to merge.
        """
        vault: dict[str, str] = {}
        masked = text
        # Replace longest matches first so a credit-card-shaped substring inside
        # a longer match doesn't get double-masked.
        entities = sorted(set(self.detect(text)), key=lambda e: -len(e[1]))
        for entity_type, value in entities:
            existing = self._seen.get(value)
            if existing is None:
                self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
                existing = f"[PII_{entity_type}_{self._counters[entity_type]}]"
                self._seen[value] = existing
            vault[existing] = value
            masked = masked.replace(value, existing)
        return MaskResult(masked_text=masked, vault=vault)

    def rehydrate(self, text: str, vault: dict[str, str]) -> str:
        """Invert `mask`: swap every placeholder back to its real value."""
        out = text
        for placeholder, value in vault.items():
            out = out.replace(placeholder, value)
        return out


def assert_masked_token_invariant(original: str, result: MaskResult) -> None:
    """The invariant this module must hold before any masked text leaves the
    device boundary: no raw entity value survives in `masked_text`, and
    rehydrating restores the original text exactly.

    Raises AssertionError with a diagnostic message on violation -- callers in
    the router treat this as fatal (never send unmasked PII to the deep brain).
    """
    guard = PIIGuard()
    for placeholder, value in result.vault.items():
        if value in result.masked_text:
            raise AssertionError(
                f"masked-token invariant violated: raw value for {placeholder!r} "
                f"still present in masked_text"
            )
    restored = guard.rehydrate(result.masked_text, result.vault)
    if restored != original:
        raise AssertionError(
            "masked-token invariant violated: rehydrate(mask(x)) != x\n"
            f"  original: {original!r}\n  restored: {restored!r}"
        )
