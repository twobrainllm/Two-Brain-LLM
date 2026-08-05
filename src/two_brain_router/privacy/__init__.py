"""PII guardrail seam -- mock stand-in for `quad.privacy` (gap G8)."""
from __future__ import annotations

from two_brain_router.privacy.guard import (
    MaskResult,
    PIIGuard,
    assert_masked_token_invariant,
)
from two_brain_router.privacy.patterns import PATTERNS

__all__ = ["MaskResult", "PIIGuard", "assert_masked_token_invariant", "PATTERNS"]
