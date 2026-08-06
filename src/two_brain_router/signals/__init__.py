"""Tool-response loading and the difficulty signal that drives escalation."""
from __future__ import annotations

from two_brain_router.signals.confidence import (
    SELF_REPORT_SUFFIX,
    confidence_to_difficulty,
    parse_self_reported,
)
from two_brain_router.signals.difficulty import HARD_QUERY_MARKERS, DifficultyEstimator
from two_brain_router.signals.loader import (
    DATA_DIR,
    PROJECT_ROOT,
    TierSignals,
    load_tool_response,
)

__all__ = [
    "DifficultyEstimator",
    "HARD_QUERY_MARKERS",
    "SELF_REPORT_SUFFIX",
    "confidence_to_difficulty",
    "parse_self_reported",
    "TierSignals",
    "load_tool_response",
    "DATA_DIR",
    "PROJECT_ROOT",
]
