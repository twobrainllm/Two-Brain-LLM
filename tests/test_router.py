import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from privacy_mask import PIIGuard, assert_masked_token_invariant
from router import TwoBrainRouter


def test_mask_rehydrate_round_trip():
    guard = PIIGuard()
    original = "Contact jane.doe@example.com or 555-123-4567, SSN 123-45-6789."
    result = guard.mask(original)
    assert "jane.doe@example.com" not in result.masked_text
    assert "555-123-4567" not in result.masked_text
    assert "123-45-6789" not in result.masked_text
    assert_masked_token_invariant(original, result)
    assert guard.rehydrate(result.masked_text, result.vault) == original


def test_mask_no_pii_is_noop():
    guard = PIIGuard()
    original = "What time zone is Tokyo in?"
    result = guard.mask(original)
    assert result.masked_text == original
    assert result.vault == {}


def test_easy_query_stays_local():
    router = TwoBrainRouter(tier="pc")
    decision = router.route("What time zone is Tokyo in?")
    assert decision.tier_answered == "local"
    assert decision.est_cost_usd == 0.0


def test_hard_query_escalates_to_cloud():
    router = TwoBrainRouter(tier="pc")
    decision = router.route(
        "Derive the time complexity of merge sort step by step and compare "
        "it to quicksort's worst case, then explain the trade-offs."
    )
    assert decision.tier_answered == "cloud"
    assert decision.est_cost_usd > 0.0


def test_escalated_pii_never_reaches_cloud_unmasked():
    router = TwoBrainRouter(tier="pc")
    query = (
        "My email is jane.doe@example.com -- derive the time complexity of "
        "merge sort step by step and compare it to quicksort's worst case."
    )
    decision = router.route(query)
    assert decision.tier_answered == "cloud"
    assert decision.pii_entities_masked >= 1
    sent_off_device = next(n for n in decision.notes if n.startswith("sent off-device"))
    assert "jane.doe@example.com" not in sent_off_device
    # the final answer shown to the user is rehydrated back to the real value
    assert "jane.doe@example.com" in decision.answer


@pytest.mark.parametrize("tier", ["mobile", "pc"])
def test_router_loads_signals_for_both_local_tiers(tier):
    router = TwoBrainRouter(tier=tier)
    assert router.local.hardware["device_tier"] in ("Mobile", "AI PC")
    assert router.cloud.hardware["device_tier"] == "Cloud"
