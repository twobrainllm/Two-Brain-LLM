import pytest

from two_brain_router.routing import RoutePolicy, TwoBrainRouter


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


def test_policy_escalates_on_latency_budget_alone():
    """A trivial query still escalates if the fast brain can't meet the budget."""
    policy = RoutePolicy()
    assert policy.should_escalate(difficulty=0.0, local_latency_est_ms=999_999)
    assert not policy.should_escalate(difficulty=0.0, local_latency_est_ms=10)


def test_policy_compresses_only_oversized_context():
    policy = RoutePolicy(max_context_chars=50)
    short, was_compressed = policy.compress_context("still small")
    assert (short, was_compressed) == ("still small", False)

    long, was_compressed = policy.compress_context("x" * 200)
    assert was_compressed
    assert len(long) <= 50
