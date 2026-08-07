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


def test_force_tier_is_ignored_unless_ui_test_is_enabled(monkeypatch):
    """A demo affordance must not become production routing behaviour.

    force_tier exists for the chat UI's local/cloud switch. Outside UI_TEST=1
    it must be refused -- and refused *visibly*, so a caller who expected it to
    work finds out from the notes rather than by misreading which brain
    answered.
    """
    monkeypatch.delenv("UI_TEST", raising=False)
    router = TwoBrainRouter(tier="pc")

    decision = router.route("What is 2 + 2?", force_tier="cloud")

    assert decision.tier_answered == "local", "policy should still decide"
    assert any("ignored force_tier=cloud" in n for n in decision.notes)


def test_force_tier_is_honoured_under_ui_test(monkeypatch):
    monkeypatch.setenv("UI_TEST", "1")
    router = TwoBrainRouter(tier="pc")

    forced = router.route("What is 2 + 2?", force_tier="cloud")
    assert forced.tier_answered == "cloud"
    assert any("forced to cloud" in n and "UI_TEST=1" in n for n in forced.notes)

    # ...and the other direction, on a query the policy would escalate.
    hard = (
        "Derive the closed-form solution for ridge regression from first principles, "
        "then compare its bias-variance trade-off against ordinary least squares in depth."
    )
    assert router.route(hard).tier_answered == "cloud", "sanity: this should escalate"
    assert router.route(hard, force_tier="local").tier_answered == "local"


def test_forcing_a_tier_does_not_bypass_masking(monkeypatch):
    """Forcing must skip the difficulty heuristic and nothing else."""
    monkeypatch.setenv("UI_TEST", "1")
    router = TwoBrainRouter(tier="pc")

    decision = router.route(
        "My email is jane.doe@example.com -- what is 2 + 2?", force_tier="cloud"
    )

    assert decision.pii_entities_masked >= 1
    sent = next(n for n in decision.notes if n.startswith("sent off-device"))
    assert "jane.doe@example.com" not in sent
    assert "jane.doe@example.com" in decision.answer, "must still rehydrate on-device"
