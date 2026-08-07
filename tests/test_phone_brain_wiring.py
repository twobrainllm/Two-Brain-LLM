"""The Mobile tier's served fast brain, and how the router treats it.

Named `test_phone_brain_wiring.py` rather than `test_phone_brain.py` on
purpose: `src/phone_brain/test_phone_brain.py` on the `local_brain` branch is
a *benchmark* that talks to a live endpoint, and pytest's
`python_files = ["test_*.py"]` would happily collect it and hang. See
docs/PHONE_BRAIN.md reconciliation point #5.

Nothing here needs a phone, a network, or the mock server: the HTTP layer is
stubbed at `_post`, so these run in the standard offline suite.
"""
from __future__ import annotations

import pytest

from two_brain_router.routing.brains import (
    BrainUnavailableError,
    LocalFastBrain,
    OpenAIHttpBrain,
)
from two_brain_router.routing.router import TwoBrainRouter, _build_fast_brain
from two_brain_router.signals import TierSignals


@pytest.fixture
def mobile_signals() -> TierSignals:
    return TierSignals.load("mobile_1b", "mobile")


def _ok_body(text: str = "42") -> dict:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


# --------------------------------------------------------------------------
# The loopback guard -- docs/PHONE_BRAIN.md reconciliation point #1.
# --------------------------------------------------------------------------


def test_remote_host_is_refused(mobile_signals):
    """The mobile tier's whole claim is that the query stays on the device.

    A `base_url` pointing off-box would send it in the clear, so it must not
    be constructible by accident.
    """
    with pytest.raises(ValueError, match="non-loopback"):
        OpenAIHttpBrain("mobile", mobile_signals, base_url="http://192.168.1.50:8000")


@pytest.mark.parametrize("host", ["http://127.0.0.1:8000", "http://localhost:8000"])
def test_loopback_hosts_are_accepted(mobile_signals, host):
    """`adb forward` makes the phone reachable on 127.0.0.1, so the strict
    default costs nothing in the intended USB setup."""
    brain = OpenAIHttpBrain("mobile", mobile_signals, base_url=host)
    assert brain.base_url == host


def test_remote_host_allowed_only_when_explicit(mobile_signals):
    brain = OpenAIHttpBrain(
        "mobile", mobile_signals, base_url="http://192.168.1.50:8000", allow_remote=True
    )
    assert brain.base_url == "http://192.168.1.50:8000"


def test_env_var_url_is_also_guarded(mobile_signals, monkeypatch):
    """The guard has to cover the env-var path too, or it is trivially bypassed."""
    monkeypatch.setenv("TWO_BRAIN_PHONE_URL", "http://10.0.0.5:8000")
    with pytest.raises(ValueError, match="non-loopback"):
        OpenAIHttpBrain("mobile", mobile_signals)


# --------------------------------------------------------------------------
# The contract -- docs/L_INTERFACE_CONTRACT.md.
# --------------------------------------------------------------------------


def test_answer_parses_the_contract_response(mobile_signals, monkeypatch):
    brain = OpenAIHttpBrain("mobile", mobile_signals)
    monkeypatch.setattr(brain, "_post", lambda messages: (_ok_body("the answer"), 123.4))

    response = brain.answer("what is 6 times 7?")

    assert response.text == "the answer"
    assert response.latency_ms == 123.4
    # On-device inference is not billed per token.
    assert response.cost_usd == 0.0


def test_context_is_sent_as_a_system_message(mobile_signals, monkeypatch):
    brain = OpenAIHttpBrain("mobile", mobile_signals)
    captured: list[list[dict]] = []

    def fake_post(messages):
        captured.append(messages)
        return _ok_body(), 10.0

    monkeypatch.setattr(brain, "_post", fake_post)
    brain.answer("q", context="prior turn")

    assert captured[0] == [
        {"role": "system", "content": "prior turn"},
        {"role": "user", "content": "q"},
    ]


def test_malformed_response_is_unavailable_not_a_crash(mobile_signals, monkeypatch):
    """A garbled body is a dead endpoint, not a bug in the router -- so it has
    to raise the type the router knows to escalate on."""
    brain = OpenAIHttpBrain("mobile", mobile_signals)
    monkeypatch.setattr(brain, "_post", lambda messages: ({"unexpected": True}, 5.0))

    with pytest.raises(BrainUnavailableError):
        brain.answer("q")


# --------------------------------------------------------------------------
# Router wiring.
# --------------------------------------------------------------------------


def test_gate_is_off_by_default(monkeypatch, mobile_signals):
    """Unset, the mobile tier keeps its stub -- the base package stays
    stdlib-only and offline, same as the NPU/GPU gates."""
    monkeypatch.delenv("TWO_BRAIN_PHONE_BRAIN", raising=False)
    assert isinstance(_build_fast_brain("mobile", mobile_signals), LocalFastBrain)


def test_gate_selects_the_served_brain(monkeypatch, mobile_signals):
    monkeypatch.setenv("TWO_BRAIN_PHONE_BRAIN", "1")
    monkeypatch.delenv("TWO_BRAIN_PHONE_URL", raising=False)
    assert isinstance(_build_fast_brain("mobile", mobile_signals), OpenAIHttpBrain)


def test_gate_does_not_leak_into_the_pc_tier(monkeypatch, mobile_signals):
    """The phone gate is mobile-only; the AI-PC tier runs in-process."""
    monkeypatch.setenv("TWO_BRAIN_PHONE_BRAIN", "1")
    assert isinstance(_build_fast_brain("pc", mobile_signals), LocalFastBrain)


# --------------------------------------------------------------------------
# "Unreachable means escalate" -- L_INTERFACE_CONTRACT.md's error handling.
# --------------------------------------------------------------------------


def test_dead_phone_escalates_instead_of_failing(monkeypatch):
    """A phone that is unplugged, asleep, or thermally throttled must not take
    the user's request down with it."""
    router = TwoBrainRouter(tier="mobile")

    def dead(*args, **kwargs):
        raise BrainUnavailableError("phone brain unreachable: connection refused")

    monkeypatch.setattr(router.fast_brain, "answer", dead)

    decision = router.route("what is 6 times 7?")

    assert decision.tier_answered == "cloud"
    assert any("local brain unavailable" in note for note in decision.notes)


def test_escalation_fallback_still_masks(monkeypatch):
    """The fallback path must not become a hole in invariant #3 -- it goes
    through _escalate, so PII is still masked before it leaves."""
    router = TwoBrainRouter(tier="mobile")

    def dead(*args, **kwargs):
        raise BrainUnavailableError("phone brain unreachable")

    monkeypatch.setattr(router.fast_brain, "answer", dead)

    decision = router.route("email bob@example.com about the meeting")

    assert decision.tier_answered == "cloud"
    assert decision.pii_entities_masked >= 1
    assert "bob@example.com" not in "".join(decision.notes)


def test_other_brain_errors_still_raise(monkeypatch):
    """Only BrainUnavailableError falls through. A genuine bug in a brain must
    not be silently converted into a cloud call."""
    router = TwoBrainRouter(tier="mobile")

    def broken(*args, **kwargs):
        raise ValueError("a real bug")

    monkeypatch.setattr(router.fast_brain, "answer", broken)

    with pytest.raises(ValueError, match="a real bug"):
        router.route("what is 6 times 7?")
