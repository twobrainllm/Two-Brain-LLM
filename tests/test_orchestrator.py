"""Orchestrator tests for the confidence-driven mobile-tier routing path.

These cover `PhoneFastBrain` and the `route()` branch it activates. They stand
up a real HTTP server speaking the `L_INTERFACE_CONTRACT.md` shape on loopback
rather than mocking `urllib`, so the request that actually goes out over the
socket is the thing being asserted on -- which is the only way the "only masked
text leaves the device" test means anything.

Self-contained on purpose: no phone, no `src/phone_brain/mock_phone_brain_server.py`
subprocess, no third-party deps. `src/phone_brain/` is a directory of scripts
rather than an importable package, and coupling the suite to it would make
these tests fail for reasons that have nothing to do with the router.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from two_brain_router.routing import (
    PhoneFastBrain,
    RemoteBrainRefused,
    RoutePolicy,
    TwoBrainRouter,
)
from two_brain_router.signals.confidence import confidence_to_difficulty, parse_self_reported
from two_brain_router.signals.loader import TierSignals

# --------------------------------------------------------------------------
# A real, tiny L-contract server
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 -- BaseHTTPRequestHandler's required name
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) or b"{}"
        self.server.received.append(json.loads(raw))

        if self.server.status != 200:
            self.send_response(self.server.status)
            self.end_headers()
            return

        text = self.server.reply_text
        payload = json.dumps(
            {
                "id": "test-cmpl-1",
                "object": "chat.completion",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": max(1, len(text.split())),
                    "total_tokens": 1 + max(1, len(text.split())),
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep pytest output clean
        pass


class _PhoneServer:
    """Loopback stand-in for L, recording every request body it receives."""

    def __init__(self, reply_text: str, status: int = 200) -> None:
        self._httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.received = []
        self._httpd.reply_text = reply_text
        self._httpd.status = status
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_PhoneServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"

    @property
    def received(self) -> list[dict]:
        return self._httpd.received

    def prompts(self) -> list[str]:
        return [body["messages"][-1]["content"] for body in self.received]


def _mobile_signals() -> TierSignals:
    return TierSignals.load("mobile_1b", "mobile")


def _phone_brain(server: _PhoneServer, **kwargs) -> PhoneFastBrain:
    return PhoneFastBrain("mobile", _mobile_signals(), base_url=server.base_url, **kwargs)


def _mobile_router(server: _PhoneServer, monkeypatch, **policy_kwargs) -> TwoBrainRouter:
    monkeypatch.setenv("TWO_BRAIN_PHONE_BRAIN", "1")
    monkeypatch.setenv("TWO_BRAIN_PHONE_URL", server.base_url)
    policy = RoutePolicy(**policy_kwargs) if policy_kwargs else None
    return TwoBrainRouter(tier="mobile", policy=policy)


# --------------------------------------------------------------------------
# The pure confidence helpers
# --------------------------------------------------------------------------


def test_parse_self_reported_extracts_and_strips_the_confidence_line():
    answer, confidence = parse_self_reported("Tokyo is UTC+9.\nCONFIDENCE: 88")
    assert answer == "Tokyo is UTC+9."
    assert confidence == pytest.approx(0.88)


def test_parse_self_reported_returns_none_when_the_model_ignored_the_format():
    """None (no signal) must stay distinct from 0.0 (definitely escalate)."""
    answer, confidence = parse_self_reported("Tokyo is UTC+9.")
    assert answer == "Tokyo is UTC+9."
    assert confidence is None


def test_parse_self_reported_clamps_out_of_range_values():
    _, confidence = parse_self_reported("...\nCONFIDENCE: 150")
    assert confidence == 1.0


def test_confidence_inverts_to_difficulty():
    assert confidence_to_difficulty(1.0) == 0.0
    assert confidence_to_difficulty(0.0) == 1.0
    assert confidence_to_difficulty(0.75) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# PhoneFastBrain
# --------------------------------------------------------------------------


def test_phone_brain_reports_confidence_and_returns_a_clean_answer():
    with _PhoneServer("Tokyo is UTC+9.\nCONFIDENCE: 91") as server:
        response = _phone_brain(server).answer("What time zone is Tokyo in?")

    assert response.text == "Tokyo is UTC+9."
    assert response.confidence == pytest.approx(0.91)
    assert response.error is None
    assert response.cost_usd == 0.0
    assert response.latency_ms > 0


def test_phone_brain_asks_the_model_to_self_rate():
    with _PhoneServer("ok\nCONFIDENCE: 80") as server:
        _phone_brain(server).answer("hello")
        sent = server.prompts()[0]

    assert sent.startswith("hello")
    assert "CONFIDENCE:" in sent


def test_phone_brain_treats_a_failed_call_as_an_escalate_signal():
    """Per L_INTERFACE_CONTRACT.md: non-200 means 'L failed' -> escalate,
    not an exception the user sees."""
    with _PhoneServer("unused", status=503) as server:
        response = _phone_brain(server).answer("hello")

    assert response.confidence == 0.0
    assert confidence_to_difficulty(response.confidence) == 1.0
    assert response.error is not None
    assert response.text == ""


def test_phone_brain_flags_an_unparseable_confidence_line():
    with _PhoneServer("I have opinions but no numbers.") as server:
        response = _phone_brain(server).answer("hello")

    assert response.confidence is None
    assert response.error is not None
    assert response.text == "I have opinions but no numbers."


def test_phone_brain_refuses_a_non_on_device_host():
    with pytest.raises(RemoteBrainRefused):
        PhoneFastBrain("mobile", _mobile_signals(), base_url="http://example.com:8000")


def test_phone_brain_allows_a_remote_host_only_on_explicit_opt_in():
    brain = PhoneFastBrain(
        "mobile", _mobile_signals(), base_url="http://example.com:8000", allow_remote=True
    )
    assert brain.base_url == "http://example.com:8000"


# --------------------------------------------------------------------------
# The router's confidence path
# --------------------------------------------------------------------------


def test_confident_fast_brain_answers_locally(monkeypatch):
    with _PhoneServer("Tokyo is UTC+9.\nCONFIDENCE: 92") as server:
        decision = _mobile_router(server, monkeypatch).route("What time zone is Tokyo in?")

    assert decision.tier_answered == "local"
    assert decision.answer == "Tokyo is UTC+9."
    assert decision.est_cost_usd == 0.0
    assert decision.difficulty_score == pytest.approx(0.08)
    assert len(server.received) == 1, "the local path must not cost a second inference"


def test_unconfident_fast_brain_escalates_and_owns_the_discarded_cost(monkeypatch):
    with _PhoneServer("Uh, maybe?\nCONFIDENCE: 20") as server:
        decision = _mobile_router(server, monkeypatch).route("Prove the Riemann hypothesis.")

    assert decision.tier_answered == "cloud"
    assert decision.difficulty_score == pytest.approx(0.8)
    assert decision.est_cost_usd > 0.0
    # The local answer was paid for and thrown away -- that has to show up.
    discarded = next(n for n in decision.notes if n.startswith("discarded the local answer"))
    assert "reported latency includes it" in discarded


def test_unparseable_confidence_falls_back_to_the_heuristic(monkeypatch):
    """A model that ignores the output format must not silently become
    'maximally confident' or 'maximally unsure'."""
    with _PhoneServer("An answer with no confidence line.") as server:
        decision = _mobile_router(server, monkeypatch).route("What time zone is Tokyo in?")

    assert any("fell back to the surface-feature heuristic" in n for n in decision.notes)
    assert decision.tier_answered == "local"


def test_latency_budget_skips_the_fast_brain_entirely(monkeypatch):
    """If the profiled estimate already blows the budget, don't spend an
    inference we would only discard."""
    with _PhoneServer("unused\nCONFIDENCE: 99") as server:
        decision = _mobile_router(
            server, monkeypatch, local_latency_budget_ms=1
        ).route("What time zone is Tokyo in?")

    assert decision.tier_answered == "cloud"
    assert server.received == [], "the fast brain should never have been called"
    assert any(n.startswith("skipped the fast brain") for n in decision.notes)


def test_unreachable_fast_brain_escalates_rather_than_failing(monkeypatch):
    with _PhoneServer("unused", status=500) as server:
        decision = _mobile_router(server, monkeypatch).route("What time zone is Tokyo in?")

    assert decision.tier_answered == "cloud"
    assert any("fast brain reported a problem" in n for n in decision.notes)


def test_pii_never_reaches_the_phone_or_the_cloud_unmasked(monkeypatch):
    """The mobile fast brain is a *separate device* over HTTP, so it sits on
    the far side of a boundary the cloud brain is usually the only occupant
    of. Masking happens before the routing decision, so it is masked for both.
    """
    query = (
        "My email is jane.doe@example.com and my SSN is 123-45-6789 -- "
        "prove the Riemann hypothesis."
    )
    with _PhoneServer("Not sure.\nCONFIDENCE: 15") as server:
        decision = _mobile_router(server, monkeypatch).route(query)

        sent_to_phone = server.prompts()[0]

    assert decision.tier_answered == "cloud"
    assert decision.pii_entities_masked == 2

    # 1. what physically went to the phone
    assert "jane.doe@example.com" not in sent_to_phone
    assert "123-45-6789" not in sent_to_phone
    assert "[PII_EMAIL_1]" in sent_to_phone

    # 2. what went to the cloud
    sent_off_device = next(n for n in decision.notes if n.startswith("sent off-device"))
    assert "jane.doe@example.com" not in sent_off_device

    # 3. the user still gets their real values back
    assert "jane.doe@example.com" in decision.answer
    assert "123-45-6789" in decision.answer
