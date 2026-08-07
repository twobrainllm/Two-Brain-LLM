"""Tests for the /route API server (`src/two_brain_router/api.py`).

Spins up a real `ThreadingHTTPServer` on an OS-assigned loopback port and
hits it with real HTTP requests, for the same reason `test_orchestrator.py`
runs a real server for the phone brain rather than mocking `urllib`: the
thing worth asserting on is what actually crosses the socket.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from two_brain_router.api import DEFAULT_PORT, _Handler


@pytest.fixture(scope="module")
def api_server():
    from two_brain_router.routing import TwoBrainRouter

    router = TwoBrainRouter(tier="pc")
    handler = type("_TestHandler", (_Handler,), {"router": router, "route_lock": threading.Lock()})
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(base_url: str, path: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_reports_the_configured_tier(api_server):
    with urllib.request.urlopen(f"{api_server}/health", timeout=5) as resp:
        body = json.loads(resp.read())
    assert resp.status == 200
    assert body == {"status": "ok", "tier": "pc"}


def test_route_matches_the_router_shape_and_adds_display_extras(api_server):
    status, body = _post(api_server, "/route", {"query": "What time zone is Tokyo in?"})
    assert status == 200
    assert body["tier_answered"] == "local"
    assert body["answer"]
    assert body["pii_entities_masked"] == 0
    assert body["pii_entities"] == []
    assert body["escalate_threshold"] == 0.55
    assert body["tier"] == "pc"


def test_route_masks_before_it_ever_leaves_the_router_and_reports_entity_types(api_server):
    status, body = _post(
        api_server,
        "/route",
        {
            "query": (
                "My email is jane.doe@example.com -- derive the time complexity of "
                "merge sort step by step and compare it to quicksort's worst case."
            )
        },
    )
    assert status == 200
    assert body["tier_answered"] == "cloud"
    assert body["pii_entities_masked"] == 1
    assert body["pii_entities"] == [{"type": "EMAIL", "count": 1}]
    sent_off_device = next(n for n in body["notes"] if n.startswith("sent off-device"))
    assert "jane.doe@example.com" not in sent_off_device
    # the answer returned to the (on-device) UI is rehydrated
    assert "jane.doe@example.com" in body["answer"]


def test_route_rejects_a_missing_query_without_raising(api_server):
    status, body = _post(api_server, "/route", {})
    assert status == 400
    assert "error" in body


def test_unknown_path_is_a_clean_404(api_server):
    status, body = _post(api_server, "/nope", {"query": "hi"})
    assert status == 404


def test_serve_binds_loopback_by_default(monkeypatch):
    """Regression guard for the on-device-only default -- see api.py's module
    docstring. A request body carries pre-mask text, same posture as
    PhoneFastBrain's on-device host check. Exercises main()'s own argparse
    default rather than re-declaring it here, so this can't silently drift."""
    from two_brain_router import api as api_module

    calls = []
    monkeypatch.setattr(
        api_module, "serve", lambda tier, host, port, **kw: calls.append((tier, host, port, kw))
    )
    api_module.main([])
    # Trace defaults on for the server: a human watching this terminal is who
    # it is for. `--no-trace` is the opt-out, checked below.
    assert calls == [("pc", "127.0.0.1", DEFAULT_PORT, {"trace": True})]

    calls.clear()
    api_module.main(["--no-trace"])
    assert calls == [("pc", "127.0.0.1", DEFAULT_PORT, {"trace": False})]
