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
    # `model` is None until a request selects one -- the fast brain came from
    # env vars at startup, not from the catalogue.
    assert body == {"status": "ok", "tier": "pc", "model": None}


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


# --------------------------------------------------------------------------
# /route/stream -- the local half arrives before the cloud half
# --------------------------------------------------------------------------


def _post_ndjson(base_url: str, path: str, payload: dict) -> list[dict]:
    """POST and collect every NDJSON line, in arrival order."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return [json.loads(line) for line in resp if line.strip()]


def test_stream_emits_only_a_result_when_nothing_escalates(api_server):
    """A locally-answered query has no partial worth streaming.

    The stub `LocalFastBrain` doesn't report gaps, so this takes the heuristic
    path and never reaches a deep-brain call -- no progress event, just the
    result. Pinned so the streaming endpoint stays a strict superset of
    `/route`: same outcome, one extra line at most.
    """
    messages = _post_ndjson(api_server, "/route/stream", {"query": "What time zone is Tokyo in?"})

    assert [m["type"] for m in messages] == ["result"]
    assert messages[-1]["tier_answered"] == "local"
    assert messages[-1]["answer"]


def test_stream_announces_the_escalation_before_the_cloud_answer(api_server):
    """An escalating query says so *before* the deep brain is called.

    This is what lets the UI explain a long wait instead of showing an
    undifferentiated spinner. `local_answer` is None rather than "" because
    "there is no partial" and "the partial was blank" are different states.
    """
    messages = _post_ndjson(
        api_server,
        "/route/stream",
        {
            "query": (
                "Derive the time complexity of merge sort step by step and compare "
                "it to quicksort's worst case, then explain the trade-offs."
            )
        },
    )

    assert [m["type"] for m in messages] == ["progress", "result"]
    progress, result = messages
    assert progress["phase"] == "escalating"
    assert progress["local_answer"] is None
    assert progress["gap"] is None
    assert result["tier_answered"] == "cloud"


def test_stream_result_is_byte_identical_to_the_plain_route_response(api_server):
    """`/route/stream`'s final line is exactly what `/route` would have sent.

    The whole value of the streaming endpoint is that it changes *when* you
    learn things, never *what* -- if these two ever diverge, the UI is showing
    something the audit trail doesn't.
    """
    query = "What time zone is Tokyo in?"
    _status, plain = _post(api_server, "/route", {"query": query})
    streamed = _post_ndjson(api_server, "/route/stream", {"query": query})[-1]

    assert streamed.pop("type") == "result"
    # Latency is measured per call and legitimately differs between two runs.
    for body in (plain, streamed):
        body.pop("est_latency_ms", None)
    assert streamed == plain


def test_stream_hands_out_the_local_half_before_the_cloud_call_finishes():
    """The point of the endpoint, asserted on timing rather than on shape.

    Uses a deliberately slow fake cloud brain so "arrived early" is measurable
    rather than inferred: the progress line must land while the deep brain is
    still blocked, not merely before the final line. A version of this feature
    that emitted both lines at the end would pass every other test in this file.
    """
    import time
    from http.server import ThreadingHTTPServer

    from two_brain_router.routing import BrainResponse, TwoBrainRouter

    cloud_delay_s = 1.5

    class _SlowCloud:
        reports_confidence = False
        reports_gaps = False
        trusted_with_raw_pii = False

        def answer(self, query, context=""):
            time.sleep(cloud_delay_s)
            return BrainResponse(text="[cloud gap fill]", latency_ms=cloud_delay_s * 1000)

    class _FastLocalWithGap:
        reports_confidence = True
        reports_gaps = True
        trusted_with_raw_pii = True

        def answer(self, query, context=""):
            return BrainResponse(
                text="The local half.", latency_ms=10.0, confidence=0.9, unknown="the missing half"
            )

    router = TwoBrainRouter(tier="pc", trace=False)
    router.fast_brain = _FastLocalWithGap()
    router.deep_brain = _SlowCloud()
    handler = type(
        "_SlowHandler", (_Handler,), {"router": router, "route_lock": threading.Lock()}
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"
        data = json.dumps({"query": "Do both halves."}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/route/stream",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        arrivals: list[tuple[str, float]] = []
        with urllib.request.urlopen(req, timeout=30) as resp:
            for line in resp:
                if not line.strip():
                    continue
                msg = json.loads(line)
                arrivals.append((msg["type"], time.perf_counter() - started))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert [kind for kind, _ in arrivals] == ["progress", "result"]
    (_, progress_at), (_, result_at) = arrivals
    # The progress line beat the cloud call, with room to spare on a slow CI box.
    assert progress_at < cloud_delay_s * 0.5, (
        f"progress arrived at {progress_at:.2f}s -- it was buffered until the "
        f"cloud call finished rather than flushed when it was ready"
    )
    assert result_at >= cloud_delay_s


# --------------------------------------------------------------------------
# /route/sse -- token-level streaming, and image upload
# --------------------------------------------------------------------------


def _post_sse(base_url: str, payload: dict) -> list[tuple[str, dict]]:
    """POST and collect `(event, data)` frames in arrival order."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/route/sse", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    frames: list[tuple[str, dict]] = []
    event = "message"
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                frames.append((event, json.loads(line[5:].strip())))
    return frames


def test_sse_falls_back_to_one_delta_when_the_brain_cannot_stream(api_server):
    """The stub brains don't stream, so the shape must degrade, not break.

    A caller renders `delta` frames the same way regardless; only the number of
    them changes. That keeps the UI free of a "does this backend stream?" branch.
    """
    frames = _post_sse(api_server, {"query": "What time zone is Tokyo in?"})
    kinds = [k for k, _ in frames]

    assert kinds[0] == "meta"
    assert "delta" in kinds
    assert kinds[-1] == "done"
    assert dict(frames)["meta"]["streaming"] is False
    answer = "".join(d["text"] for k, d in frames if k == "delta")
    assert answer == dict(frames)["done"]["answer"]


def test_sse_done_frame_carries_the_same_extras_as_plain_route(api_server):
    """One response shape across endpoints -- the UI parses `done` exactly as
    it parses `/route`'s body."""
    query = "What time zone is Tokyo in?"
    _status, plain = _post(api_server, "/route", {"query": query})
    done = dict(_post_sse(api_server, {"query": query}))["done"]

    for key in ("escalate_threshold", "tier", "pii_entities", "pii_entities_detected"):
        assert key in done, f"{key} missing from the done frame"
        assert done[key] == plain[key]


def test_sse_reports_a_bad_image_as_a_400_not_a_broken_stream(api_server):
    status, body = _post(api_server, "/route", {"query": "hi", "image": "not-a-data-url"})
    assert status == 400
    assert "data:" in body["error"]


def test_an_oversized_image_is_refused_before_it_is_decoded(api_server):
    """The cap exists so an unbounded body cannot exhaust memory on a loopback
    server that deliberately has no other auth."""
    from two_brain_router.api import MAX_IMAGE_CHARS

    huge = "data:image/png;base64," + ("A" * (MAX_IMAGE_CHARS + 10))
    status, body = _post(api_server, "/route", {"query": "hi", "image": huge})
    assert status == 400
    assert "too large" in body["error"]


def test_an_image_without_a_vision_brain_is_a_clean_400(api_server):
    """The stub fast brain cannot see. That has to be a readable error rather
    than a 500 -- and the decoded temp file must still be cleaned up."""
    import base64 as _b64
    import glob
    import tempfile as _tmp

    png = _b64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()
    before = set(glob.glob(f"{_tmp.gettempdir()}/two-brain-*"))
    status, body = _post(
        api_server, "/route", {"query": "what is this?", "image": f"data:image/png;base64,{png}"}
    )
    after = set(glob.glob(f"{_tmp.gettempdir()}/two-brain-*"))

    assert status == 400
    assert "cannot see" in body["error"]
    assert after == before, f"temp image left behind: {after - before}"


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------


def test_models_lists_only_what_is_installed(api_server):
    """`/models` is discovery, not configuration.

    Offering a model whose weights are absent means a failure 17 seconds into a
    cold load instead of an option that simply isn't there.
    """
    req = urllib.request.Request(f"{api_server}/models", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())

    assert "models" in body and isinstance(body["models"], list)
    for m in body["models"]:
        assert m["available"] is True, f"{m['id']} listed but not installed"
        assert {"id", "label", "backend", "detail", "vision"} <= set(m)
        assert m["backend"] in ("npu", "gpu")


def test_an_unknown_model_is_a_clean_400_and_does_not_swap_the_brain(api_server):
    """A bad id must not leave the router brainless.

    `use_model` clears `fast_brain` before building the replacement, so a
    failure there could strand the server -- an unknown id is rejected before
    any of that happens, and the next request still works.
    """
    status, body = _post(api_server, "/route", {"query": "hi", "model": "no-such-model"})
    assert status == 400
    assert "no-such-model" in body["error"]

    # The server is still answering with whatever it had before.
    status, body = _post(api_server, "/route", {"query": "What time zone is Tokyo in?"})
    assert status == 200
    assert body["answer"]


def test_an_absent_or_null_model_never_triggers_a_switch(api_server):
    """The UI sends `model` on every request so the server cannot drift out of
    sync with the dropdown. That is only viable if an unchanged -- or absent --
    id is free, because a switch costs a 12-17s cold load."""
    from two_brain_router.routing import TwoBrainRouter

    swaps = []
    original = TwoBrainRouter.use_model
    try:
        TwoBrainRouter.use_model = lambda self, mid: swaps.append(mid)  # type: ignore[method-assign]
        for payload in ({"query": "hi"}, {"query": "hi", "model": None}):
            status, _ = _post(api_server, "/route", payload)
            assert status == 200
    finally:
        TwoBrainRouter.use_model = original  # type: ignore[method-assign]

    assert swaps == [], f"a switch was attempted for {swaps}"
