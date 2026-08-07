"""Backend for the chat UI: serves the static files and bridges to TwoBrainRouter.

Until now `ui/` was a pure front-end with `mockRespond()` returning canned
strings. This is the piece that makes the chat real -- the same
`TwoBrainRouter` the CLI uses, with the same privacy ordering, reached over a
tiny JSON API.

    UI_TEST=1 TWO_BRAIN_GPU_BRAIN=1 python ui/server.py

Stdlib only, matching the rest of the base package: `http.server` and `json`,
no framework, no build step.

## Why UI_TEST matters here

The UI's local/cloud switch pins which brain answers, which `route()` only
honours under `UI_TEST=1` (see `router.py`). Without it the server still works,
but the switch becomes a *display* preference while the policy actually decides
-- so `/api/health` reports the flag and the UI labels the toggle accordingly
rather than lying about what it controls.

## Endpoints

    GET  /api/health  -> what is actually wired: which brains, vision, UI_TEST
    POST /api/chat    -> {message, tier?, image?} -> the full RouteDecision
    GET  /*           -> static files from this directory
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_DIR = Path(__file__).parent
REPO = UI_DIR.parent
sys.path.insert(0, str(REPO / "src"))

from two_brain_router.routing.brains import GpuLocalBrain  # noqa: E402
from two_brain_router.routing.router import TwoBrainRouter, ui_test_enabled  # noqa: E402
from two_brain_router.signals.loader import TierSignals  # noqa: E402

#: Only these are decoded from a data URI. An image is written to a temp file
#: for llama.cpp, so the extension must not be attacker-controlled.
_ALLOWED_IMAGE = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
_MAX_IMAGE_BYTES = 12 * 1024 * 1024

_router: TwoBrainRouter | None = None
_router_error: str | None = None


def get_router() -> TwoBrainRouter:
    """Built once, lazily -- constructing a brain starts a llama-server child."""
    global _router, _router_error
    if _router is None and _router_error is None:
        try:
            router = TwoBrainRouter(tier="pc")
            # Upgrade the fast brain to a vision-capable one when the GPU brain
            # is on and the VLM weights are present, so the UI can accept
            # images. Falls back silently to text-only otherwise -- an image
            # then gets a clear error rather than being dropped.
            if os.environ.get("TWO_BRAIN_GPU_BRAIN") == "1":
                try:
                    router.fast_brain.close()
                    router.fast_brain = GpuLocalBrain.for_vision(
                        "pc", TierSignals.load("pc_3b", "ai_pc")
                    )
                except Exception as exc:  # noqa: BLE001 -- vision is optional
                    print(f"[server] vision brain unavailable, staying text-only: {exc}")
            _router = router
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the server
            _router_error = f"{type(exc).__name__}: {exc}"
            raise
    if _router is None:
        raise RuntimeError(_router_error or "router unavailable")
    return _router


def decode_image(data_uri: str) -> Path:
    """Data URI -> a temp file path llama.cpp can read."""
    match = re.match(r"^data:([\w/\-.+]+);base64,(.*)$", data_uri, re.DOTALL)
    if not match:
        raise ValueError("image must be a base64 data URI")
    mime, payload = match.group(1), match.group(2)
    if mime not in _ALLOWED_IMAGE:
        raise ValueError(f"unsupported image type {mime!r}; allowed: {', '.join(_ALLOWED_IMAGE)}")
    raw = base64.b64decode(payload, validate=True)
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image too large ({len(raw)} bytes, limit {_MAX_IMAGE_BYTES})")
    fd, name = tempfile.mkstemp(suffix=_ALLOWED_IMAGE[mime], prefix="twobrain-ui-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(raw)
    return Path(name)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(UI_DIR), **kw)

    def log_message(self, fmt, *args):  # quieter than the default
        if "/api/" in (self.path or ""):
            sys.stderr.write(f"[server] {fmt % args}\n")

    def _json(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/api/health":
            return self._health()
        super().do_GET()

    def _health(self) -> None:
        try:
            router = get_router()
        except Exception as exc:  # noqa: BLE001
            return self._json(503, {"ok": False, "error": str(exc)})
        self._json(200, {
            "ok": True,
            # The UI uses this to label the toggle honestly: with UI_TEST off,
            # the switch cannot actually choose the tier.
            "ui_test": ui_test_enabled(),
            "fast_brain": type(router.fast_brain).__name__,
            "deep_brain": type(router.deep_brain).__name__,
            "vision": bool(getattr(router.fast_brain, "can_see", False)),
        })

    def _sse(self, event: str, data: dict) -> None:
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _chat_stream(self, message: str, tier, image_path: Path | None) -> None:
        """Server-Sent Events, so the browser paints tokens as they generate.

        A local reply runs at ~21 tok/s, so a long answer is a minute-plus of
        blank screen without this. Errors are delivered as an `error` event
        rather than an HTTP status, since the 200 and headers are already gone
        by the time generation starts.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # This app is same-origin, but proxies buffer SSE by default and that
        # would defeat the whole point.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            router = get_router()
            for kind, payload in router.route_stream(message, image=image_path, force_tier=tier):
                self._sse(kind, payload if isinstance(payload, dict) else {"text": payload})
        except Exception as exc:  # noqa: BLE001 -- the stream is the only channel left
            try:
                self._sse("error", {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:  # noqa: BLE001 -- client already gone
                pass
        finally:
            if image_path is not None:
                image_path.unlink(missing_ok=True)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] not in ("/api/chat", "/api/chat/stream"):
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": f"bad request body: {exc}"})

        message = (payload.get("message") or "").strip()
        if not message:
            return self._json(400, {"error": "message is required"})
        tier = payload.get("tier")
        if tier not in (None, "local", "cloud"):
            return self._json(400, {"error": "tier must be 'local' or 'cloud'"})

        image_path: Path | None = None
        if payload.get("image"):
            try:
                image_path = decode_image(payload["image"])
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})

        if self.path.split("?")[0] == "/api/chat/stream":
            # _chat_stream owns the temp file from here, including deleting it.
            return self._chat_stream(message, tier, image_path)

        try:
            router = get_router()
            decision = router.route(message, image=image_path, force_tier=tier)
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- surface it to the UI
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            # The image was only ever a temp file for llama.cpp; it does not
            # outlive the request, and it never left this machine.
            if image_path is not None:
                image_path.unlink(missing_ok=True)

        self._json(200, {
            "answer": decision.answer,
            "tier_answered": decision.tier_answered,
            "difficulty_score": decision.difficulty_score,
            "est_latency_ms": decision.est_latency_ms,
            "est_cost_usd": decision.est_cost_usd,
            "pii_entities_masked": decision.pii_entities_masked,
            "notes": decision.notes,
        })


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    # Loopback by default. HOST=0.0.0.0 makes it reachable from the LAN, which
    # is sometimes the only way in (a remote editor whose port forwarding is
    # broken, say) -- but it also exposes /api/chat to everyone on that
    # network, and every escalated request spends real cloud tokens. Opt in
    # deliberately; do not make it the default.
    host = os.environ.get("HOST", "127.0.0.1")
    if not ui_test_enabled():
        print("[server] UI_TEST is not 1 -- the local/cloud switch will NOT force a tier;")
        print("[server] the policy will decide. Start with UI_TEST=1 to make it authoritative.")
    if host != "127.0.0.1":
        print(f"[server] WARNING: bound to {host} -- anyone who can reach this host on")
        print("[server] port {0} can use /api/chat, which spends real cloud tokens.".format(port))
    print(f"[server] http://{host}:{port}  (serving {UI_DIR})")
    print("[server] brains build lazily on the first request; the first reply may be slow.")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
