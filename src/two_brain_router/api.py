"""A thin HTTP wrapper around `TwoBrainRouter`, for `ui/`.

    python -m two_brain_router.api                    # pc tier, stub brains
    python -m two_brain_router.api --tier mobile
    $env:TWO_BRAIN_NPU_BRAIN=1;   python -m two_brain_router.api --tier pc
    $env:TWO_BRAIN_PHONE_BRAIN=1; python -m two_brain_router.api --tier mobile

One route:

    POST /route   {"query": "...", "context": ""}
              ->  RouteDecision, as JSON, plus a couple of display-only extras
                  (see `_build_response`)

This is deliberately not a general API: one `TwoBrainRouter` instance is built
at startup for a single fixed tier (matching how `python -m two_brain_router`
already works) and reused across requests -- both real brains keep an
expensive resource alive across calls (`NpuFastBrain`'s Genie dialog session;
`PhoneFastBrain`'s HTTP connection), so building a fresh router per request
would silently reintroduce the cold-load cost `NpuFastBrain` was written to
avoid. `ui/README.md`'s own wiring plan asked for exactly this shape.

stdlib-only (`http.server`), matching this package's zero-runtime-dependency
policy and `src/phone_brain/mock_phone_brain_server.py`'s own precedent.

Binds to `127.0.0.1` by default, not `0.0.0.0` -- same reasoning as
`PhoneFastBrain._assert_on_device`: nothing about serving a browser tab on
this machine requires the network, and a query in the request body is
*pre-mask* input, so this endpoint is on the sensitive side of the boundary
this project is built around. `--host` exists for a deliberate LAN demo, not
as a default.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from two_brain_router.privacy import PIIGuard
from two_brain_router.routing import RouteDecision, Tier, TwoBrainRouter

DEFAULT_PORT = 8765


def _build_response(router: TwoBrainRouter, decision: RouteDecision, query: str) -> dict:
    """`RouteDecision` plus display-only extras `ui/profiler.js` wants.

    Everything from `dataclasses.asdict(decision)` is the real routing
    record -- the same fields `cli.py --json` already emits, so this is not a
    second shape to keep in sync. The three additions below are computed
    independently, read-only, and change nothing about the routing decision
    itself:

    - `escalate_threshold` / `tier`: config the UI needs to label the
      response, not part of any one decision.
    - `pii_entities`: a per-type breakdown (`[{"type": "EMAIL", "count": 1},
      ...]`), which `RouteDecision.pii_entities_masked` (a bare count)
      doesn't carry. Computed with a *second*, throwaway `PIIGuard` against
      the original query purely for display -- this never touches
      `masked_query`, the vault, or anything that crosses `route()`'s
      boundary, so it can't affect the invariant `route()` already enforced
      before this function ever sees the query.
    """
    body = dataclasses.asdict(decision)
    body["escalate_threshold"] = router.policy.escalate_threshold
    body["tier"] = router.tier

    entities = PIIGuard().detect(query)
    counts: dict[str, int] = {}
    for entity_type, _value in entities:
        counts[entity_type] = counts.get(entity_type, 0) + 1
    body["pii_entities"] = [
        {"type": entity_type, "count": count} for entity_type, count in counts.items()
    ]
    return body


class _Handler(BaseHTTPRequestHandler):
    #: Set by `serve`. One `TwoBrainRouter`, shared and reused across
    #: requests/threads -- see the module docstring for why.
    router: TwoBrainRouter
    #: Real brains (`NpuFastBrain`'s Genie dialog, `PhoneFastBrain`'s HTTP
    #: call) are not asserted thread-safe for concurrent calls on one
    #: instance -- `tests/test_npu_brain.py` documents a real crash from two
    #: *separate* concurrent Genie sessions, and a shared session is not
    #: obviously safer. `ThreadingHTTPServer` keeps the server responsive
    #: (a slow request doesn't block new connections), but every actual
    #: `route()` call is serialized through this lock.
    route_lock: threading.Lock

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # ui/ is served from a different origin (python -m http.server on its
        # own port per ui/README.md) -- both are loopback-only regardless, so
        # a permissive CORS header here doesn't widen the real boundary.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's required name
        # fetch() with a JSON body triggers a CORS preflight; answer it.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(200, {"status": "ok", "tier": self.router.tier})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/route":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            raw = self.rfile.read(length) or b"{}"
            body = json.loads(raw)
            query = body["query"]
        except (json.JSONDecodeError, KeyError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return
        context = body.get("context", "")

        try:
            with self.route_lock:
                decision = self.router.route(query, context)
        except AssertionError as exc:
            # The masked-token invariant is fatal by design (CLAUDE.md) -- it
            # must not become a 200 with a quietly-wrong answer.
            self._send_json(500, {"error": f"privacy invariant violated: {exc}"})
            return

        self._send_json(200, _build_response(self.router, decision, query))

    def log_message(self, fmt, *args) -> None:  # keep the default noisy access log
        print(f"[two-brain-api] {self.address_string()} - {fmt % args}")


def serve(tier: Tier, host: str, port: int) -> None:
    router = TwoBrainRouter(tier=tier)
    handler = type("_BoundHandler", (_Handler,), {"router": router, "route_lock": threading.Lock()})
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"two-brain-router API -> http://{host}:{port} (tier={tier})")
    print(f"  POST http://{host}:{port}/route   {{\"query\": \"...\", \"context\": \"\"}}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        close = getattr(router.fast_brain, "close", None)
        if callable(close):
            close()
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="two_brain_router.api")
    parser.add_argument("--tier", choices=["pc", "mobile"], default="pc")
    parser.add_argument("--host", default="127.0.0.1", help="default is loopback-only; see module docstring")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    serve(args.tier, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
