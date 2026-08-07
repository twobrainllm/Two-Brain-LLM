#!/usr/bin/env python3
"""Serves the chat UI and exposes the router over HTTP.

This is README.md step 1 -- the thin API layer that lets the browser talk to
`TwoBrainRouter`. It is a new module rather than an edit to an existing one,
so nothing in `routing/` had to change to support a UI.

    python ui/server.py                      # http://127.0.0.1:8600
    python ui/server.py --host 0.0.0.0       # reachable from a phone on the LAN
    TWO_BRAIN_PHONE_BRAIN=1 python ui/server.py    # try the phone fast brain

Stdlib only, matching the rest of the package -- no FastAPI, no Flask.

Endpoints
---------
GET  /              the chat UI (static files from this directory)
GET  /api/health    which brains are live, so the UI can label itself honestly
POST /api/route     {"query": str, "context": str} -> the RouteDecision

**The vault never crosses this boundary.** `/api/route` returns the
*rehydrated* answer and a count of masked entities, never the mapping itself.
The masking, the invariant assertion, and the rehydration all happen inside
`TwoBrainRouter.route`, on this side of the wire.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UI_DIR.parent / "src"))

from two_brain_router.routing.policy import RoutePolicy  # noqa: E402
from two_brain_router.routing.router import TwoBrainRouter  # noqa: E402

MAX_BODY_BYTES = 256 * 1024


@lru_cache(maxsize=4)
def get_router(tier: str) -> TwoBrainRouter:
    """One router per tier, built lazily.

    Cached because constructing a router loads the tier's captured signals and,
    for the real backends, can attach to a runtime -- neither of which should
    happen per keystroke.
    """
    return TwoBrainRouter(tier=tier)


def brain_names(tier: str) -> dict:
    router = get_router(tier)
    return {
        "tier": tier,
        "fast_brain": type(router.fast_brain).__name__,
        "deep_brain": type(router.deep_brain).__name__,
        # A served fast brain has an endpoint; an in-process one does not.
        "fast_brain_endpoint": getattr(router.fast_brain, "base_url", None),
        "escalate_threshold": router.policy.escalate_threshold,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    # -- helpers ----------------------------------------------------------

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The UI is served from this same origin, so no CORS header is needed
        # and none is added -- the router should not be callable from an
        # arbitrary page the user happens to have open.
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/api/health":
            tier = os.environ.get("TWO_BRAIN_UI_TIER", "pc")
            try:
                self._send_json({"ok": True, **brain_names(tier)})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/api/route":
            self.send_error(404, "no such endpoint")
            return

        body = self._read_json()
        if body is None:
            self._send_json({"error": "invalid or oversized JSON body"}, status=400)
            return

        query = (body.get("query") or "").strip()
        if not query:
            self._send_json({"error": "query is required"}, status=400)
            return

        tier = body.get("tier") or os.environ.get("TWO_BRAIN_UI_TIER", "pc")
        if tier not in ("mobile", "pc"):
            self._send_json({"error": f"unknown tier {tier!r}"}, status=400)
            return

        started = time.perf_counter()
        try:
            router = get_router(tier)
            decision = router.route(query, body.get("context") or "")
        except Exception as exc:  # noqa: BLE001
            # Surface the type and message, never a traceback: a traceback from
            # a router that just handled a masked query can echo it back.
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=502,
            )
            return

        self._send_json(
            {
                "tier": decision.tier_answered,
                "answer": decision.answer,
                "difficulty": decision.difficulty_score,
                "escalate_threshold": router.policy.escalate_threshold,
                "est_latency_ms": decision.est_latency_ms,
                "est_cost_usd": decision.est_cost_usd,
                "pii_entities_masked": decision.pii_entities_masked,
                "notes": decision.notes,
                "actual_latency_ms": (time.perf_counter() - started) * 1000,
                "fast_brain": type(router.fast_brain).__name__,
                "deep_brain": type(router.deep_brain).__name__,
                "router_tier": tier,
            }
        )

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="0.0.0.0 to reach it from a phone on the same network",
    )
    parser.add_argument(
        "--tier",
        default=os.environ.get("TWO_BRAIN_UI_TIER", "pc"),
        choices=["mobile", "pc"],
        help="which local tier the router uses when the request does not say",
    )
    args = parser.parse_args()
    os.environ["TWO_BRAIN_UI_TIER"] = args.tier

    info = brain_names(args.tier)
    print(f"Two-Brain UI  http://{args.host}:{args.port}")
    print(f"  tier       {info['tier']}")
    print(f"  fast brain {info['fast_brain']}" + (f"  ({info['fast_brain_endpoint']})" if info["fast_brain_endpoint"] else ""))
    print(f"  deep brain {info['deep_brain']}")
    if args.host == "0.0.0.0":
        print("\n  Bound to 0.0.0.0: anyone on this network can reach the router.")
        print("  Fine on a workshop LAN for a demo; do not leave it running.")
    print()

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
