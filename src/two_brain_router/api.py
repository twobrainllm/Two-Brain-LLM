"""A thin HTTP wrapper around `TwoBrainRouter`, for `ui/`.

    python -m two_brain_router.api                    # pc tier, stub brains
    python -m two_brain_router.api --tier mobile
    $env:TWO_BRAIN_NPU_BRAIN=1;   python -m two_brain_router.api --tier pc
    $env:TWO_BRAIN_PHONE_BRAIN=1; python -m two_brain_router.api --tier mobile
    # both flags, mobile tier: phone answers when confident; when it isn't,
    # the same AI PC model above answers as a second opinion instead of
    # escalating to the cloud -- see docs/ORCHESTRATOR.md
    $env:TWO_BRAIN_PHONE_BRAIN=1; $env:TWO_BRAIN_NPU_BRAIN=1; python -m two_brain_router.api --tier mobile

Three routes:

    POST /route          {"query": "...", "context": ""}
                     ->  RouteDecision, as JSON, plus a couple of display-only
                         extras (see `_build_response`)

    POST /route/stream   same body
                     ->  NDJSON: an optional {"type":"progress"} line carrying
                         the local half as soon as it exists, then
                         {"type":"result"} with the identical body /route would
                         have returned. Measured on real hardware: the local
                         half is readable ~11s before the merged answer.
                         See `_do_route_stream` and docs/ORCHESTRATOR.md.

    POST /route/sse      same body, plus an optional `image` data: URL
                     ->  Server-Sent Events: `meta`, then `delta` frames as each
                         brain generates, tier-attributed, then `done`. Only the
                         local model's `solution` field is streamed -- never the
                         JSON around it. First visible text at 0.83s vs 4.15s for
                         the whole answer, measured on the real NPU.

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
import base64
import binascii
import dataclasses
import json
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from two_brain_router.privacy import PIIGuard
from two_brain_router.routing import RouteDecision, Tier, TwoBrainRouter

DEFAULT_PORT = 8765

#: Cap on an attached image, in base64 characters (~3.4 MB of image bytes).
#: An unbounded body is a trivial memory-exhaustion lever on a loopback
#: server that deliberately has no other auth, and no phone photo this demo
#: cares about comes near it.
MAX_IMAGE_CHARS = 4_500_000


def _build_response(
    router: TwoBrainRouter, decision: RouteDecision, query: str, context: str = ""
) -> dict:
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
    return {**dataclasses.asdict(decision), **_display_extras(router, query, context)}


def _display_extras(router: TwoBrainRouter, query: str, context: str = "") -> dict:
    """The read-only additions `ui/profiler.js` wants, computed independently.

    Shared by `/route` and `/route/sse` so both return one shape. Nothing here
    touches the vault or anything that crossed `route()`'s boundary -- it is a
    second, throwaway `PIIGuard` over the caller's own inputs, purely for
    display, and cannot affect the invariant `route()` already enforced.
    """
    # Query *and* context: conversation history is real user text, carries PII
    # the current query may not, and crosses the boundary on an escalation
    # exactly like the query does. De-duplicated by value so an address written
    # in both places counts once -- matching how the vault (and therefore
    # `pii_entities_masked`) counts it, rather than reading as an extra entity
    # that never got masked.
    guard = PIIGuard()
    entities = guard.detect(query) + (guard.detect(context) if context else [])
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for entity_type, value in entities:
        if value in seen:
            continue
        seen.add(value)
        counts[entity_type] = counts.get(entity_type, 0) + 1
    return {
        "escalate_threshold": router.policy.escalate_threshold,
        "tier": router.tier,
        "pii_entities": [
            {"type": entity_type, "count": count} for entity_type, count in counts.items()
        ],
    }


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

    def _select_model(self, body: dict) -> str | None:
        """Honour a `model` field, switching the fast brain if it changed.

        Returns an error string, or None on success. Called inside the route
        lock by every endpoint, so a switch can never race a request that is
        mid-inference on the brain being closed.

        A switch is **not cheap** -- the old brain is closed and the new one
        cold-loaded (~12s NPU, ~17s for the 8B GPU model) -- which is why it is
        driven by an explicit field rather than inferred from anything.
        """
        model_id = body.get("model")
        if not model_id or model_id == self.router.model_id:
            return None
        try:
            self.router.use_model(model_id)
        except Exception as exc:  # noqa: BLE001 -- reported to the caller verbatim
            return f"could not load model {model_id!r}: {type(exc).__name__}: {exc}"
        return None

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "tier": self.router.tier,
                "model": self.router.model_id,
            })
            return
        if self.path == "/models":
            # Only what is actually installed -- see routing/models.py. A
            # dropdown offering something that fails to load 17s later is worse
            # than a shorter dropdown.
            from two_brain_router.routing.models import available_models

            self._send_json(200, {
                "models": [m.as_json() for m in available_models()],
                "active": self.router.model_id,
            })
            return
        self._send_json(404, {"error": "not found"})

    def _read_query(self) -> tuple[str, str] | None:
        """Parse `{"query": ..., "context": ...}`, or answer 400 and return None."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            raw = self.rfile.read(length) or b"{}"
            body = json.loads(raw)
            self._body = body
            return body["query"], body.get("context", "")
        except (json.JSONDecodeError, KeyError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return None

    def _read_image(self) -> Path | None:
        """Decode an attached `data:image/...;base64,...` into a temp file.

        The router takes a filesystem `Path` because that is what a local VLM
        wants; the browser has bytes. This is the only bridge between them.

        Written to a temp file rather than kept in memory because
        `GpuLocalBrain` re-reads it to build a data URI for `llama-server`, and
        because the caller deletes it in a `finally` -- an image is the most
        sensitive thing a user can hand this system, and leaving copies around
        after the answer is exactly the kind of quiet residue this project
        exists to avoid.

        Size-capped: an unbounded base64 body is a trivial way to exhaust
        memory on a loopback server that deliberately has no other auth.
        """
        raw = (getattr(self, "_body", None) or {}).get("image")
        if not raw:
            return None
        if len(raw) > MAX_IMAGE_CHARS:
            raise ValueError(
                f"image too large: {len(raw)} chars of base64 exceeds the "
                f"{MAX_IMAGE_CHARS} cap"
            )
        match = re.match(r"^data:image/(png|jpeg|jpg|webp);base64,(.+)$", raw, re.DOTALL)
        if not match:
            raise ValueError("image must be a data: URL of type png, jpeg or webp")
        suffix = "." + match.group(1).replace("jpg", "jpeg")
        try:
            blob = base64.b64decode(match.group(2), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image is not valid base64: {exc}") from exc
        handle, name = tempfile.mkstemp(prefix="two-brain-", suffix=suffix)
        with os.fdopen(handle, "wb") as fh:
            fh.write(blob)
        return Path(name)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/route":
            self._do_route()
        elif self.path == "/route/stream":
            self._do_route_stream()
        elif self.path == "/route/sse":
            self._do_route_sse()
        else:
            self._send_json(404, {"error": "not found"})

    def _do_route_sse(self) -> None:
        """Token-level streaming: text appears as each brain writes it.

        Server-Sent Events rather than `/route/stream`'s NDJSON, because this
        carries several *kinds* of message (`meta`, `delta`, `tier`, `done`,
        `error`) and SSE names them in the frame instead of making every
        consumer branch on a `type` field it had to parse first.

        `EventSource` still cannot be used on the client: it is GET-only, and
        this request carries a JSON body with an optional base64 image. The UI
        reads the `fetch` body as a stream and parses frames itself.

        **Only the local model's `solution` field is streamed** -- see
        `TwoBrainRouter.route_stream`. Deltas carry the tier that produced them
        so a split renders as two attributed bubbles.
        """
        parsed = self._read_query()
        if parsed is None:
            return
        query, context = parsed
        try:
            image = self._read_image()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")  # defeat proxy buffering
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def emit(event: str, payload: dict) -> None:
            frame = f"event: {event}\ndata: {json.dumps(payload)}\n\n"
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()  # without this the whole point is lost to buffering

        stream = None
        try:
            with self.route_lock:
                err = self._select_model(getattr(self, "_body", {}) or {})
                if err:
                    emit("error", {"error": err})
                    return
                stream = self.router.route_stream(query, context, image=image)
                for kind, payload in stream:
                    if kind == "done":
                        # `done` carries the RouteDecision as a dict already;
                        # add the same display-only extras `/route` returns so
                        # the UI has one shape to consume either way.
                        payload = {**payload, **_display_extras(self.router, query, context)}
                    emit(kind, payload)
        except AssertionError as exc:
            # The masked-token invariant is fatal by design. The status line is
            # already sent, so it has to travel in-band.
            emit("error", {"error": f"privacy invariant violated: {exc}"})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-answer; not an error
        except Exception as exc:  # noqa: BLE001 -- the stream is the only channel left
            emit("error", {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            # Closing the generator unwinds `answer_stream`'s `with response:`
            # so the upstream HTTP connection is released rather than left to a
            # GC that may never come.
            if stream is not None:
                stream.close()
            if image is not None:
                image.unlink(missing_ok=True)

    def _do_route(self) -> None:
        parsed = self._read_query()
        if parsed is None:
            return
        query, context = parsed

        try:
            image = self._read_image()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        # One `finally` around every exit from here, so the decoded image is
        # deleted whichever way this ends -- including the invariant-violation
        # path, which returns early. An image is the most sensitive thing a user
        # can hand this system; leaving a copy in the temp directory after an
        # error is exactly the quiet residue this project exists to avoid.
        try:
            try:
                with self.route_lock:
                    err = self._select_model(getattr(self, "_body", {}) or {})
                    if err:
                        self._send_json(400, {"error": err})
                        return
                    decision = self.router.route(query, context, image=image)
            except AssertionError as exc:
                # The masked-token invariant is fatal by design (CLAUDE.md) --
                # it must not become a 200 with a quietly-wrong answer.
                self._send_json(500, {"error": f"privacy invariant violated: {exc}"})
                return
            except ValueError as exc:
                # e.g. an image with no vision-capable fast brain configured.
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, _build_response(self.router, decision, query, context))
        finally:
            if image is not None:
                image.unlink(missing_ok=True)

    def _do_route_stream(self) -> None:
        """`/route`, but the local half is sent as soon as it exists.

        Newline-delimited JSON, one object per line, each with a `type`:

            {"type": "progress", "phase": "local_answer", "local_answer": ..., "gap": ...}
            {"type": "result",   ...the full RouteDecision, same shape as /route...}
            {"type": "error",    "error": "..."}

        NDJSON rather than Server-Sent Events: SSE's `EventSource` cannot issue
        a POST, so a browser would have to read the stream by hand anyway --
        at which point SSE's `data:` framing is pure overhead over one JSON
        object per line.

        No `Content-Length`, and the body is terminated by closing the
        connection -- which is exactly what `BaseHTTPRequestHandler`'s default
        HTTP/1.0 makes correct, so no manual chunked-transfer framing is
        needed. Every line is flushed as it is written; without the flush the
        whole point (the local half arriving early) is lost to buffering.

        Why this endpoint exists at all: the local half lands in ~4s and the
        cloud half has been measured at 15s+, so `/route` spends most of its
        wall-clock holding an answer that was ready the whole time.
        """
        parsed = self._read_query()
        if parsed is None:
            return
        query, context = parsed

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def emit(payload: dict) -> None:
            self.wfile.write(json.dumps(payload).encode("utf-8") + b"\n")
            self.wfile.flush()

        def on_progress(progress) -> None:
            emit({"type": "progress", **dataclasses.asdict(progress)})

        try:
            with self.route_lock:
                decision = self.router.route(query, context, on_progress=on_progress)
        except AssertionError as exc:
            # Same fatal invariant as `/route`. The status line is already sent
            # (200, unavoidably -- the failure happens mid-stream), so the error
            # has to travel in-band. A client that ignores `type` would
            # otherwise see a truncated stream and no reason for it.
            emit({"type": "error", "error": f"privacy invariant violated: {exc}"})
            return
        except Exception as exc:  # noqa: BLE001 -- surface it in-band, same reason
            emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            return

        emit({"type": "result", **_build_response(self.router, decision, query, context)})

    def log_message(self, fmt, *args) -> None:  # keep the default noisy access log
        print(f"[two-brain-api] {self.address_string()} - {fmt % args}")


def serve(tier: Tier, host: str, port: int, trace: bool = True) -> None:
    # Bind *before* building the router, and refuse to share the port.
    #
    # Both halves of that are load-bearing, and the second is the subtle one:
    # `socketserver` sets `allow_reuse_address = 1` by default, and on Windows
    # SO_REUSEADDR permits binding a port that is **already in LISTEN**. So a
    # second server starts "successfully" and the two race for connections.
    # That is not hypothetical -- it happened here: a stale server from the
    # previous session held 8765, a freshly-started one bound alongside it
    # without complaint, and the *old build* kept answering. A UI calling a
    # newly-added endpoint got 404s and reported the backend as offline, three
    # layers away from the actual cause.
    #
    # Turning reuse off makes the duplicate bind fail loudly instead. The cost
    # is a possible TIME_WAIT delay when restarting quickly on Unix, which is a
    # far better failure than silently serving stale code.
    #
    # Binding first also means an unusable port costs nothing: no Genie session
    # is created (~12s) only to be thrown away.
    class _ExclusiveServer(ThreadingHTTPServer):
        allow_reuse_address = False

    try:
        httpd = _ExclusiveServer((host, port), _Handler)
    except OSError as exc:
        raise SystemExit(
            f"cannot bind {host}:{port} -- {exc}\n"
            f"Another two-brain server is probably still running on that port. "
            f"Stop it, or pass --port to run alongside it."
        ) from exc

    router = TwoBrainRouter(tier=tier, trace=trace)
    httpd.RequestHandlerClass = type(
        "_BoundHandler", (_Handler,), {"router": router, "route_lock": threading.Lock()}
    )
    print(f"two-brain-router API -> http://{host}:{port} (tier={tier})")
    print(f"  POST http://{host}:{port}/route   {{\"query\": \"...\", \"context\": \"\"}}")
    if trace:
        # Said plainly and once, at startup. The trace exists so you can *see*
        # that PII stayed on-device, which means it necessarily prints that PII
        # -- and a terminal is a place data ends up. Better to name that than to
        # let someone discover it while screen-sharing.
        print(
            "  trace: ON -- every call in/out is printed below, including raw PII\n"
            "         (--no-trace to silence it, TWO_BRAIN_TRACE_REDACT=1 to show\n"
            "          placeholders instead of real values)"
        )
    # `route()` is serialized by `route_lock`, so trace blocks never interleave
    # even though this is a threading server.
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        router.close()  # closes fast_brain and escalation_brain, whichever are real
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="two_brain_router.api")
    parser.add_argument("--tier", choices=["pc", "mobile"], default="pc")
    parser.add_argument("--host", default="127.0.0.1", help="default is loopback-only; see module docstring")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="don't print each call's input/output (the trace includes raw PII)",
    )
    args = parser.parse_args(argv)
    serve(args.tier, args.host, args.port, trace=not args.no_trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
