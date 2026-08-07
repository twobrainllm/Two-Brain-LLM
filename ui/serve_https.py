#!/usr/bin/env python3
"""Serve the chat UI over HTTPS, so the camera and microphone work.

    python3 ui/serve_https.py            # https://<this-machine>:8643

Browsers gate `getUserMedia` behind a *secure context*: over plain `http://`
on a LAN address, the camera and mic buttons are dead no matter how correct
the JavaScript is. `localhost` is exempt, which is why they work on the
laptop and not on the phone. This closes that gap for the phone.

Two jobs, deliberately in one process:

1. **Serves `ui/` over TLS**, with a self-signed certificate generated on
   first run (openssl, already present on macOS and Linux).
2. **Proxies `/route` and `/health`** through to `api.py` on plain HTTP.
   That matters: an HTTPS page may not `fetch()` an `http://` URL -- browsers
   block it as mixed content. Proxying keeps everything on one secure origin,
   which also means no CORS headers are involved at all.

So the router stays exactly as it is. `api.py` needs no TLS, no changes, and
keeps its loopback-only default -- this process is the only thing talking to
it, over 127.0.0.1.

The certificate is self-signed, so the phone will show a warning once; tap
through it ("Advanced" -> "Proceed"). That is expected for a LAN demo and is
not a sign anything is wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
CERT_DIR = UI_DIR / ".certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"

#: Everything api.py serves. Kept in sync with its handler by hand -- a path
#: missing here is not a 404, it is the HTTPS page silently failing to reach a
#: backend that is running, which is a much more confusing symptom.
#:
#: /route/stream and /route/sse are proxied incrementally -- see `_relay_stream`
#: and `STREAMING_TYPES`. That is not cosmetic: the UI opens the cloud's bubble
#: with a "Thinking..." placeholder on the `tier` frame, which the router emits
#: the moment it *decides* to escalate and up to 15s before the first cloud
#: token. Buffering the response (which this proxy originally did) delivers that
#: frame at the same instant as the answer it was meant to precede, so the phone
#: shows a finished local answer and no sign anything else is coming.
PROXY_PATHS = (
    "/route",
    "/route/stream",
    "/route/sse",
    "/health",
    "/models",
)
MAX_BODY_BYTES = 8 * 1024 * 1024

#: Content types the proxy must forward incrementally instead of buffering.
#: Matches what `api.py` sends for `/route/sse` and `/route/stream`; anything
#: else is a complete document and is relayed in one piece.
STREAMING_TYPES = frozenset({"text/event-stream", "application/x-ndjson"})


def lan_ip() -> str:
    """Best-effort LAN address. No packet is actually sent -- connecting a UDP
    socket just asks the OS which interface it *would* use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def ensure_cert(ip: str) -> bool:
    """Generate a self-signed cert on first run. Returns False if openssl is
    missing, so the caller can say so plainly rather than crashing."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return True
    if not shutil.which("openssl"):
        return False

    CERT_DIR.mkdir(exist_ok=True)
    print(f"generating a self-signed certificate for {ip} ...")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
            "-days", "365", "-subj", "/CN=two-brain-ui",
            "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    # The key is readable only by this user; it is a throwaway for a LAN demo
    # but there is no reason to leave it world-readable.
    os.chmod(KEY_FILE, 0o600)
    return True


class Handler(SimpleHTTPRequestHandler):
    api_base = "http://127.0.0.1:8765"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def _send_buffered(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _relay_stream(self, resp, content_type: str) -> None:
        """Forward an incremental response as it arrives, not once it ends.

        Deliberately sends **no Content-Length**: the length is unknowable
        until the router finishes, and declaring one would mean buffering,
        which is the entire bug this exists to avoid. `protocol_version` is
        HTTP/1.0 here, so the client reads until the connection closes and no
        chunked framing is needed.

        `read1` rather than `read(n)`: `read(n)` blocks until it has n bytes or
        the stream ends, which would re-introduce buffering at a smaller
        granularity. `read1` returns whatever one underlying read produced --
        for SSE that is a frame as soon as `api.py` flushes it.

        Each write is flushed, for the same reason `api.py` flushes: without
        it the frames sit in this process's buffer instead of the previous one.
        """
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the phone navigated away mid-answer; not an error

    def _proxy(self, method: str) -> None:
        body = None
        if method == "POST":
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self.send_error(400, "bad Content-Length")
                return
            if length > MAX_BODY_BYTES:
                self.send_error(413, "body too large")
                return
            body = self.rfile.read(length) if length else b""

        request = urllib.request.Request(
            f"{self.api_base}{self.path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            resp = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as exc:
            # An error body is small and complete, so buffering it is correct.
            self._send_buffered(exc.code, "application/json", exc.read())
            return
        except urllib.error.URLError as exc:
            # The router being down is an expected state, not a stack trace.
            # The UI already handles a failed /route as "offline preview".
            self._send_buffered(
                502, "application/json",
                json.dumps({"error": f"router unreachable: {exc.reason}"}).encode(),
            )
            return

        # The upstream Content-Type is forwarded rather than hardcoded to JSON.
        # It is what says "this one is incremental" -- api.py sends
        # `text/event-stream` for /route/sse and `application/x-ndjson` for
        # /route/stream, and neither carries a Content-Length.
        with resp:
            content_type = resp.headers.get("Content-Type", "application/json")
            if content_type.split(";")[0].strip() in STREAMING_TYPES:
                self._relay_stream(resp, content_type)
            else:
                self._send_buffered(resp.status, content_type, resp.read())

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] in PROXY_PATHS:
            self._proxy("GET")
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] in PROXY_PATHS:
            self._proxy("POST")
            return
        self.send_error(404, "no such endpoint")

    def end_headers(self) -> None:
        # index.html references app.js/styles.css without a version query, so
        # a phone will happily serve a cached copy after a pull -- new markup
        # driven by old script, which reads as "the buttons don't work" rather
        # than as a caching problem. Cost an hour to diagnose once already.
        # no-store rather than no-cache: no-cache still permits a stored copy
        # revalidated by ETag, and mobile browsers are inconsistent about it.
        if not self.path.split("?")[0] in PROXY_PATHS:
            self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8643)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--api", default="http://127.0.0.1:8765", help="where api.py is listening")
    args = parser.parse_args()

    ip = lan_ip()
    if not ensure_cert(ip):
        print("openssl not found -- cannot generate a certificate.", file=sys.stderr)
        print("Install it, or serve over plain http and accept that the", file=sys.stderr)
        print("camera and mic will stay disabled.", file=sys.stderr)
        return 1

    Handler.api_base = args.api.rstrip("/")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"\n  Two-Brain UI over HTTPS")
    print(f"  on this machine   https://localhost:{args.port}")
    print(f"  from your phone   https://{ip}:{args.port}")
    print(f"  proxying /route and /health -> {Handler.api_base}")
    print("\n  The certificate is self-signed, so the phone will warn once.")
    print("  Tap Advanced -> Proceed. Camera and mic then work.\n")

    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
