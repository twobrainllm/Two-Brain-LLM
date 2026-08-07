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

PROXY_PATHS = ("/route", "/health")
MAX_BODY_BYTES = 8 * 1024 * 1024


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
            with urllib.request.urlopen(request, timeout=300) as resp:
                payload, status = resp.read(), resp.status
        except urllib.error.HTTPError as exc:
            payload, status = exc.read(), exc.code
        except urllib.error.URLError as exc:
            # The router being down is an expected state, not a stack trace.
            # The UI already handles a failed /route as "offline preview".
            payload = f'{{"error": "router unreachable: {exc.reason}"}}'.encode()
            status = 502

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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
        # Static assets are versioned in index.html, but a phone that already
        # cached an unversioned copy needs telling. Cheap insurance on a demo
        # server; this is not a production static host.
        if not self.path.split("?")[0] in PROXY_PATHS:
            self.send_header("Cache-Control", "no-cache")
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
