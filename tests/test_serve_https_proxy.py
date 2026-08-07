"""`ui/serve_https.py`'s proxy must forward a stream incrementally.

This is the phone's only path to the router: an HTTPS page may not `fetch()` an
`http://` URL, so everything the UI calls goes through that proxy. The first
version of it did `resp.read()` -- the whole body -- before replying, which is
correct for a finished JSON document and wrong for the two endpoints that exist
*because* they are incremental.

The symptom was not "no animation". `route_stream` emits a `tier` frame the
moment the router decides to escalate, which the UI renders as the cloud
bubble's "Thinking..." placeholder, and that decision can precede the first
cloud token by 15s or more. Buffered, that frame arrives at the same instant as
the answer it was meant to precede -- so the phone showed a finished local
answer and no sign anything else was coming.

Timing-based, because buffering is a timing property and nothing else
distinguishes it: a buffered proxy returns the identical bytes. The margin is
deliberately loose (frames must span more than one gap, not all N-1 of them) so
this does not become a flaky test on a loaded machine.
"""
from __future__ import annotations

import importlib.util
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

N_FRAMES = 4
GAP_S = 0.15


def _load_serve_https():
    """Imported by path: `ui/` is not a package and is not on sys.path."""
    path = Path(__file__).resolve().parents[1] / "ui" / "serve_https.py"
    spec = importlib.util.spec_from_file_location("serve_https", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Upstream(BaseHTTPRequestHandler):
    """Stands in for `api.py`: one incremental endpoint, one ordinary one."""

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/route/sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(N_FRAMES):
                self.wfile.write(f'event: delta\ndata: {{"n": {i}}}\n\n'.encode())
                self.wfile.flush()
                time.sleep(GAP_S)
            return
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: A003 -- silence the default stderr log
        pass


@pytest.fixture
def proxy():
    """(base_url, handler_class), with an upstream behind it. Both torn down."""
    serve_https = _load_serve_https()
    serve_https.Handler.log_message = lambda *a: None

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    serve_https.Handler.api_base = f"http://127.0.0.1:{upstream.server_address[1]}"

    server = ThreadingHTTPServer(("127.0.0.1", 0), serve_https.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", serve_https.Handler
    finally:
        server.shutdown()
        upstream.shutdown()


def _post(url: str):
    return urllib.request.Request(
        url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
    )


def test_sse_frames_arrive_as_they_are_produced(proxy):
    base, _ = proxy
    start = time.perf_counter()
    arrivals = []
    with urllib.request.urlopen(_post(f"{base}/route/sse"), timeout=30) as resp:
        # Forwarded, not hardcoded to JSON -- the browser and the UI's frame
        # parser both key off this.
        assert resp.headers.get("Content-Type") == "text/event-stream"
        # A length would have to be known up front, which means buffering.
        assert resp.headers.get("Content-Length") is None
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            arrivals.append((time.perf_counter() - start, chunk.count(b"event:")))

    assert sum(n for _, n in arrivals) == N_FRAMES, "frames were lost in transit"
    spread = arrivals[-1][0] - arrivals[0][0]
    assert spread > GAP_S, (
        f"all {N_FRAMES} frames landed within {spread:.3f}s -- the proxy buffered "
        f"the response instead of relaying it"
    )


def test_an_ordinary_response_is_still_sent_whole(proxy):
    """The streaming path must not cost the common case its Content-Length."""
    base, _ = proxy
    with urllib.request.urlopen(_post(f"{base}/health"), timeout=10) as resp:
        assert resp.headers.get("Content-Length") == "12"
        assert resp.read() == b'{"ok": true}'


def test_a_dead_router_is_a_502_not_a_traceback(proxy):
    """The UI treats a failed /route as "offline preview", so this has to be a
    clean HTTP error with a JSON body rather than a crashed handler."""
    base, handler = proxy
    handler.api_base = "http://127.0.0.1:9"  # reserved discard port; nothing listens
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(_post(f"{base}/route"), timeout=10)
    assert excinfo.value.code == 502
    assert b"router unreachable" in excinfo.value.read()
