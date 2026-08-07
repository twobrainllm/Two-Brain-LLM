#!/usr/bin/env python3
"""
mock_phone_brain_server.py

Stand-in for the real on-device L server. Speaks the exact same contract
described in L_INTERFACE_CONTRACT.md, so whoever is building the
orchestrator (O) can develop and test routing logic today, without the S25
or the exported Genie bundle. Swap the base URL later for the real device
-- nothing else should need to change.

No dependencies beyond the Python standard library (ease-of-install matters
for a hackathon teammate who just wants to start coding).

Run:
    python mock_phone_brain_server.py --port 8000

Behavior:
  - Simulates latency roughly in line with a real quantized 3B on-device
    model (not precise, just enough to make timing-dependent O code behave
    sanely during dev).
  - Deliberately gives shakier, hedging answers on prompts that look "hard"
    (crude keyword heuristic) and confident answers on easy ones, so
    self-consistency-based routing logic in O actually has something to
    catch. Run the same prompt 2-3 times and compare -- hard prompts will
    visibly disagree more across calls.
"""

import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HARD_KEYWORDS = [
    "proof", "differential", "diagnos", "interaction risk", "derive",
    "prove", "optimi", "algorithm complexity", "legal implications",
]


def fake_answer(prompt: str):
    is_hard = any(k in prompt.lower() for k in HARD_KEYWORDS)
    if is_hard:
        # Vary the phrasing/content each call on purpose, so repeated
        # sampling disagrees -- this is what a genuinely under-confident
        # small model looks like under self-consistency checks.
        hedge = random.choice([
            "This likely involves {t}, though I'd want to double check the specifics.",
            "I believe it relates to {t}, but I'm not fully certain here.",
            "My best guess touches on {t} -- worth verifying with a stronger source.",
        ])
        text = hedge.format(t=prompt[:40].strip())
        fake_confidence = random.randint(20, 45)
    else:
        text = f"Sure — here's a concise take on: {prompt[:60].strip()}"
        fake_confidence = random.randint(75, 96)
    # Only real L would emit this line if the prompt explicitly asked for
    # it (see confidence_estimator.py's SELF_REPORT_SUFFIX) -- the mock
    # includes it unconditionally so self_reported/hybrid strategies have
    # something real to parse during dev, whether or not the caller added
    # the suffix.
    text = f"{text}\nCONFIDENCE: {fake_confidence}"
    return text, is_hard


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        messages = body.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""

        # simulate latency roughly scaled like a real quantized 3B model
        time.sleep(random.uniform(0.4, 1.2))

        text, is_hard = fake_answer(prompt)
        completion_tokens = max(8, len(text.split()))
        prompt_tokens = max(1, len(prompt.split()))

        response = {
            "id": "mock-cmpl-1",
            "object": "chat.completion",
            "model": body.get("model", "mock-llama-3.2-3b-instruct"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[mock-L] {self.address_string()} - {fmt % args}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"Mock phone-brain (L) server -> http://localhost:{args.port}/v1/chat/completions")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
