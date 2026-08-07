#!/usr/bin/env python3
"""
bench_phone_brain.py

Smoke-tests the on-device "L" (fast brain) model once it's being served
locally on the phone via an OpenAI-compatible endpoint (e.g. GenieX's
`geniex serve`, or your own thin wrapper around the Genie/QAIRT runtime).

Typical setup before running this:

    adb forward tcp:8000 tcp:8000
    # on-device (adb shell, or however GenieX exposes it):
    geniex serve --bundle /data/local/tmp/genie_bundle_l --port 8000

Then from your dev machine:

    python bench_phone_brain.py --base-url http://localhost:8000 \
        --model llama-3.2-3b-instruct

It sends a small battery of prompts spanning easy / medium / hard
difficulty (mirroring what O will eventually see) and reports latency and
tokens/sec per prompt -- these are the raw numbers you want for the
technical-implementation writeup, and this same battery is a good starting
point for the "escalation rate by difficulty" plot in the demo.

NOTE: field names below (choices[0].message.content, usage.completion_tokens)
follow the standard OpenAI chat-completions response shape. If GenieX's
server diverges slightly, adjust `call_local_model` accordingly -- check
its actual response with `curl` once before trusting this script's output.
"""

import argparse
import json
import time
import urllib.request

TEST_PROMPTS = [
    {"tag": "easy",   "prompt": "What's a good rule of thumb for staying hydrated during exercise?"},
    {"tag": "easy",   "prompt": "Rewrite this more politely: 'send me the file now'"},
    {"tag": "medium",  "prompt": "Summarize the difference between TCP and UDP in two sentences."},
    {"tag": "hard",   "prompt": "A patient on warfarin is prescribed a short course of fluconazole. Walk through the interaction risk and what should be monitored."},
    {"tag": "hard",   "prompt": "Write a proof that there are infinitely many primes."},
]


def call_local_model(base_url: str, model: str, prompt: str, max_tokens: int = 256) -> dict:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0

    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {}) or {}
    completion_tokens = usage.get("completion_tokens")
    tok_per_sec = (completion_tokens / elapsed) if completion_tokens else None

    return {
        "elapsed_sec": round(elapsed, 3),
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(tok_per_sec, 2) if tok_per_sec else None,
        "response": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="llama-3.2-3b-instruct")
    args = ap.parse_args()

    results = []
    for item in TEST_PROMPTS:
        print(f"\n[{item['tag']}] {item['prompt'][:70]}...")
        try:
            r = call_local_model(args.base_url, args.model, item["prompt"])
            r["tag"] = item["tag"]
            r["prompt"] = item["prompt"]
            results.append(r)
            print(f"  -> {r['elapsed_sec']}s, {r['tokens_per_sec']} tok/s")
            print(f"  -> {r['response'][:150]}...")
        except Exception as e:
            print(f"  !! request failed: {e}")

    if results:
        print("\n=== Summary ===")
        print(f"{'tag':8s} {'latency':>10s} {'tok/s':>8s}")
        for r in results:
            print(f"{r['tag']:8s} {r['elapsed_sec']:>9}s {str(r['tokens_per_sec']):>8}")


if __name__ == "__main__":
    main()
