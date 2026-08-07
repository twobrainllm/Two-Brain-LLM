#!/usr/bin/env python3
"""
confidence_estimator.py

Reference implementation of the confidence-estimation strategies discussed
in L_INTERFACE_CONTRACT.md, for O's Route decision. Runtime-agnostic: talks
to L over the same /v1/chat/completions contract everything else in this
repo uses, so it works unchanged against mock_phone_brain_server.py or the
real on-device Genie/GenieX server -- swap --base-url only.

THIS IS SCAFFOLDING FOR WHOEVER OWNS O, NOT A FINISHED ORCHESTRATOR. It
exists so the self-consistency vs. self-reported-confidence tradeoff is
something you can actually run and measure, not just reason about. Wire
the winning strategy into the real routing/mask/compress logic separately.

CLI usage:
    python confidence_estimator.py --prompt "..." --strategy hybrid
    python confidence_estimator.py --prompt "..." --strategy self_consistency
    python confidence_estimator.py --prompt "..." --strategy self_reported

Library usage:
    from confidence_estimator import estimate_confidence
    result = estimate_confidence(base_url, model, prompt, strategy="hybrid")
    if result.should_escalate:
        ...
"""

import argparse
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConfidenceResult:
    strategy: str
    confidence: float          # 0.0-1.0, higher = more confident L can handle it
    should_escalate: bool
    calls_made: int
    total_latency_sec: float
    raw_responses: list = field(default_factory=list)


def _call_l(base_url: str, model: str, prompt: str, temperature: float,
            max_tokens: int = 256) -> dict:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_text(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def _word_overlap_similarity(a: str, b: str) -> float:
    """Cheap agreement heuristic: Jaccard similarity over lowercased word
    sets. Good enough for a first pass and has zero extra dependencies;
    swap in embedding similarity later if you have an embedding model
    available locally and want a less crude signal."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def self_consistency(base_url: str, model: str, prompt: str,
                      n_samples: int = 3, temperature: float = 0.6,
                      confidence_threshold: float = 0.5) -> ConfidenceResult:
    """Sample L n_samples times, use pairwise agreement as the confidence
    signal. Expensive: n_samples sequential on-device calls per query --
    measure real latency on the S25 before committing to this for every
    query in the demo."""
    t0 = time.perf_counter()
    responses = [
        _extract_text(_call_l(base_url, model, prompt, temperature=temperature))
        for _ in range(n_samples)
    ]
    elapsed = time.perf_counter() - t0

    if len(responses) < 2:
        confidence = 0.0
    else:
        pairs = [
            _word_overlap_similarity(responses[i], responses[j])
            for i in range(len(responses))
            for j in range(i + 1, len(responses))
        ]
        confidence = sum(pairs) / len(pairs)

    return ConfidenceResult(
        strategy="self_consistency",
        confidence=round(confidence, 3),
        should_escalate=confidence < confidence_threshold,
        calls_made=n_samples,
        total_latency_sec=round(elapsed, 3),
        raw_responses=responses,
    )


SELF_REPORT_SUFFIX = (
    "\n\nAfter answering, on a new line output exactly: "
    "CONFIDENCE: <a number from 0 to 100>"
)
_CONF_RE = re.compile(r"CONFIDENCE:\s*(\d{1,3})", re.IGNORECASE)


def self_reported(base_url: str, model: str, prompt: str,
                   temperature: float = 0.2,
                   confidence_threshold: float = 0.5) -> ConfidenceResult:
    """Single call: ask L to answer and self-rate confidence in the same
    response. Cheap (1 call), but the self-rating isn't guaranteed to be
    well-calibrated -- treat it as a heuristic signal, not a probability."""
    t0 = time.perf_counter()
    text = _extract_text(_call_l(base_url, model, prompt + SELF_REPORT_SUFFIX, temperature=temperature))
    elapsed = time.perf_counter() - t0

    match = _CONF_RE.search(text)
    confidence = (int(match.group(1)) / 100.0) if match else 0.0
    answer_only = _CONF_RE.sub("", text).strip()

    return ConfidenceResult(
        strategy="self_reported",
        confidence=round(confidence, 3),
        should_escalate=confidence < confidence_threshold,
        calls_made=1,
        total_latency_sec=round(elapsed, 3),
        raw_responses=[answer_only],
    )


def hybrid(base_url: str, model: str, prompt: str,
           borderline_band: tuple = (0.35, 0.65),
           confidence_threshold: float = 0.5) -> ConfidenceResult:
    """Cheap self-report first (1 call). Only pay for a second sample (2
    calls total) when the self-report lands in the borderline band -- i.e.
    when it's genuinely ambiguous, not on every query. Realistic middle
    ground for a phone: most queries stay cheap, only the uncertain ones
    cost extra."""
    t0 = time.perf_counter()
    first = self_reported(base_url, model, prompt, confidence_threshold=confidence_threshold)

    lo, hi = borderline_band
    if not (lo <= first.confidence <= hi):
        first.strategy = "hybrid(self_report_only)"
        return first

    second_text = _extract_text(_call_l(base_url, model, prompt, temperature=0.6))
    elapsed = time.perf_counter() - t0
    agreement = _word_overlap_similarity(first.raw_responses[0], second_text)
    blended = round((first.confidence + agreement) / 2, 3)

    return ConfidenceResult(
        strategy="hybrid(self_report+confirm)",
        confidence=blended,
        should_escalate=blended < confidence_threshold,
        calls_made=2,
        total_latency_sec=round(elapsed, 3),
        raw_responses=[first.raw_responses[0], second_text],
    )


STRATEGIES = {
    "self_consistency": self_consistency,
    "self_reported": self_reported,
    "hybrid": hybrid,
}


def estimate_confidence(base_url: str, model: str, prompt: str,
                         strategy: Literal["self_consistency", "self_reported", "hybrid"] = "hybrid",
                         **kwargs) -> ConfidenceResult:
    return STRATEGIES[strategy](base_url, model, prompt, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="llama-3.2-3b-instruct")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="hybrid")
    args = ap.parse_args()

    result = estimate_confidence(args.base_url, args.model, args.prompt, strategy=args.strategy)
    print(f"strategy:        {result.strategy}")
    print(f"confidence:      {result.confidence}")
    print(f"should_escalate: {result.should_escalate}")
    print(f"calls_made:      {result.calls_made}")
    print(f"latency_sec:     {result.total_latency_sec}")
    for i, r in enumerate(result.raw_responses):
        print(f"\n--- response {i + 1} ---\n{r[:300]}")


if __name__ == "__main__":
    main()
