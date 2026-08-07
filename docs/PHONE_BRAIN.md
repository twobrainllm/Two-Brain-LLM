# Phone Brain (L) — What's Been Built

Summary of the work in `src/phone_brain/` (branch `local_brain`, commit
`e7130ce`), what it decides, what it verifies, and how it connects to the
router in `src/two_brain_router/`.

**In one line:** the Mobile tier's fast brain, built end-to-end — plus a mock
server so orchestrator work isn't blocked on hardware.

> **Now wired in.** This document was written as an audit *before* the phone
> brain was connected to the router. It has since been absorbed as
> `routing/brains.py::PhoneFastBrain` on branch `js/orchestrator`, resolving
> four of the five reconciliation points below. For how the router actually
> uses it today, read [`ORCHESTRATOR.md`](ORCHESTRATOR.md); this file is kept
> as the record of what was built on the phone side and what it cost to
> integrate.

| | |
|---|---|
| Target device | Samsung Galaxy S25 — Snapdragon 8 Elite, 12 GB RAM |
| Model | Llama-3.2-3B-Instruct, **w4a16** |
| Toolchain | Qualcomm AI Hub (`qai_hub_models`) → Genie/QNN bundle → `adb push` → GenieX serve |
| Interface | OpenAI-compatible `POST /v1/chat/completions` |
| Runtime deps | none — Python standard library only |

---

## The strategic move: around QUAD's compiler, not through it

This does **not** use QUAD's `convert_model`. That was the right call —
[`GAPS.md`](GAPS.md) #3/#3b document two independent, unfixable-from-here
defects in that toolchain, and `CLAUDE.md` says not to retry it. Instead it
uses Qualcomm AI Hub's cloud compile service, which is Qualcomm's own
maintained path for this exact chipset and is independent of the broken
`qairt-converter` binary.

`superpowers/deploy-local-brain-npu.md` reached the same conclusion
independently for the AI PC tier (via Genie's own C API rather than ONNX
Runtime GenAI's QNN EP, per that doc's Phase 1 addendum) — and, unlike this
folder, that effort is finished and merged: `NpuFastBrain` in
`routing/brains.py` is wired into `TwoBrainRouter` for the `pc_3b` tier
today. Two tiers, two people, same escape route — worth noting as a signal
that bypassing `convert_model` via Qualcomm's own tooling is the pattern to
standardize on, and a concrete example of what "absorbed into the seam"
looks like for `phone_brain` to follow.

---

## The four pieces

### 1. Export — `export_phone_brain.sh`

One command: `./export_phone_brain.sh genie_bundle_l_phone`. Compiles
Llama-3.2-3B-Instruct into QNN context binaries for
`qualcomm-snapdragon-8-elite` at context length 2048 (overridable via
`CONTEXT_LEN`, with a note to trim it under device memory pressure).

Prereqs are spelled out and are all real gates: an AI Hub account + API token,
and HuggingFace access to `meta-llama/Llama-3.2-3B-Instruct` (gated weights).

**Decision recorded — w4a16, not w4a8.** The reasoning is documented rather
than assumed: w4a8 isn't broken or unsupported, it's just only reachable by
assembling your own `submit_quantize_job` + `submit_compile_job` pipeline. The
packaged one-liner ships Qualcomm's published, benchmarked w4a16 recipe. Ship
the proven one, treat w4a8 as a phase-2 optimization. The script also tells you
to run `export --help` and confirm what flags your installed version actually
exposes rather than trusting the doc.

### 2. The contract — `L_INTERFACE_CONTRACT.md`

The most valuable artifact in the folder. Standard OpenAI chat-completions
request/response, and the whole point is stated plainly:

> Whoever builds O does not need to know anything about Genie, QNN, GenieX, or
> the S25 — just this contract. It's identical whether O is talking to the real
> on-device model or the mock server.

Two structural decisions it pins down:

- **L never decides to escalate.** It only ever produces a number; the
  orchestrator owns the decision every time — explicitly parallel to the
  privacy mask, where L also doesn't get to decide what's sensitive. This
  matches how `RoutePolicy` already works in our router.
- **Don't build routing around logprobs.** Whether the device server exposes
  them is unconfirmed; treat them as a bonus signal if they appear, never a
  dependency.

Error handling is specified too: timeout or non-200 means "L failed → escalate
to cloud," never retry indefinitely or block the user.

### 3. Mock server — `mock_phone_brain_server.py`

Stdlib-only stand-in speaking the same contract, so all routing logic can be
developed and tested today without the S25. Two details that make it useful
rather than decorative:

- Sleeps 0.4–1.2 s per call, so timing-dependent orchestrator code behaves
  sanely during development.
- **Deliberately hedges and varies its wording on hard-looking prompts** (crude
  keyword match) and answers confidently on easy ones. Repeated sampling of a
  hard prompt genuinely disagrees — so self-consistency logic has something
  real to catch instead of trivially agreeing with itself.

It emits a `CONFIDENCE: <n>` line unconditionally (20–45 for hard prompts,
75–96 for easy), so the self-report parsers have something to chew on whether
or not the caller appended the suffix.

### 4. Confidence estimation — `confidence_estimator.py`

Three strategies against the same contract, runtime-agnostic (`--base-url`
switches mock ↔ real device). This is the piece that replaces surface-feature
difficulty scoring with a real signal.

| Strategy | Calls | How it works |
|---|---|---|
| `self_consistency` | 3 (fixed) | Sample at temp 0.6, score pairwise Jaccard word overlap; agreement = confidence |
| `self_reported` | 1 | Append "output CONFIDENCE: 0-100", regex the number out, strip it from the answer |
| `hybrid` | 1–2 | Self-report first; pay for a second sample **only** when confidence lands in the borderline band (0.35–0.65), then blend with agreement |

**Decision recorded — self-reported, with hybrid as the fallback.** Rationale
is cost, and it's stated honestly: self-consistency means 2–3 sequential
on-device inferences *before you even know whether to escalate* — on a phone
that's 2–3× the latency and battery of a single answer, "not a rounding error."
The doc is candid that self-reported confidence isn't necessarily
well-calibrated, and says to test it against the real device early and fall
back to `hybrid()` rather than trying to fix calibration by prompt-tweaking.

`verify_confidence_estimator.sh` runs all three against the mock on an easy and
a hard prompt. That's what produced the measured tradeoff already cited in the
README: **self-consistency spends a fixed 3 calls regardless of difficulty;
hybrid spends 1 on the easy prompt and 2 on the hard one.**

### Plus: benchmark harness — `bench_phone_brain.py`

Five prompts tagged easy/medium/hard, reporting latency and tokens/sec each
plus a summary table. It's a smoke test *and* the source of the real numbers
for two things: the technical writeup, and the "escalation rate by difficulty"
plot for the demo.

Sanity bounds to check on the S25 are listed and are the right ones — public
benchmarks put a similar 3B/w4a16 setup around 10–13 tok/s on this chipset
class (explicitly "a rough sanity bound, not a guarantee"), memory headroom via
`dumpsys meminfo`, and thermal behavior across back-to-back prompts, since
sustained NPU load throttles on a phone in ways it wouldn't on a PC.

---

## Status

| | |
|---|---|
| **Verified working** | Everything on the mock path — mock server, all three confidence strategies, the benchmark harness against the mock. Since integration, also the full router path: `TwoBrainRouter --tier mobile` routes real confidence-driven decisions against the mock server, covered by `tests/test_orchestrator.py` |
| **Written, not yet run** | The export (needs AI Hub token + gated Llama access), device push/serve, and every real-device number |
| **Explicitly open** | Shared vs. split weights for the orchestrator — does O reuse this same L with a routing-only system prompt, or get its own smaller model? Moot for now: O does no inference of its own, so it needs no weights. The question returns if the router ever summarizes context with the fast brain ([WALKTHROUGH next-step #5](WALKTHROUGH.md#next-steps)) |

The Track A / Track B split (needs the phone / needs nothing) is the same
instinct as this project's `data/` fixtures: keep the dependent work
unblocked while the hardware path is still in progress.

---

## How it connects to `two_brain_router`

It fills **two** of our seams, one of which is
[next step #4](WALKTHROUGH.md#next-steps) — the biggest quality win available:

| Our seam | What phone_brain provides | Status |
|---|---|---|
| `routing/brains.py::LocalFastBrain` | A real HTTP-backed brain for the Mobile tier | **Absorbed** as `PhoneFastBrain` |
| `signals/difficulty.py::DifficultyEstimator` | `confidence_estimator.py`, inverted — `score = 1 - confidence` | **Absorbed** as `signals/confidence.py` |
| `data/profile_workload/mobile_1b.json` (mocked) | `bench_phone_brain.py` produces exactly the real numbers this mock stands in for | **Still open** — needs a real S25 run |

Vocabulary maps directly: **L** = fast brain, **O** = router, **C** = deep brain.

### Five things to reconcile before merging — 4 of 5 resolved

Resolved on branch `js/orchestrator`. See
[`ORCHESTRATOR.md`](ORCHESTRATOR.md) for how the routing actually works now.

**1. Privacy ordering — the one that matters. ✅ Resolved.** The original
concern: self-reported confidence sends the **raw prompt** to L, and
`--base-url` is a plain argument, so pointing it off-device would leak raw PII
silently. Fixed structurally rather than by convention — `PhoneFastBrain`
implements the `Brain` protocol, whose parameter is literally `masked_query`,
and the router masks before it calls *any* brain. Belt and braces:
`PhoneFastBrain` also refuses a non-loopback host unless `allow_remote=True`.
`tests/test_orchestrator.py::test_pii_never_reaches_the_phone_or_the_cloud_unmasked`
asserts on the bytes that actually crossed the socket.

**2. Self-report merges two steps of `route()`. ✅ Resolved.** Confirmed as a
real structural change, and made deliberately: `Brain.reports_confidence`
selects between two decision shapes, and the self-rating shape asks the brain
before deciding. Escalating does discard a paid-for answer — that cost is
reported in `RouteDecision.est_latency_ms` and called out in the notes rather
than hidden. One refinement fell out of building it: the latency budget is
checked *before* the call, never after, because re-checking it while holding a
finished answer could only make total latency worse.

**3. The threshold lives in two places. ✅ Resolved.** O consumes `confidence`
and ignores `confidence_estimator.py`'s `should_escalate` entirely, exactly as
their contract prescribes. `signals/confidence.py` inverts confidence into a
difficulty so the existing `RoutePolicy.escalate_threshold` is still the only
threshold in the system — which is also why `routing/policy.py` needed no
changes at all.

**4. Model mismatch with our fixtures. ⬜ Still open.** They're on
Llama-3.2-3B w4a16; `data/*/mobile_1b.json` still says Qwen2.5-1.5B int4
(ttft 180 ms, 27.5 ms/token) and is still `_mock: true`. Real numbers need a
`bench_phone_brain.py` run against the actual S25, with a receipt per the
`CLAUDE.md` rule. **This is not cosmetic:** that fixture drives the latency
budget pre-check that decides whether the phone is asked at all, so the mobile
tier is currently reasoning about the wrong model's speed.

**5. `test_phone_brain.py` is not a pytest test. ✅ Resolved.** Renamed to
`bench_phone_brain.py`, with references updated across the phone-brain docs.
(The byte-identical `PHONE_DEPLOYMENT_GUIDE_final.md` duplicate flagged under
Housekeeping below was deleted in the same pass.)

---

## Where this code lives — done

This section originally proposed breaking `routing/brains.py` into a
`routing/brains/` subpackage (`base.py`/`stub.py`/`openai_http.py`). That
didn't happen, and shouldn't: `NpuFastBrain` had already established the
opposite precedent — add a class to the **flat** module, import its runtime
lazily, and register it in `router.py::_build_fast_brain` behind an env var.
`PhoneFastBrain` followed that precedent. What actually landed:

```
src/two_brain_router/
  routing/brains.py   # PhoneFastBrain, alongside NpuFastBrain /
                      # LocalFastBrain / CloudDeepBrain. urllib imported
                      # lazily inside the method, mirroring NpuFastBrain's
                      # lazy `import onnxruntime_qnn`. Generic over any
                      # OpenAI-shaped endpoint -- not phone-specific.
  routing/router.py   # _build_fast_brain grew a mobile branch behind
                      # TWO_BRAIN_PHONE_BRAIN=1, same shape as the NPU switch
  signals/
    confidence.py     # self-report parsing + confidence->difficulty. Pure,
                      # no I/O -- so it unit-tests like policy.py does.

src/phone_brain/      # unchanged, still per-device tooling by nature
  export_phone_brain.sh
  mock_phone_brain_server.py
  bench_phone_brain.py            # renamed from test_phone_brain.py (see #5)
  verify_confidence_estimator.sh
  confidence_estimator.py         # kept: the strategy *comparison* harness

docs/ORCHESTRATOR.md              # how the router decides, both shapes
```

Two deviations from the plan above, both deliberate:

- **Only the `self_reported` strategy was absorbed**, not all three.
  `L_INTERFACE_CONTRACT.md` decided self-report, and `hybrid`/`self_consistency`
  are multi-call strategies whose value is in *comparing* costs — that belongs
  in `confidence_estimator.py`'s harness, not on the router's hot path. If
  self-report turns out badly calibrated on the real device, `hybrid` is the
  documented fallback and porting it is a `signals/confidence.py` change only.
- **`signals/confidence.py` does not replace `difficulty.py`.** The heuristic
  is still the signal for every brain with `reports_confidence = False`, and
  the fallback when a self-rating brain returns an unparseable confidence.

Still-open tidying: `src/phone_brain/.gitignore` (genie bundles, etc.) could
fold into the root `.gitignore`, and `L_INTERFACE_CONTRACT.md` is arguably
cross-cutting enough to promote into `docs/` now that the router depends on
it. Neither blocks anything.
