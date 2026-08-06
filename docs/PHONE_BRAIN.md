# Phone Brain (L) — What's Been Built

Summary of the work in `src/phone_brain/` (branch `local_brain`, commit
`e7130ce`), what it decides, what it verifies, and how it connects to the
router in `src/two_brain_router/`.

**In one line:** the Mobile tier's fast brain, built end-to-end — the real
thing that `routing/brains.py::LocalFastBrain` is currently a labeled stub
for — plus a mock server so orchestrator work isn't blocked on hardware.

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

### Plus: benchmark harness — `test_phone_brain.py`

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
| **Verified working** | Everything on the mock path — mock server, all three confidence strategies, the benchmark harness against the mock |
| **Written, not yet run** | The export (needs AI Hub token + gated Llama access), device push/serve, and every real-device number |
| **Explicitly open** | Shared vs. split weights for the orchestrator — does O reuse this same L with a routing-only system prompt, or get its own smaller model? Recommended default: reuse L with a small `max_new_tokens` |

The Track A / Track B split (needs the phone / needs nothing) is the same
instinct as this project's `data/` fixtures: keep the dependent work
unblocked while the hardware path is still in progress.

---

## How it connects to `two_brain_router`

It fills **two** of our seams, one of which is
[next step #4](WALKTHROUGH.md#next-steps) — the biggest quality win available:

| Our seam | What phone_brain provides |
|---|---|
| `routing/brains.py::LocalFastBrain` | A real HTTP-backed brain for the Mobile tier |
| `signals/difficulty.py::DifficultyEstimator` | `confidence_estimator.py`, inverted — `score = 1 - confidence` |
| `data/profile_workload/mobile_1b.json` (mocked) | `test_phone_brain.py` produces exactly the real numbers this mock stands in for |

Vocabulary maps directly: **L** = fast brain, **O** = router, **C** = deep brain.

### Five things to reconcile before merging

**1. Privacy ordering — the one that matters.** Our invariant is
mask-before-any-routing-decision. Self-reported confidence sends the **raw
prompt** to L. On-device that's defensible (nothing leaves the phone), but
`--base-url` is a plain argument — point it at a non-local host and raw PII
goes off-device silently. Needs a hard localhost/on-device assertion on that
call, or masked text passed in.

**2. Self-report merges two steps of `route()`.** The answer and the confidence
score arrive in the *same* call. Our router currently scores difficulty and
*then* calls the brain; here escalation means discarding an answer already paid
for. That's a structural change to `route()`, not a drop-in.

**3. The threshold lives in two places.** `confidence_estimator.py` returns
`should_escalate` using its own `confidence_threshold=0.5`; our
`RoutePolicy.escalate_threshold` is `0.55`. Their own contract says O owns the
decision — so consume `confidence` and ignore `should_escalate`, or the two
drift apart.

**4. Model mismatch with our fixtures.** They're on Llama-3.2-3B w4a16; our
`data/*/mobile_1b.json` says Qwen2.5-1.5B int4 (ttft 180 ms, 27.5 ms/token).
Those need real numbers from `test_phone_brain.py`, with a receipt per the
`CLAUDE.md` rule.

**5. `test_phone_brain.py` is not a pytest test.** The name matches
`python_files = ["test_*.py"]` in `pyproject.toml`. If it ever lands under
`testpaths`, pytest will collect it and it will fail or hang waiting on a live
endpoint. Rename to `bench_phone_brain.py` when absorbing.

---

## Where this code should live

**Superseded by what actually merged — read this before doing the split
below.** This section originally proposed breaking `routing/brains.py` into
a `routing/brains/` subpackage (`base.py`/`stub.py`/`openai_http.py`). Since
then, the AI-PC tier's real brain (`NpuFastBrain`, see `../CLAUDE.md`'s
Branch state section) merged into `main` and **did not do that split** — it
added a fourth class (`NpuFastBrain`, alongside `Brain`, `BrainResponse`,
`LocalFastBrain`, `CloudDeepBrain`) straight into the existing flat
`brains.py`, imported `onnxruntime_qnn` lazily inside a method so the module
stays importable without it, and registered itself via a tier/env-var switch
in `router.py::_build_fast_brain` (`TWO_BRAIN_NPU_BRAIN=1`). That's now the
working, tested precedent for "add a real brain without breaking the base
package" — follow it instead of introducing a subpackage split the codebase
has already diverged from:

```
src/two_brain_router/
  routing/brains.py   # stays flat -- add OpenAIHttpBrain here, next to
                      # NpuFastBrain, LocalFastBrain, CloudDeepBrain.
                      # Import urllib/requests lazily inside the method,
                      # same as NpuFastBrain's lazy `import onnxruntime_qnn`.
  routing/router.py   # _build_fast_brain grows a phone branch, gated behind
                      # its own env var (e.g. TWO_BRAIN_PHONE_BRAIN=1),
                      # same shape as the existing NPU-tier switch.
  signals/
    confidence.py     # <- absorbs the three strategies; replaces difficulty.py

tools/phone/          # per-device by nature, outside the package
  export_phone_brain.sh
  mock_phone_brain_server.py
  bench_phone_brain.py            # renamed from test_phone_brain.py (see #5)
  verify_confidence_estimator.sh

docs/
  L_INTERFACE_CONTRACT.md         # promote -- it's cross-cutting, not phone-only
  phone-deployment.md             # the deployment guides, deduplicated
```

The original point still holds even without the subpackage: `OpenAIHttpBrain`
should be **one** class that serves the phone today and any other
OpenAI-shaped served model later — including a future hosted AI-PC or cloud
endpoint — not a phone-specific class. Just implement it as an addition to
the existing `brains.py`, not a new file tree, so the two real-brain efforts
don't leave the module split two different ways.

**Housekeeping:** `PHONE_DEPLOYMENT_GUIDE.md` and
`PHONE_DEPLOYMENT_GUIDE_final.md` are byte-identical duplicates. Keep one.
`src/phone_brain/.gitignore` (genie bundles, etc.) should fold into the root
`.gitignore` when the folder moves.
