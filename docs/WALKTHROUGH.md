# Implementation Walkthrough

How this sample was actually built, in the order it was built, and what each
step produced. Written to be read top-to-bottom by someone picking the project
up cold.

Companion documents:

- [`../README.md`](../README.md) — what the project is, how to run it
- [`GAPS.md`](GAPS.md) — the five gaps, with verbatim error strings
- [`../CLAUDE.md`](../CLAUDE.md) — working agreement (invariants, receipts rule)

**Ground rule that shaped everything below:** a mocked artifact is only
acceptable with a receipt — the real call that was attempted, its arguments,
and the verbatim failure. Every mock in `data/` carries one. Nothing here is a
plausible-looking number invented to make a demo run.

---

## Step 0 — Picking the workflow order

The spec this project came from described the tool sequence two different ways.
Neither was usable as written: `profile_workload` and `orchestrate_workload`
both require a `model_path`, and the only thing that produces one is
`convert_model`. So the repo's own canonical order
(`samples/sample_projects.md`) is the only one where each tool's output can
feed the next:

```
hardware_detect  ->  convert_model  ->  profile_workload  ->  orchestrate_workload
   what am I         make an artifact     what does it cost     where do the ops go
   running on?       for this tier        to run it here?
```

Three tiers run through it: **Mobile** (Snapdragon 8 Elite, 1B), **AI PC**
(Snapdragon X Elite, 3B — this machine), **Cloud AI 100** (large,
escalation-only).

---

## Step 1 — `hardware_detect`: two real captures, one impossible

**AI PC (real).** Calling the MCP tool directly returned `AMD EPYC 7B12`,
Ubuntu 24.04, no NPU — the *hosted server's own container*, not this machine
(gap #1). `quad-client detect --json` sidesteps it with a `client-local-static`
probe, and that is what `data/hardware_detect/ai_pc.json` holds: Snapdragon X
Elite X1E80100, Hexagon v73 @ 45 TOPS, QAIRT 2.38.0.250901.

**Mobile (real, the hard way).** This one cost the most wall-clock for the
least code:

1. No `adb` on the build host at all. winget's `Google.PlatformTools` failed
   its own installer hash check; platform-tools was installed directly from
   `dl.google.com` instead.
2. Windows confirmed a phone was physically attached, but `adb devices` listed
   nothing until USB debugging was enabled and the on-device authorization
   prompt was accepted **by hand, on the phone**. That is an intentional
   Android security control with no software bypass — UI Automator runs *over*
   adb, so it cannot bootstrap the authorization it depends on.
3. With the device authorized and demonstrably reachable,
   `quad-client detect --platform android` *still* returned placeholders:
   `chipset: "unknown"`, `cpu_cores: 1`, `ram_gb: 0.1`, and `storage_gb: 475.6`
   — that last value being this PC's own disk size, which is what proved the
   android path falls through to defaults rather than probing (gap #5b, a real
   client bug worth filing upstream).

Worked around with direct queries — `adb shell getprop ro.soc.model` (SM8750),
`ro.product.model` (SM-S938U1), `ro.build.version.release` (16),
`cat /proc/meminfo` (11381324 kB). So `mobile.json` is real end-to-end
(`_mock: false`), with `npu_tops: null` because TOPS isn't exposed via
`getprop` and isn't in the client's SoC hint table for this platform.

**Cloud AI 100 (mocked, and unavoidably so).** `hardware_detect`'s platform
enum is `windows|linux|android|gateway|robotics`; `convert_model`'s
`target_sdk` enum is `qnn|snpe|nwaios|executorch`. Neither has a cloud value.
There was no call to attempt (gap #4) — the tier is modeled from Qualcomm's
public datasheet.

> **Takeaway for anyone extending this:** don't trust a `hardware_detect`
> response without checking `discovery_source`. Anything that consumes chipset
> data (including `profile_workload` and `orchestrate_workload` internally)
> inherits gap #1.

---

## Step 2 — `convert_model`: five attempts, two independent compiler defects

The full log with verbatim errors is
[`../data/convert_model/_real_attempts_log.md`](../data/convert_model/_real_attempts_log.md).
Condensed:

| # | What was tried | Result |
|---|---|---|
| 1 | HF source, `repo_id` only | Rejected — needs an exact `filename`, not a checkout |
| 2 | HF source, `repo_id` + `filename` | **`huggingface_hub` not installed on the server** — this source kind is dead on the deployment regardless of args |
| 3 | URL source, pre-quantized ONNX | **Correctly rejected**: ConvInteger/MatMulInteger dynamic quant is unsupported on Hexagon HTP, which needs static QDQ |
| 4 | URL source, self-contained FP32, let QUAD quantize | **`qairt-converter` cannot import**: `ImportError: libpython3.10.so.1.0` |
| 5 | Bypass the server, run QAIRT locally on Windows | Got much further, then hit a **second, unrelated defect** |

Attempt 3 is worth dwelling on: it is not a failure. The server downloaded the
graph, inspected it, and caught exactly the trap this archetype depends on
catching — a ConvInteger model *loads without error and silently runs on CPU*,
which would have quietly invalidated every per-tier NPU number in this project.
That's the validation layer working.

Attempt 4 is the hosted blocker: the server's own QAIRT install is missing a
shared library, so the compiler cannot start for *any* model.

Attempt 5 bypassed the server entirely using this machine's local QAIRT SDK,
which ships Windows-native converter builds — a completely different toolchain
that doesn't share the missing-`.so` problem. Four real environment bugs had to
be fixed to get there:

1. **QAIRT's Windows arch-detection is broken on this CPU.** It picks its
   native-extension folder from `platform.processor()`, which on this machine
   returns the physical CPU ID regardless of the running process architecture —
   always resolving to `windows-arm64ec`, which won't load under native ARM64
   CPython. Fixed with a launcher shim that monkeypatches `platform.processor()`
   before the import.
2. **Needed a genuinely x64 Python** (3.10.11, matching what QAIRT's own
   dependency script supports) to load the `x86_64-windows-msvc` `.pyd`s.
3. **`onnx.mapping` was removed from modern `onnx`.** The converter's import
   failed inside a bare `except:`, set `onnx = None`, and surfaced much later
   as `AttributeError: 'NoneType' object has no attribute 'AttributeProto'`.
   Fixed by pinning `onnx==1.14.1`.
4. **51 dynamic inputs needed explicit static shapes** (`input_ids`,
   `attention_mask`, `position_ids`, `past_key_values.{0..23}.{key,value}` —
   standard decoder-with-KV-cache export), supplied as 51 `-s` flags.

With all four resolved the converter began genuinely transforming ops
(`INFO_STATIC_RESHAPE: Applying static reshape to model.embed_tokens.weight`),
then failed in `ReshapeOp::calculateShape` on an `Unsqueeze` of
`attention_mask` — a 2-element tensor by construction, since Unsqueeze never
changes element count. Three back-to-back runs of the identical command
produced three different totals: `2016977728`, `-1863713320`, `1598663576`.
A negative element count and non-determinism across identical inputs is the
signature of an uninitialized-memory read, inside a closed-source compiled
`.pyd`. Not fixable from the Python or CLI layer.

**Net result:** two execution paths, two genuine compiler-level defects, no
real artifact. Everything downstream of the compiler is therefore mocked — but
mocked against a real response envelope, with `unsupported_ops` and
`quant_format: "QDQ"` reflecting the constraint attempt 3 actually taught us.

---

## Step 3–4 — `profile_workload` and `orchestrate_workload`: prove the tool, then mock the numbers

With no LLM artifact, both tools were still called for real against the one
complete artifact already sitting in `quad_artifacts/` — not to get an LLM
number, but to prove the tools themselves work and to capture their real
response shape.

`profile_workload` succeeded with every timing field zeroed, and — importantly
— **said so honestly** rather than fabricating numbers:

```json
"measurement_notes": {
  "latency": "not_measured:parser_no_match",
  "layers": "synthetic_composite:no_diagview_csv",
  "memory": "not_measured:process exited before first sample",
  "power": "estimated:host_thermal_model"
}
```

`orchestrate_workload` succeeded too, correctly forcing the whole graph to CPU
with `"The NPU/HTP does not support op types: composite."` — its
fallback/decision-reporting logic working correctly on a degenerate input.

That `measurement_notes` convention is what the mocked per-tier files copy:
every one of them carries `_mock: true` and a `_reason` naming the blocker. The
router reads only the response *shape*, so replacing a mock with a real capture
requires no code change.

---

## Step 5 — The code

All source is under `src/two_brain_router/`. The package is split along the
lines that will need replacing, not along conceptual lines.

### 5.1 The privacy guard (`privacy/`)

Stand-in for `quad.privacy` (gap G8), implementing the same
detect/mask/rehydrate contract with a regex table:

- `mask()` replaces each entity with `[PII_<TYPE>_<n>]` and returns a vault
  mapping placeholder → real value. Longest matches are replaced first so a
  card-shaped substring inside a longer match isn't double-masked.
- `rehydrate()` inverts it.
- `assert_masked_token_invariant()` enforces the two properties that matter:
  **no raw entity value survives in the masked text**, and
  **`rehydrate(mask(x)) == x` exactly**. It raises; it is not advisory.

A fresh `PIIGuard` is constructed per request, so placeholder numbering can't
collide across requests.

### 5.2 The signals (`signals/`)

`TierSignals.load()` reads the four captured tool responses for a tier.
`DATA_DIR` is overridable via `TWO_BRAIN_DATA_DIR`, so you can point the router
at real captures without overwriting the logged mocks.

`DifficultyEstimator.score()` is the seam standing in for a real confidence
signal. It scores surface features — length, hard-query markers (`derive`,
`complexity`, `step by step`, `trade-off`, …), multiple question marks —
because no compiled model exists to emit logprobs.

### 5.3 The policy (`routing/policy.py`)

Deliberately pure — no I/O, no brain calls — so escalation rules are unit
testable on their own. Two thresholds:

- `escalate_threshold = 0.55` on the difficulty score
- `local_latency_budget_ms = 3000`, checked against the tier's *own profiled*
  per-token rate

Escalation fires on **either**. That second axis is the "power-aware" half of
the archetype: a trivial query still goes to the cloud if the local tier can't
answer it in time.

### 5.4 The brains (`routing/brains.py`)

One `Brain` protocol, two implementations, both returning `BrainResponse(text,
latency_ms, cost_usd)`. Both are labeled stubs — the text says so — with
latency and cost computed from the tier's real profile envelope. `CloudDeepBrain`
adds `network_rtt_mean` and charges `token_cost_usd_per_1k`; `LocalFastBrain`
costs nothing but has a worse per-token rate.

### 5.5 The router (`routing/router.py`)

The ordering *is* the guarantee:

```
1. mask the query                     <- before any routing decision
2. assert the masked-token invariant  <- fatal if raw PII survived
3. score difficulty / estimate local latency
4. answer locally, OR mask+compress context and escalate
5. rehydrate                          <- only after the answer is back
```

Step 1 comes before step 3 on purpose: a decision made on raw text has already
read the PII. Step 4's escalation path masks the *context* too, not just the
query, then compresses it to the last 800 characters on a sentence boundary.
The vault never leaves the process.

### 5.6 Worked trace — the third demo query

```
> My email is jane.doe@example.com and my phone is 555-123-4567 -- can you
  draft a reply telling the sender their SSN 123-45-6789 was found in an old
  backup and needs to be rotated?
```

| Stage | Value | Where it comes from |
|---|---|---|
| PII masked | 3 entities (EMAIL, PHONE, SSN) | `privacy/patterns.py` |
| Difficulty | `0.40` | 178 chars → 0.4 (capped); no hard markers; single `?` |
| Local latency est. | `6190 ms` | `142 + 64 × 94.5` from `profile_workload/pc_3b.json` (real capture as of the NPU deployment workflow — see `superpowers/deploy-local-brain-npu.md`) |
| Decision | **escalate** | `0.40 < 0.55`, but `6190 > 3000` — **latency, not difficulty** |
| Sent off-device | `My email is [PII_EMAIL_1] and my phone is [PII_PHONE_1] … SSN [PII_SSN_1] …` | masked text only |
| Cloud latency | `1159 ms` | `45 rtt + 320 ttft + 64 × 12.4` from `cloud_large.json` |
| Cloud cost | `$0.11520` | `64/1000 × $1.80` |
| Answer | rehydrated to real values | on-device, after return |

Worth noting what this example demonstrates: the query escalated on the
**power/latency axis while carrying PII** — the exact case where a naive router
leaks. The masking happened first, so it didn't matter which axis fired.

---

## Step 6 — Tests

Nine tests, no third-party runtime deps:

- `test_privacy.py` — round-trip and no-op masking
- `test_routing.py` — easy stays local, hard escalates, **PII never reaches the
  cloud unmasked** (the regression test for the whole guarantee), both local
  tiers load, and the two pure-policy tests (latency-only escalation,
  compression thresholds)

`tests/conftest.py` puts `src/` on `sys.path`, so the suite runs without
installing anything.

---

## Step 7 — Packaging and repository

- `src/` layout matching the parent repo's `src/quad_mcp_client/`
- `pyproject.toml` — hatchling, editable install, `two-brain-router` console
  script; **zero runtime dependencies** (everything it consumes is captured
  JSON, so the logic runs anywhere)
- `run.ps1` — the repo's generic uv bootstrap convention, unmodified except for
  the example commands
- `git init` rooted at this sample directory (the surrounding
  `QUAD-Client-main/` is a ZIP download with no `.git`), three commits:
  original flat form → `src/` package → `CLAUDE.md`

---

## Next steps

Ordered by what unblocks the most. Items 1 and 2 are not fixable from this
client — they need someone with server or SDK access.

> **Update:** item 3 is now done for the AI-PC tier (see below) — bypassing
> item 1 rather than waiting on it. The Mobile tier's counterpart is a
> separate, unmerged, in-progress effort — see `../CLAUDE.md`'s "Branch
> state" section for the full picture across branches.

### 1. Unblock `convert_model` (P0 — blocks 3, 4, 5)

Two independent defects, both with complete repro steps already written down.
Either one being fixed unblocks a real artifact for its path.

| | Hosted server | Local Windows QAIRT |
|---|---|---|
| Defect | `qairt-converter` missing `libpython3.10.so.1.0` | non-deterministic uninitialized read in `ReshapeOp::calculateShape` |
| Fix owner | QUAD platform team (install `libpython3.10`/`python3.10-dev` in the server image) | QAIRT SDK team (native `.pyd`, closed source) |
| Repro | `_real_attempts_log.md` attempt 4 | attempt 5, incl. the three differing garbage totals |
| Also worth fixing | `huggingface_hub` not installed (kills the whole `kind: "huggingface"` source) | — |

**Action:** file both upstream with the logs as-is. **Done when:** a real
`.bin`/`.dlc` exists for `pc_3b`.

### 2. Replace the mocked `data/` files with real captures (P1 — depends on 1) — DONE for `pc_3b`

Item 1 (`convert_model`) is still genuinely blocked (see `docs/GAPS.md`
#3/#3b) — but `pc_3b` didn't end up depending on it. See item 3: a
different, real toolchain produced a real artifact, and
`data/profile_workload/pc_3b.json` / `data/orchestrate_workload/pc_3b.json`
are now real captures (`_mock: false`) from it, diffed against the old
mocked values in each directory's `_real_call_log.md`. `mobile_1b.json` and
`cloud_large.json` remain mocked — still genuinely blocked by 1 (mobile) and
gap #4 (cloud), out of scope for the NPU workflow below.

### 3. Run a real fast brain (P1 — depends on 1) — DONE, via a different path than planned

This item assumed an artifact from `convert_model`, which is still blocked.
Instead of waiting on item 1,
[`superpowers/deploy-local-brain-npu.md`](../superpowers/deploy-local-brain-npu.md)
routes around QUAD's compiler entirely — Qualcomm's own pre-built
Genie/QNN context binary for Phi-3.5-mini-instruct, run in-process via
`ctypes` against `Genie.dll`. `NpuFastBrain` in `routing/brains.py`
implements the `Brain` protocol for real, real inference runs on this
machine's Hexagon NPU (confirmed via `QnnGraph_execute` logs and a real
QNN profiler capture — see that workflow doc's Phase 3/6), and it's wired
into `TwoBrainRouter` behind the `TWO_BRAIN_NPU_BRAIN=1` env var so the
base package stays stdlib-only by default. Nothing above `brains.py`
changed — the seam held. `tests/test_npu_brain.py` has the Phase 6
verification suite (skips without the real runtime/artifact, passes for
real under `.venv-npu`).

**The Mobile tier is closed too, on branch `js/orchestrator`.**
`PhoneFastBrain` wires `src/phone_brain/`'s Genie/QNN server for
Llama-3.2-3B-Instruct on a Galaxy S25 in behind the same `Brain` seam, over
the OpenAI-shaped contract in `src/phone_brain/L_INTERFACE_CONTRACT.md`.
Enabled with `TWO_BRAIN_PHONE_BRAIN=1`; testable against the mock server with
no phone attached. See [`ORCHESTRATOR.md`](ORCHESTRATOR.md) and
[`PHONE_BRAIN.md`](PHONE_BRAIN.md).

What's still mocked for mobile is the *fixture*, not the brain:
`data/profile_workload/mobile_1b.json` still describes Qwen2.5-1.5B int4
rather than the Llama-3.2-3B w4a16 that actually runs, so the latency budget
pre-check reasons about the wrong model until a real `bench_phone_brain.py`
run against an S25 replaces it.

### 4. Replace the difficulty heuristic with a real confidence signal (P2 — depends on 3) — DONE for both real tiers

Once the fast brain runs, `DifficultyEstimator.score()` becomes obsolete:
escalate on the fast brain's own logprob entropy or margin instead of on the
presence of the word "derive". Keep the `score(query) -> float` signature so
`policy.py` is untouched. This is the single biggest quality improvement
available — surface features misfire in both directions (the demo's query 3
scores 0.40 for a genuinely easy request).

**Done for the mobile tier** via `signals/confidence.py` — not logprobs
(unavailable, and `L_INTERFACE_CONTRACT.md` says explicitly not to depend on
them) but the model's own self-reported confidence, inverted into the same
difficulty scale so `policy.py` really is untouched, exactly as this item
asked. The predicted misfire is now observable: with a real signal, demo
query 3 scores **0.08** and stays on-device, where the heuristic escalated it.

Two follow-ups this opened. The first is now closed; the second got *worse* news
than expected:

- ~~**The AI PC tier still uses the heuristic.**~~ **DONE.** `NpuFastBrain` now
  self-rates too. It cost exactly what was predicted — a prompt change plus a
  re-profile, no router change (`route()` and `policy.py` untouched; only
  `brains.py` and `data/profile_workload/pc_3b.json` moved). Both real brains
  are now Shape B, so `DifficultyEstimator` is the signal only for the stub
  brains, i.e. the default stdlib-only path. Receipts, including three real
  defects found and fixed on the way:
  `data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md` Attempt 5.
- **Calibration is unverified on real hardware** — now **partly verified, and
  the answer is not encouraging.** On the AI PC tier, across 8 real queries the
  self-report came back 0.85–1.00 with one unparseable: a merge-sort derivation
  rated itself the same 0.95 as "What is the capital of France?". Inverted, that
  is difficulty 0.00–0.15, all far under the 0.55 threshold — so this tier's
  escalations are in practice decided by the latency budget pre-check, not by
  the confidence. The signal reliably separates "produced a number" from
  "didn't"; it does not yet separate easy from hard, which is most of what this
  item was after. The threshold was deliberately not retuned to compensate.
  `confidence_estimator.py`'s `hybrid()` remains the documented fallback and is
  still a `signals/confidence.py`-only change. The **mobile** tier's calibration
  is still unverified — that needs the real S25.

### 5. Compress context with the fast brain instead of truncating (P2 — depends on 3)

`policy.compress_context()` currently keeps the last 800 characters on a
sentence boundary. The intended design is to have the fast brain *summarize*
before escalation — cheaper on cloud tokens and much less lossy. Blocked on the
same missing local model.

### 6. Swap in the real `quad.privacy` when G8 is reachable (P2 — independent)

Replace `privacy/guard.py`, keep the contract, re-run
`test_escalated_pii_never_reaches_cloud_unmasked` **unmodified**. If it needs
editing to pass, the guarantee changed and that needs review, not a test fix.
Until then, the regex detector's coverage is the honest limit here: it will
miss names, addresses, and account numbers. Widening `patterns.py` is cheap and
worth doing regardless.

### 7. Give the Cloud AI 100 tier real plumbing (P3 — independent)

Nothing in any tool schema maps to it (gap #4). Two options, in order of
preference: (a) file a schema request for a cloud platform/target value, or
(b) implement `CloudDeepBrain` against a real serving endpoint directly,
bypassing the tool layer — the `Brain` protocol already permits this, and it
would make the escalation path genuinely end-to-end even while the local tier
stays stubbed.

### 8. Fix `quad-client detect --platform android` (P3 — independent)

Gap #5b is a real bug with a clean repro: authorized device, reachable, still
returns `chipset: "unknown"` and the *host's* disk size. Small upstream fix in
`quad_mcp_client`'s android path; it would remove the `adb shell` workaround
this project needed.

### 9. Additions worth having once the above lands

- **An audit log of what actually crossed the boundary** — masked text, entity
  counts, tier, cost — per request. This is the artifact a privacy review would
  ask for, and the data already exists in `RouteDecision.notes`.
- **A power/cost budget policy** — escalate against a rolling spend or energy
  budget, not just per-request latency. `power_mw` and
  `energy_per_1k_tokens_mj` are already in the profile fixtures and currently
  unused.
- **Streaming** — `BrainResponse` is a single blob today; TTFT is already
  profiled separately from per-token rate, so the data supports it.
- **Real INT4 calibration data** for the Mobile tier's static QDQ quantization
  — not attempted here because the compiler blockers made it moot.
