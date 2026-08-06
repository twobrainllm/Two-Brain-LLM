# Two-Brain LLM -- Privacy & Power-Aware Orchestrator

**Archetype:** Privacy-Aware Edge<->Cloud Inference Routing -- decide per-request
*where* a workload runs (phone / PC / Cloud AI 100) and *what* is safe to send
(route, mask, compress).

**Use case:** a local "fast brain" answers what it can and escalates only
hard queries to a Cloud AI 100 "deep brain"; PII is masked and context is
compressed before anything crosses the device boundary.

**Target tiers:** Mobile (Snapdragon 8 Elite, 1B model) · AI PC (Snapdragon X
Elite, 3B model, this machine) · Cloud AI 100 (large model, escalation-only).

**Docs:** [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — step-by-step
implementation walkthrough, a worked trace of a routed query, and prioritized
next steps · [`docs/GAPS.md`](docs/GAPS.md) — the five gaps with verbatim error
strings · [`CLAUDE.md`](CLAUDE.md) — working agreement for agents.

## Workflow order actually driven

`hardware_detect -> convert_model -> profile_workload -> orchestrate_workload`
-- the repo's own canonical pattern (`samples/sample_projects.md`), chosen
over the two conflicting orders in the original spec because it's the only
one where each tool's output can actually feed the next (`profile`/
`orchestrate` need a `model_path` that only `convert_model` produces).

## Real vs. mocked, tier by tier

| Tool | Mobile (1B) | AI PC (3B) | Cloud AI 100 (large) |
|---|---|---|---|
| `hardware_detect` | **real** -- direct `adb shell` probe of a Galaxy S25 Ultra (see below) | **real** -- `quad-client detect --json`, Snapdragon X Elite X1E80100, Hexagon v73 @ 45 TOPS | **mocked** -- no cloud platform value exists in the tool schema at all |
| `convert_model` | mocked | mocked (QUAD's own tool -- still blocked, see below) | mocked |
| `profile_workload` | mocked | **real** -- see note below | mocked |
| `orchestrate_workload` | mocked | **real** -- see note below | n/a (cloud tier isn't an on-device op-placement problem) |

`convert_model`/`profile_workload`/`orchestrate_workload` are mocked for
mobile and cloud. The hosted server's `qairt-converter` cannot even import
(missing `libpython3.10.so.1.0`) -- a real, reproducible server-side defect.
Bypassing the server and running the SDK's own Windows-native converter
*locally* got much further (four real environment bugs fixed, real ops
actually transformed) before hitting a second, independent native-code
defect: non-deterministic uninitialized-memory reads in
`ReshapeOp::calculateShape`, proven by three identical runs producing three
different garbage totals. Two different execution paths, two different
genuine compiler-level defects -- see `docs/GAPS.md` #3/#3b and
`data/convert_model/_real_attempts_log.md` (attempts 1-5) for the full
trail. `profile_workload` and `orchestrate_workload` were still called for
real against a placeholder artifact to prove the tools themselves work; see
the `_real_*_log.md` files under `data/` for the exact calls, arguments,
and error strings. Nothing in this project's `data/` folder is a guess with
no receipt -- every mocked file states which real attempt(s) preceded it
and why they failed.

**AI PC `profile_workload`/`orchestrate_workload` -- real, but not from
QUAD's own tool call.** `convert_model` (QUAD's tool) is still genuinely
blocked -- the table entry above stays "mocked" for it, honestly. But
`data/profile_workload/pc_3b.json` and
`data/orchestrate_workload/pc_3b.json` are real captures (`_mock: false`)
from a *different* toolchain that routes around QUAD's compiler entirely:
Qualcomm's own pre-built Genie/QNN artifact for Phi-3.5-mini-instruct,
downloaded from Hugging Face and run for real on this machine's Hexagon
NPU. See [`superpowers/deploy-local-brain-npu.md`](superpowers/deploy-local-brain-npu.md)
for the full workflow and `data/npu_model/phi-3.5-mini-instruct/` for the
receipts -- real per-token latency (~94.5 ms/token, ~2.1x slower than the
old mock's guess), real `QnnGraph_execute` HTP-execution evidence, and a
real QNN profiler capture.

**Mobile hardware_detect:** `adb` was missing entirely (installed Android
SDK platform-tools mid-session after winget's own package failed a hash
check). The attached Samsung Galaxy S25 Ultra needed USB debugging enabled
and the on-device authorization prompt accepted -- both require a physical
tap on the phone, done by the user mid-session. Once authorized,
`quad-client detect --platform android --json` reached the device
(`discovery_source: client-local-static`) but returned placeholder values
(`chipset: "unknown"`, `storage_gb: 475.6` -- this PC's own disk size, not
the phone's) -- a real bug in the client's android probe path, separate
from the MCP server's own self-detect issue (#1). Worked around with
direct `adb shell getprop`/`/proc/meminfo` queries, so
`data/hardware_detect/mobile.json` is real end to end (Snapdragon 8 Elite
for Galaxy / SM8750, 8 cores, ~10.9 GB RAM, Android 16) -- see
`docs/GAPS.md` #5 for the full trail.

## Gaps

See [`docs/GAPS.md`](docs/GAPS.md) for the full writeup. Summary:

1. **`hardware_detect` self-detects the server's own container**, not the
   client device, when called directly over MCP (`quad-client detect`
   works around this with a local static probe).
2. **G8** -- the real `quad.privacy` PII guardrail (`[G8 - DELIVERED]` in
   QUAD's private core repo) isn't present in this checkout.
   `src/two_brain_router/privacy/`
   mocks its detect/mask/rehydrate contract so the router logic is real and
   testable; swap it in once this environment has access to that component.
3. **`convert_model`'s compiler is broken on two independent execution
   paths.** Hosted server: `qairt-converter` fails to import (missing
   shared library). Local Windows QAIRT install (bypassing the server
   entirely): got past that and several other bugs, but hit a
   non-deterministic uninitialized-memory read in the native shape-inference
   code. Five real attempts total, logged in
   `data/convert_model/_real_attempts_log.md`.
4. **Cloud AI 100 has no plumbing anywhere in QUAD-Client-main** -- no
   platform/SDK/device value maps to it in any tool schema.

## What's real code (not mocked)

Everything under `src/two_brain_router/` is fully working Python -- only the
*model artifacts and hardware probes it consumes* are mocked (via `data/`),
not the routing/masking logic itself:

- **`privacy.PIIGuard`** -- regex-based detect/mask/rehydrate with a
  masked-token invariant check (`assert_masked_token_invariant`): no raw
  entity value may survive in masked text, and `rehydrate(mask(x)) == x`
  exactly. Mock stand-in for gap G8 (see above).
- **`routing.TwoBrainRouter`** -- masks PII *before* any routing decision is
  made, scores query difficulty, and either answers locally or escalates:
  compresses + masks context, calls the (stubbed) deep brain with masked
  text only, then rehydrates PII in the final answer before it reaches the
  user. Routing thresholds consume the real `profile_workload` response
  shape (latency/power/token-cost) from `data/`.

Run the demo:

```powershell
.\run.ps1                                     # one-time venv + deps (installs the package with -e .)
.venv\Scripts\python.exe -m two_brain_router  # 3 example queries: easy/local, hard/cloud, PII/cloud
.venv\Scripts\python.exe -m pytest tests/ -q  # masking invariant + routing behavior
```

Other entry points:

```powershell
.venv\Scripts\python.exe -m two_brain_router --tier mobile          # route as the 1B mobile tier
.venv\Scripts\python.exe -m two_brain_router --query "..." --json   # one query, machine-readable
```

Sample output (query 3 shows the privacy guarantee end-to-end -- masked
text is what actually leaves the device, the final answer is rehydrated):

```
> My email is jane.doe@example.com and my phone is 555-123-4567 -- can you draft a reply telling the sender their SSN 123-45-6789 was found in an old backup and needs to be rotated?
  routed to: cloud | difficulty=0.40 | est_latency_ms=1159 | est_cost_usd=0.11520
  - masked 3 PII entities before any routing decision
  - escalating: difficulty=0.40 (threshold 0.55) or local_latency_est=6190ms > budget 3000ms
  - sent off-device (masked): 'My email is [PII_EMAIL_1] and my phone is [PII_PHONE_1] -- can you draft a reply telling the sender their SSN [PII_SSN_1] was found in an old backup and needs to be rotated?'
  answer: [cloud:ai100 mock deep-brain response to: 'My email is jane.doe@example.com and my phone is 555-123-4567 -- can you draft a reply telling the sender their SSN 123-45-6789 was found in an old backup and needs to be rotated?' | context_used='']
```

## Directory layout

**All source code lives under `src/two_brain_router/`** -- a `src/` layout,
matching the parent repo's own `src/quad_mcp_client/`. Each subpackage is one
replaceable seam, so a mock can be swapped for the real thing without touching
anything above it.

```
two_brain_privacy_router/
  pyproject.toml            # package metadata; `-e .` puts src/ on the path
  requirements.txt
  run.ps1                   # generic uv venv bootstrap (repo convention)
  src/two_brain_router/
    __init__.py             # public API re-exports + package map
    __main__.py             # `python -m two_brain_router`
    cli.py                  # arg parsing + demo output
    privacy/                # SEAM: swap for quad.privacy once G8 lands
      patterns.py           #   regex table -- add entity types here
      guard.py              #   PIIGuard.detect/mask/rehydrate + invariant
    signals/                # what the four QUAD tools tell the router
      loader.py             #   TierSignals: reads data/<tool>/<name>.json
      difficulty.py         #   SEAM: swap for a real logprob/entropy signal
    routing/
      policy.py             #   RoutePolicy (thresholds, compression) + RouteDecision
      brains.py             #   SEAM: NpuFastBrain (real, pc_3b) / LocalFastBrain (mobile stub) / CloudDeepBrain (stub)
      router.py             #   TwoBrainRouter: mask -> decide -> answer -> rehydrate
  tests/
    conftest.py             # puts src/ on sys.path (no install needed)
    test_privacy.py         # masking invariant
    test_routing.py         # routing behavior + policy
  data/                     # captured tool responses -- real and mocked, each logged
    hardware_detect/{ai_pc,mobile,cloud_ai100}.json
    convert_model/{mobile_1b,pc_3b,cloud_large}.json + _real_attempts_log.md
    profile_workload/{mobile_1b,pc_3b,cloud_large}.json + _real_call_log.md
    orchestrate_workload/{mobile_1b,pc_3b}.json + _real_call_log.md
    npu_model/phi-3.5-mini-instruct/ # real Genie/QNN artifact (gitignored) + receipts, see superpowers/deploy-local-brain-npu.md
  docs/GAPS.md
```

### Where new code goes

| You're adding... | Put it in |
|---|---|
| A new PII entity type | `src/two_brain_router/privacy/patterns.py` |
| The real `quad.privacy` guardrail | replace `privacy/guard.py`; keep the `PIIGuard` contract |
| A real fast/deep brain | a new `Brain` implementation in `routing/brains.py` -- `NpuFastBrain` (pc_3b, real) didn't need `convert_model`; see `superpowers/deploy-local-brain-npu.md` |
| A different escalation rule | `routing/policy.py` -- `should_escalate` is pure and unit-tested |
| A new tier (e.g. a second PC SKU) | a `data/<tool>/<name>.json` set + an entry in `routing/router.py`'s `_TIER_FILES` |
| A new tool response to consume | `signals/loader.py` |
| A new CLI flag | `cli.py` |

`data/` is pointed at by `signals/loader.DATA_DIR`; override it with the
`TWO_BRAIN_DATA_DIR` env var to run against a different capture set (e.g. real
responses recorded once the server-side blockers are fixed).

### Version control

This sample is its own git repository, rooted at this directory -- the
surrounding `QUAD-Client-main/` is a GitHub ZIP download with no `.git`, so
nothing tracks it from above. If `QUAD-Client-main` is later turned into a
proper clone (see the workspace `CLAUDE.md`), fold this in as a subtree or
submodule rather than leaving a nested `.git` inside a tracked tree.

Per the workspace convention, commit messages here carry **no AI-assistant
attribution trailer or footer**.

## What's still needed to go further

Summarized here; the prioritized version with owners, dependencies, and
done-when criteria is in
[`docs/WALKTHROUGH.md` § Next steps](docs/WALKTHROUGH.md#next-steps).

- **The PC tier now has a real fast brain (`NpuFastBrain`)** -- built by
  routing around `convert_model` entirely (Qualcomm's pre-built Genie/QNN
  artifact instead of a self-compiled one); see
  `superpowers/deploy-local-brain-npu.md`. `convert_model` itself is still
  broken.
- **A fixed QAIRT `ReshapeOp::calculateShape`** (the uninitialized-memory
  bug found locally, gap #3b) or a fixed hosted-server install (missing
  `libpython3.10.so.1.0`, gap #3) -- either would unblock real
  `convert_model` for the Mobile tier -> real `profile_workload`/
  `orchestrate_workload` numbers -> a real `LocalFastBrain.answer` for
  mobile, the one tier that's still a labeled stub. Both are reported with
  exact repro steps; neither is fixable from this client.
- **Real calibration data (representative prompts) for INT4 static QDQ
  quantization** would sharpen the Mobile-tier conversion once a compiler
  path works -- not attempted here since the compiler blockers (#3/#3b
  above) made it moot.
- **A fix to `quad_mcp_client`'s android hardware-detection path** (gap #5b)
  -- it currently returns placeholder values even when adb is authorized
  and reachable; this project's real mobile hardware data came from direct
  `adb shell` queries instead.
