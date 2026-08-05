# Two-Brain LLM -- Privacy & Power-Aware Orchestrator

**Archetype:** Privacy-Aware Edge<->Cloud Inference Routing -- decide per-request
*where* a workload runs (phone / PC / Cloud AI 100) and *what* is safe to send
(route, mask, compress).

**Use case:** a local "fast brain" answers what it can and escalates only
hard queries to a Cloud AI 100 "deep brain"; PII is masked and context is
compressed before anything crosses the device boundary.

**Target tiers:** Mobile (Snapdragon 8 Elite, 1B model) · AI PC (Snapdragon X
Elite, 3B model, this machine) · Cloud AI 100 (large model, escalation-only).

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
| `convert_model` | mocked | mocked | mocked |
| `profile_workload` | mocked | mocked | mocked |
| `orchestrate_workload` | mocked | mocked | n/a (cloud tier isn't an on-device op-placement problem) |

`convert_model`/`profile_workload`/`orchestrate_workload` are mocked for
**every** tier. The hosted server's `qairt-converter` cannot even import
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
   QUAD's private core repo) isn't present in this checkout. `privacy_mask.py`
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

`privacy_mask.py` and `router.py` are fully working Python -- only the
*model artifacts and hardware probes they consume* are mocked (via
`data/`), not the routing/masking logic itself:

- **`privacy_mask.PIIGuard`** -- regex-based detect/mask/rehydrate with a
  masked-token invariant check (`assert_masked_token_invariant`): no raw
  entity value may survive in masked text, and `rehydrate(mask(x)) == x`
  exactly. Mock stand-in for gap G8 (see above).
- **`router.TwoBrainRouter`** -- masks PII *before* any routing decision is
  made, scores query difficulty, and either answers locally or escalates:
  compresses + masks context, calls the (stubbed) deep brain with masked
  text only, then rehydrates PII in the final answer before it reaches the
  user. Routing thresholds consume the real `profile_workload` response
  shape (latency/power/token-cost) from `data/`.

Run the demo:

```powershell
.\run.ps1                                    # one-time venv + deps
.venv\Scripts\python.exe router.py            # 3 example queries: easy/local, hard/cloud, PII/cloud
.venv\Scripts\python.exe -m pytest tests/ -q  # masking invariant + routing behavior
```

Sample output (query 3 shows the privacy guarantee end-to-end -- masked
text is what actually leaves the device, the final answer is rehydrated):

```
> My email is jane.doe@example.com and my phone is 555-123-4567 -- can you draft a reply telling the sender their SSN 123-45-6789 was found in an old backup and needs to be rotated?
  routed to: cloud | difficulty=0.40 | est_latency_ms=1159 | est_cost_usd=0.11520
  - masked 3 PII entities before any routing decision
  - escalating: difficulty=0.40 (threshold 0.55) or local_latency_est=3007ms > budget 3000ms
  - sent off-device (masked): 'My email is [PII_EMAIL_1] and my phone is [PII_PHONE_1] -- can you draft a reply telling the sender their SSN [PII_SSN_1] was found in an old backup and needs to be rotated?'
  answer: [cloud:ai100 mock deep-brain response to: 'My email is jane.doe@example.com and my phone is 555-123-4567 -- can you draft a reply telling the sender their SSN 123-45-6789 was found in an old backup and needs to be rotated?' | context_used='']
```

## Directory layout

```
two_brain_privacy_router/
  privacy_mask.py           # G8 mock: PII detect/mask/rehydrate + invariant
  router.py                 # Two-Brain orchestrator (fast brain / deep brain / routing policy)
  requirements.txt
  run.ps1
  tests/test_router.py
  data/
    hardware_detect/{ai_pc,mobile,cloud_ai100}.json
    convert_model/{mobile_1b,pc_3b,cloud_large}.json + _real_attempts_log.md
    profile_workload/{mobile_1b,pc_3b,cloud_large}.json + _real_call_log.md
    orchestrate_workload/{mobile_1b,pc_3b}.json + _real_call_log.md
  docs/GAPS.md
```

## What's still needed to go further

- **A fixed QAIRT `ReshapeOp::calculateShape`** (the uninitialized-memory
  bug found locally, gap #3b) or a fixed hosted-server install (missing
  `libpython3.10.so.1.0`, gap #3) -- either would unblock real
  `convert_model` for the Mobile/PC tiers -> real
  `profile_workload`/`orchestrate_workload` numbers -> a real
  `_local_answer` in `router.py` instead of a labeled stub. Both are
  reported with exact repro steps; neither is fixable from this client.
- **Real calibration data (representative prompts) for INT4 static QDQ
  quantization** would sharpen the Mobile-tier conversion once a compiler
  path works -- not attempted here since the compiler blockers (#3/#3b
  above) made it moot.
- **A fix to `quad_mcp_client`'s android hardware-detection path** (gap #5b)
  -- it currently returns placeholder values even when adb is authorized
  and reachable; this project's real mobile hardware data came from direct
  `adb shell` queries instead.
