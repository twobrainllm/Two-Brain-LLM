# Two-Brain LLM — Privacy & Power-Aware Orchestrator

A chat application that decides, **per request**, whether your question can be
answered by a model running on your own device or whether part of it has to go
to the cloud — and if any of it does leave, masks the personal information out
of it first.

A small "fast brain" runs locally (Phi-3.5-mini on the Snapdragon X Elite's
Hexagon NPU, a Qwen3-VL GGUF on the Adreno GPU, or Llama-3.2-3B on a Galaxy
S25). It answers what it can and **names the part it cannot**. Only that named
gap is masked, compressed, and sent to a Cloud AI 100 "deep brain" — so a
question about your own email address is usually answered without your email
address ever leaving the machine, and when something must cross, you can see
exactly what did.

The routing and masking logic is **real, working Python**. What is mocked is
model artifacts and hardware probes for the tiers whose toolchain is broken —
each one labeled, with the failed real attempt logged beside it. See
[Real vs. mocked](#real-vs-mocked-tier-by-tier).

**Archetype:** Privacy-Aware Edge↔Cloud Inference Routing — decide per-request
*where* a workload runs (phone / PC / Cloud AI 100) and *what* is safe to send
(route, mask, compress).

**Target tiers:** Mobile (Snapdragon 8 Elite, 3B model) · AI PC (Snapdragon X
Elite, 3B–8B models, this machine) · Cloud AI 100 (large model,
escalation-only).

---

## Team

Fill in — these are placeholders.

| Name | Email | Area |
|---|---|---|
| Jaisurya | `jaisuryasundar2001@gmail.com` |
| Thrisha Ambareesharaje Urs Urs | `thrishaaurs@gmail.com` |
| Nikhita Neelakanta | `nikhita.neelakanta@gmail.com` |
| Vishnu Teja Kunde| `kvishnutez@gmail.com` | 

---

## How it decides

Three routing shapes, depending on what the tier's fast brain can tell the
router. Full explanation in [`docs/ORCHESTRATOR.md`](docs/ORCHESTRATOR.md).

| Shape | The brain reports | The decision |
|---|---|---|
| **A** | nothing | a local keyword/length heuristic scores the query |
| **B** | its own confidence | escalate when confidence ≤ 0.90 |
| **C** | confidence **and** a named gap | keep the local answer, send **only the gap** to the cloud |

Shape C is what both AI-PC brains use. A named gap escalates on its own
whatever the confidence says — deliberately, because this tier's confidence
number is measurably uninformative (0.95–1.00 on nearly everything) while the
gap field discriminates cleanly.

### The privacy guarantee

Four invariants, all living in the *ordering* inside
`routing/router.py::TwoBrainRouter.route`:

1. **Mask at the boundary, not at the front door.** Brains that run on this
   machine (`NpuFastBrain` in-process, `GpuLocalBrain` in a loopback child)
   receive the query **exactly as typed** — nothing they are given is
   transmitted, and masking them would cost answer quality while protecting
   nothing. Everything else — both cloud brains, and `PhoneFastBrain`, which
   runs on a physically separate device — gets masked text only.
2. **The invariant check is fatal.** `assert_masked_token_invariant` runs on
   everything that crosses and raises. Never a warning.
3. **Only masked text crosses** — including text the local model wrote itself.
   On a Shape C split the model was handed the *raw* query, so its partial
   answer and its gap can quote a real address verbatim; both are masked at the
   crossing.
4. **Rehydrate last, on-device**, only once the answer is back.

---

## Setup

**Prerequisites:** Windows on ARM (Snapdragon X Elite) for the local brains,
[`uv`](https://docs.astral.sh/uv/), and Python 3.10–3.12. The base package is
**stdlib-only** — everything below runs with no hardware and no model weights
until you opt in.

```powershell
cd samples\two_brain_privacy_router
.\run.ps1        # one-time: creates .venv and installs this package with -e .
```

Verify:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q   # router, masking, orchestration
node ui\selftest.mjs                           # UI logic, zero dependencies
```

### Optional: the real brains

All are **off by default**. Turn on only what you have hardware for.

| Want | Set | Also needs |
|---|---|---|
| AI PC on the **NPU** | `$env:TWO_BRAIN_NPU_BRAIN=1` | the Genie/QNN bundle under `data/npu_model/` |
| AI PC on the **GPU** | `$env:TWO_BRAIN_GPU_BRAIN=1` | a GGUF + `llama-server` (see below) |
| The **real cloud** brain | `$env:TWO_BRAIN_CLOUD_BRAIN=1` | `INFERENCE_CLOUD_API_KEY`, `INFERENCE_CLOUD_ENDPOINT` |
| The **phone** as the mobile tier | `$env:TWO_BRAIN_PHONE_BRAIN=1` | a served S25, or the mock server |

Model weights are gitignored and large, so they usually live in one place and
several checkouts point at it:

```powershell
$env:TWO_BRAIN_MODEL_ROOT="C:\path\to\models"   # laid out like this repo's data/
```

`GpuLocalBrain` also wants `TWO_BRAIN_LLAMA_BIN` (the `llama-server` build) and,
for vision models, `TWO_BRAIN_GPU_MMPROJ`. The complete env-var table is in
[`docs/ORCHESTRATOR.md`](docs/ORCHESTRATOR.md).

---

## Running

### The chat app (server + UI)

Two processes. **Use two PowerShell windows** — the first one stays in the
foreground printing the trace.

```powershell
# Window 1 — the router API on 127.0.0.1:8765
.\.venv\Scripts\python.exe -m two_brain_router.api --tier pc

# Window 2 — static file server for the UI
cd ui
python -m http.server 8080
```

Then open **`http://localhost:8080`**.

> Serve the UI on 8080, **not** 8000 — port 8000 is the phone brain's, and the
> two collide the moment you try the mobile tier.

To route against a real local model, set the env var **in window 1 before
starting the server**:

```powershell
$env:TWO_BRAIN_NPU_BRAIN=1
$env:TWO_BRAIN_CLOUD_BRAIN=1
.\.venv\Scripts\python.exe -m two_brain_router.api --tier pc
```

The server prints a full **trace** of every call — each brain's input and
output, tagged `[ON-DEVICE -- raw text]` or `[OFF-DEVICE -- masked]`, every
value→placeholder substitution at a crossing, and the rehydration on the way
back. It is on by default here; pass `--no-trace` to silence it. **The trace
contains raw PII**, which is the point — it is how you verify the guarantee
rather than take it on faith.

### From a phone on the same network

The camera and microphone need a *secure context*, so plain HTTP won't do:

```powershell
.\.venv\Scripts\python.exe ui\serve_https.py     # https://<this-machine>:8643
```

This serves the UI over TLS **and** proxies the API on the same origin, so no
mixed-content block and no CORS. The certificate is self-signed — the phone
warns once; tap through it. `ui/app.js` derives the API address from the page,
so nothing needs configuring on the phone.

### Command line, no UI

```powershell
.venv\Scripts\python.exe -m two_brain_router                        # 3-query demo
.venv\Scripts\python.exe -m two_brain_router --tier mobile          # route as the mobile tier
.venv\Scripts\python.exe -m two_brain_router --query "..." --json   # one query, machine-readable
.venv\Scripts\python.exe -m two_brain_router --query "..." --trace  # with the full data-path trace
```

### The mobile tier

No phone needed — the mock server speaks the same contract:

```powershell
.venv\Scripts\python.exe src\phone_brain\mock_phone_brain_server.py --port 8000
$env:TWO_BRAIN_PHONE_BRAIN=1
.venv\Scripts\python.exe -m two_brain_router --tier mobile
```

With a real Galaxy S25 attached, `tools/phone/live_test.py` walks the whole
path one step at a time and tells you exactly which one failed:

```powershell
.venv\Scripts\python.exe tools\phone\live_test.py          # needs adb + the phone
.venv\Scripts\python.exe tools\phone\live_test.py --mock   # harness self-check, no hardware
```

> Use **`adb forward`**, not `adb reverse` — the server is on the phone and the
> caller is your PC, so the port has to open on the host. Every doc in this repo
> said `reverse` until it was run against real hardware; the mock server hides
> the difference because it listens on the dev machine's own loopback either
> way. See `src/phone_brain/L_INTERFACE_CONTRACT.md`.

> Use **`127.0.0.1`, not `localhost`** anywhere you point at a local brain. On
> this machine `localhost` resolves to `::1` first, the servers bind IPv4 only,
> and the failed attempt costs **~2s per call** — enough to blow the routing
> budget on its own.

---

## Using the app

**Pick a local model** from the dropdown in the sidebar. Four are wired up:

| Model | Runs on |
|---|---|
| Phi-3.5-mini — NPU | Hexagon NPU, in-process via Genie/`ctypes` |
| Phi-3.5-mini — GPU | Adreno, `llama-server` on loopback |
| Qwen3-VL 4B — GPU | Adreno |
| Qwen3-VL 8B — GPU | Adreno, LLM layers on GPU, projector on CPU |

**Watch a query get split.** Ask something where part of the answer is
knowable and part isn't — post-cutoff facts, live data, or an exact figure:

```
What is the current spot price of Brent crude, and how does that compare with the 2020 low?
```

The local brain answers the 2020 half in a blue bubble; the live price becomes
the named gap and goes to the cloud in a separate amber bubble.

**Watch masking happen.** Include something the guard detects — email, phone,
SSN, or card number:

```
I got a mail from refunds@paypal-secure-billing.com asking me to confirm SSN
402-88-1955. Is this a known active phishing campaign, and what is the current
FTC reporting URL?
```

The cloud bubble carries a collapsed **"2 items masked before leaving this
device"** row. Expand it to see each `typed value → placeholder` pair and the
exact masked text that crossed. If the local model answers the whole thing,
nothing crosses and the profiler says *"2 detected — none left the device"*.

**The profiler pill** (bottom right) shows the tier that answered, difficulty
against the threshold, what was detected vs. masked, estimated latency and
cost, and the routing notes for that turn.

Other things worth knowing:

- **Conversation history lives in the browser**, not the router — `api.py`
  shares one router across every request and tab, so per-chat state there would
  cross-contaminate. The local brain gets recent turns raw; the cloud gets them
  masked.
- **Only the `solution` field is streamed.** The local model emits a JSON
  object; the router strips the envelope before it reaches the wire, so you
  never watch a bubble fill with `{"solution": "`.
- **Images need a vision model selected** — the two Qwen3-VL entries. Attaching
  one while a Phi-3.5 model is picked is refused rather than silently ignored.
  An image never leaves the device: there is no cloud vision model in this
  system at all, so when an image-bearing query escalates, what crosses is a
  *description* the local model wrote, masked like any other text.
- **An image-bearing query takes the heuristic path (Shape A)**, not the split.
  Shape C's JSON contract has nowhere to put an image yet.

---

## Real vs. mocked, tier by tier

| Tool | Mobile (3B) | AI PC (3B) | Cloud AI 100 (large) |
|---|---|---|---|
| `hardware_detect` | **real** — direct `adb shell` probe of a Galaxy S25 Ultra | **real** — `quad-client detect --json`, Snapdragon X Elite X1E80100, Hexagon v73 @ 45 TOPS | **mocked** — no cloud platform value exists in the tool schema at all |
| `convert_model` | mocked | mocked (QUAD's own tool — still blocked) | mocked |
| `profile_workload` | mocked | **real** — see note below | mocked |
| `orchestrate_workload` | mocked | **real** — see note below | n/a (the cloud tier isn't an on-device op-placement problem) |

`convert_model`/`profile_workload`/`orchestrate_workload` are mocked for mobile
and cloud. The hosted server's `qairt-converter` cannot even import (missing
`libpython3.10.so.1.0`) — a real, reproducible server-side defect. Bypassing
the server and running the SDK's own Windows-native converter *locally* got
much further (four real environment bugs fixed, real ops actually transformed)
before hitting a second, independent native-code defect: non-deterministic
uninitialized-memory reads in `ReshapeOp::calculateShape`, proven by three
identical runs producing three different garbage totals. Two execution paths,
two genuine compiler-level defects — see `docs/GAPS.md` #3/#3b and
`data/convert_model/_real_attempts_log.md` (attempts 1–5).

**AI PC `profile_workload`/`orchestrate_workload` — real, but not from QUAD's
own tool call.** `convert_model` is still genuinely blocked, so the table entry
stays "mocked" for it. But `data/profile_workload/pc_3b.json` and
`data/orchestrate_workload/pc_3b.json` are real captures (`_mock: false`) from
a *different* toolchain that routes around QUAD's compiler entirely:
Qualcomm's pre-built Genie/QNN artifact for Phi-3.5-mini-instruct, run for real
on this machine's Hexagon NPU. See
[`superpowers/deploy-local-brain-npu.md`](superpowers/deploy-local-brain-npu.md)
and `data/npu_model/phi-3.5-mini-instruct/` for the receipts — real per-token
latency (~74.2 ms/token, ~13.5 tok/s across 8 queries), real `QnnGraph_execute`
HTP-execution evidence, and a real QNN profiler capture.

**Mobile `hardware_detect`:** `adb` was missing entirely (installed Android SDK
platform-tools mid-session after winget's own package failed a hash check). The
attached Galaxy S25 Ultra needed USB debugging enabled and the on-device
authorization prompt accepted. Once authorized, `quad-client detect --platform
android --json` reached the device but returned placeholder values (`chipset:
"unknown"`, `storage_gb: 475.6` — this PC's own disk size, not the phone's), a
real bug in the client's android probe path. Worked around with direct
`adb shell getprop` / `/proc/meminfo` queries, so
`data/hardware_detect/mobile.json` is real end to end (Snapdragon 8 Elite for
Galaxy / SM8750, 8 cores, ~10.9 GB RAM, Android 16) — `docs/GAPS.md` #5.

### The `data/` receipts rule

**Nothing in `data/` is a guess without a receipt.** Every mocked file has a
`_real_*_log.md` beside it recording the actual tool calls attempted, their
arguments, and the verbatim error strings that forced the mock. Real captures
are marked `_mock: false`. A labeled stub with a logged reason is correct here;
an invented latency figure is not.

---

## Gaps

Full writeup in [`docs/GAPS.md`](docs/GAPS.md). Summary:

1. **`hardware_detect` self-detects the server's own container**, not the
   client device, when called directly over MCP (`quad-client detect` works
   around this with a local static probe).
2. **G8** — the real `quad.privacy` PII guardrail isn't present in this
   checkout. `src/two_brain_router/privacy/` mocks its detect/mask/rehydrate
   contract so the router logic is real and testable.
3. **`convert_model`'s compiler is broken on two independent execution
   paths** — see above. Five real attempts, logged. Not fixable from this
   client; don't spend a session retrying variations.
4. **Cloud AI 100 has no plumbing anywhere in QUAD-Client-main** — no
   platform/SDK/device value maps to it in any tool schema.
5. **`quad_mcp_client`'s android hardware-detection path** returns placeholder
   values even when adb is authorized and reachable.

---

## Sample output

Byte-exact from `python -m two_brain_router`. Query 3 shows the privacy
guarantee end to end — PII is detected up front but only masked at the moment
it crosses, and the final answer is rehydrated on-device:

```
> My email is jane.doe@example.com and my phone is 555-123-4567 -- can you draft a reply telling the sender their SSN 123-45-6789 was found in an old backup and needs to be rotated?
  routed to: cloud | difficulty=0.40 | est_latency_ms=1479 | est_cost_usd=0.00004
  - detected 3 PII entities in the query
  - LocalFastBrain runs on this device, so it gets the query unmasked -- nothing is transmitted
  - escalating: difficulty=0.40 (threshold 0.55) or local_latency_est=4855ms > budget 3000ms
  - sent off-device (masked): 'My email is [PII_EMAIL_1] and my phone is [PII_PHONE_1] -- can you draft a reply telling the sender their SSN [PII_SSN_1] was found in an old backup and needs to be rotated?'
  answer: [cloud:ai100 mock deep-brain response to: 'My email is jane.doe@example.com and my phone is 555-123-4567 -- can you draft a reply telling the sender their SSN 123-45-6789 was found in an old backup and needs to be rotated?' | context_used='']
```

If a change alters this, update it in the same commit — a stale transcript is
worse than none.

---

## Directory layout

**All source lives under `src/two_brain_router/`** — a `src/` layout matching
the parent repo's own `src/quad_mcp_client/`. Each subpackage is one
replaceable seam, so a mock can be swapped for the real thing without touching
anything above it.

```
two_brain_privacy_router/
  pyproject.toml            # package metadata; `-e .` puts src/ on the path
  requirements.txt
  run.ps1                   # generic uv venv bootstrap (repo convention)
  src/two_brain_router/
    __main__.py             # `python -m two_brain_router`
    cli.py                  # arg parsing + demo output
    api.py                  # HTTP server the chat UI talks to (/route, /route/sse, /models)
    trace.py                # ASCII-only terminal trace of every call and crossing
    privacy/                # SEAM: swap for quad.privacy once G8 lands
      patterns.py           #   regex table -- add entity types here
      guard.py              #   PIIGuard.detect/mask/rehydrate + invariant
    signals/                # what the fast brain / the four QUAD tools tell the router
      loader.py             #   TierSignals: reads data/<tool>/<name>.json
      difficulty.py         #   surface-feature score (Shape A, brains that don't self-rate)
      confidence.py         #   self-report parsing + confidence -> difficulty (Shape B)
      structured.py         #   JSON {solution, confidence, unknown} + SolutionStreamer (Shape C)
    routing/
      policy.py             #   RoutePolicy (thresholds, compression) + RouteDecision -- pure
      models.py             #   the local-model catalogue behind the UI's picker
      brains.py             #   NpuFastBrain / GpuLocalBrain / PhoneFastBrain / Cirrascale
                            #   + LocalFastBrain and CloudDeepBrain (stdlib-only stubs)
      router.py             #   TwoBrainRouter: decide -> answer -> mask at the boundary -> rehydrate
  ui/                       # the chat app -- plain HTML/CSS/JS, no build step
    index.html app.js styles.css markdown.js profiler.js
    serve_https.py          #   TLS + API proxy, so a phone gets camera/mic
    selftest.mjs            #   zero-dependency UI tests
  tools/phone/live_test.py  # staged adb -> endpoint -> router diagnostic for the S25
  src/phone_brain/          # the mobile fast brain: contract, deployment guide, mock server
  tests/                    # pytest; conftest.py puts src/ on sys.path (no install needed)
  data/                     # captured tool responses -- real and mocked, each logged
  docs/                     # ORCHESTRATOR.md, WALKTHROUGH.md, GAPS.md, PHONE_BRAIN.md
```

### Where new code goes

| You're adding... | Put it in |
|---|---|
| A new PII entity type | `privacy/patterns.py` |
| The real `quad.privacy` guardrail | replace `privacy/guard.py`; keep the `PIIGuard` contract |
| A real fast/deep brain | a new class in `routing/brains.py`, registered in `router.py::_build_fast_brain` behind its own env var |
| A model for the UI picker | `routing/models.py`'s `CATALOGUE` |
| A different escalation rule | `routing/policy.py` — pure, no I/O, unit-tested directly |
| A new tier | a `data/<tool>/<name>.json` set + an entry in `router.py`'s `_TIER_FILES` |
| A new tool response to consume | `signals/loader.py` |

When swapping a mock for the real thing, **change only that module**. If the
swap forces edits in `router.py`, the seam was drawn in the wrong place.

`signals/loader.DATA_DIR` can be pointed elsewhere with `TWO_BRAIN_DATA_DIR`,
to run against real captures rather than overwriting the logged mocks.

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q   # router, privacy, orchestration, HTTPS proxy
node ui\selftest.mjs                           # UI logic -- no package.json, by design
```

Three tests are the privacy guarantee itself. If the router changes, all three
must still pass **unmodified** — if one needs editing to pass, the guarantee
changed, and that is a review-worthy decision rather than a test fix:

- `test_routing.py::test_escalated_pii_never_reaches_cloud_unmasked`
- `test_orchestrator.py::test_pii_never_reaches_the_phone_or_the_cloud_unmasked`
  — asserts against the bytes that actually went out over the socket
- `test_structured_routing.py::test_pii_the_local_model_invented_in_the_gap_is_masked_before_it_crosses`

---

## Docs

[`docs/ORCHESTRATOR.md`](docs/ORCHESTRATOR.md) — how the router decides, the
three shapes, and the full env-var table ·
[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — build history, a worked trace,
and prioritized next steps · [`docs/GAPS.md`](docs/GAPS.md) — the five gaps
with verbatim error strings · [`docs/npu-deployment.md`](docs/npu-deployment.md)
and [`docs/PHONE_BRAIN.md`](docs/PHONE_BRAIN.md) — the AI-PC and Mobile fast
brains · [`CLAUDE.md`](CLAUDE.md) — the working agreement for agents.

### Version control

This sample is its own git repository, rooted at this directory — the
surrounding `QUAD-Client-main/` is a GitHub ZIP download with no `.git`. If it
is later turned into a proper clone, fold this in as a subtree or submodule
rather than leaving a nested `.git` inside a tracked tree.

Per the workspace convention, commit messages here carry **no AI-assistant
attribution trailer or footer**.
