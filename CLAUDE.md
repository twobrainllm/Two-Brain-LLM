# CLAUDE.md — Two-Brain Privacy Router

Guidance for Claude Code working in
`QUAD-Client-main/samples/two_brain_privacy_router/`.

This file covers what is *specific to this sample* and non-obvious from the
code. The workspace-level `QUAD-Client-main/CLAUDE.md` still applies for
everything else (Python version, `run.ps1` convention, commit-message policy,
MCP connectivity triage). The `README.md` here is the user-facing description —
read it for the archetype, the real-vs-mocked table, and the layout; this file
is the working agreement.

Before starting non-trivial work here, read
[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — it has the build history, a
worked trace of a routed query, and the prioritized next steps with their
dependencies. Picking up an item from that list is usually the right default.
If the item you're picking up touches the fast-brain seam (`routing/brains.py`
or `signals/difficulty.py`), read the Branch state section below first — one
tier's seam is already closed, and there's unmerged, in-progress work on the
other.

---

## What this project is

A privacy- and power-aware router: a local "fast brain" (Mobile 1B / AI PC 3B)
answers what it can, and hard queries are masked, compressed, and escalated to
a Cloud AI 100 "deep brain". Routing thresholds are driven by captured
`hardware_detect` → `convert_model` → `profile_workload` →
`orchestrate_workload` responses under `data/`.

**The routing and masking logic is real, working Python. Only the model
artifacts and hardware probes it consumes are mocked** — and only because the
tools that would produce them are genuinely broken (see Gaps below).

---

## This is its own git repository

`git init` is rooted at **this directory**, not at `QUAD-Client-main/` (which
is a GitHub ZIP download with no `.git`). Consequences:

- Run `git` commands from this directory. `git status` in the parent will fail.
- If `QUAD-Client-main` is ever turned into a real clone, fold this in as a
  subtree or submodule — do not leave a nested `.git` inside a tracked tree.
- Commit messages carry **no AI-assistant attribution** (no `Co-Authored-By:`
  naming a model or vendor, no "Generated with …" footer). This is the
  workspace policy from `QUAD-Client-main/CLAUDE.md` and it applies here.

---

## Branch state — both local seams closed, evaluation still in flight

This sample has grown beyond `main`'s original architecture, across several
branches, faster than the docs. Know which branch you're on before assuming
what exists:

| Branch | Adds | Status |
|---|---|---|
| `main` | Everything in the base architecture, **plus the AI-PC-tier `NpuFastBrain`** (merged via PR #1, `ca97bdd`) | Merged, stable |
| `local_brain` | `main` + `src/phone_brain/` — the Mobile-tier fast brain (Genie/QNN on a Galaxy S25), built by Nikhita, as a standalone package | Superseded by `js/orchestrator` |
| `js/orchestrator` | `local_brain` + **`PhoneFastBrain`**, wiring that phone brain in behind the `Brain` seam, plus confidence-driven routing | Both local tiers now have a real brain |
| `p4-eval-demo` (`origin` only — not checked out here) | `evaluation/` — a benchmark/scenario harness, built by THRISHA | In progress; not reviewed against this branch |

**The AI-PC tier's fast-brain seam is closed for real.** `NpuFastBrain`
(`routing/brains.py`) runs Qualcomm's pre-built Phi-3.5-mini-instruct
Genie/QNN artifact in-process via `ctypes`, confirmed executing on this
machine's Hexagon NPU, wired into `TwoBrainRouter` for the `pc_3b` tier
behind `TWO_BRAIN_NPU_BRAIN=1` (`router.py::_build_fast_brain` — unset by
default so the base package stays stdlib-only). `data/profile_workload/
pc_3b.json` and `data/orchestrate_workload/pc_3b.json` are now real
captures. Full build history, real bugs found and fixed, and every receipt:
[`superpowers/deploy-local-brain-npu.md`](superpowers/deploy-local-brain-npu.md)
(the finished version) and [`docs/npu-deployment.md`](docs/npu-deployment.md)
(short architecture summary). See also [next step #3](docs/WALKTHROUGH.md#next-steps).

**`NpuFastBrain` now self-rates as well** (Shape B), so the AI PC tier routes on
the model's own confidence rather than the keyword heuristic. As
`ORCHESTRATOR.md` predicted, this needed a prompt change plus a re-profile and
**no** router change — `route()` and `policy.py` were untouched. Three real
defects turned up while getting there and are fixed and logged in
`data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md` (Attempt 5):
`_check` treated Genie *warnings* as fatal (so the class's own token cap crashed
the call it was meant to truncate), `close()` was not idempotent (double-free
corrupting the next session), and `genie-t2t-run.exe` hits Windows MAX_PATH in a
deep checkout. **Read Attempt 5's calibration finding before trusting this
tier's confidence number** — it separates "answered" from "didn't", not easy
from hard.

**The Mobile tier's fast-brain seam is closed too**, on branch
`js/orchestrator`. `PhoneFastBrain` (`routing/brains.py`) speaks the
OpenAI-shaped contract in `src/phone_brain/L_INTERFACE_CONTRACT.md` and is
wired into `TwoBrainRouter` behind `TWO_BRAIN_PHONE_BRAIN=1`. It was the first
brain to **self-rate**, which flips the order of the routing decision —
read **[`docs/ORCHESTRATOR.md`](docs/ORCHESTRATOR.md)** before changing
`route()`; it explains the two decision shapes and why the latency budget is
checked before the call and never after.

Four of the five reconciliation points from
[`docs/PHONE_BRAIN.md`](docs/PHONE_BRAIN.md) are resolved in that work (privacy
ordering, the duplicated threshold, the `route()` restructure, and the
`test_phone_brain.py` → `bench_phone_brain.py` rename). **One is still open:**
the model/fixture mismatch — `data/profile_workload/mobile_1b.json` is still a
mock describing Qwen2.5-1.5B int4, while the phone brain runs
Llama-3.2-3B w4a16. Real numbers need a `bench_phone_brain.py` run against the
actual S25, with a receipt, per the `data/` rule below. Until then the mobile
tier's *latency budget pre-check* is reasoning from the wrong model.

**Mobile's "not confident" path now has a third destination, not just the
cloud.** When `TWO_BRAIN_NPU_BRAIN=1` is also set, an unconfident phone
answer is followed by a direct call to the AI PC's `NpuFastBrain` — the same
real model the `pc` tier uses — instead of falling back to the surface-feature
heuristic or escalating straight to the cloud. Still `tier_answered="local"`
(nothing crosses the cloud boundary); still masked-text-only into that brain;
deliberately not latency-optimized (two real local inferences on one query
when the phone isn't confident). See
[`docs/ORCHESTRATOR.md`](docs/ORCHESTRATOR.md)'s "The escalation brain"
section before touching `TwoBrainRouter.escalation_brain` or
`_build_escalation_brain`.

---

## Invariants — do not break these

The point of this sample is a privacy guarantee. It lives in the *ordering*
inside `routing/router.py::TwoBrainRouter.route`, and any change to routing
must preserve all four:

1. **Mask first.** The query is masked *before* any routing decision is made —
   before difficulty scoring, before latency estimation, and before any brain
   is called. A decision made on raw text has already read the PII.
2. **The invariant check is fatal.** `assert_masked_token_invariant` runs on
   every request and raises. Never downgrade it to a warning, a log line, or a
   test-only assertion.
3. **Only masked text crosses the boundary.** `CloudDeepBrain.answer` receives
   masked query + masked, compressed context. The vault never leaves the
   process. Escalated *context* gets masked too, not just the query.
   **`PhoneFastBrain` is on the far side of a boundary as well** — the mobile
   model runs on a physically separate device over HTTP — so the same rule
   applies to it, and it additionally refuses a non-loopback host without an
   explicit opt-in. **`escalation_brain` (`NpuFastBrain`, when mobile's phone
   isn't confident) is not on the far side of this specific boundary** — it
   runs in-process on this machine, so it never reaches the cloud — but it
   still only ever receives `masked_query.masked_text`, same as every other
   brain here.
4. **Rehydrate last, on-device.** Only after the answer is back.

Note that invariant 1 is *why* the confidence path is safe: a self-rating brain
is called before the routing decision (see
[`docs/ORCHESTRATOR.md`](docs/ORCHESTRATOR.md)), but masking is still ahead of
both, so what it receives is masked either way.

`tests/test_routing.py::test_escalated_pii_never_reaches_cloud_unmasked` is the
regression test for all of this, and
`tests/test_orchestrator.py::test_pii_never_reaches_the_phone_or_the_cloud_unmasked`
is its counterpart for the phone boundary — it asserts against the bytes that
actually went out over the socket, not a mocked call. If you change the router,
both must still pass unmodified — if either needs editing to pass, the
guarantee changed and that is a review-worthy decision, not a test fix.

---

## Where code goes

All source lives under `src/two_brain_router/`. The README has the full
file-by-file table; the short version is that each subpackage is a **seam** for
a piece that is currently mocked:

| Seam | Replace when | Contract to keep |
|---|---|---|
| `privacy/guard.py` | `quad.privacy` (gap G8) becomes available here | `mask` / `rehydrate` / invariant |
| `signals/difficulty.py` | a fast brain can emit a real signal — **done for both local tiers**, see `signals/confidence.py` | `score(query) -> float` in `[0, 1]` |
| `routing/brains.py` | a tier gets a real artifact — **done for both local tiers** | the `Brain` protocol → `BrainResponse` |

When swapping a mock for the real thing, **change only that module** — if the
swap forces edits in `router.py`, the seam was drawn in the wrong place.
`routing/brains.py`'s AI-PC filler (`NpuFastBrain`) is the worked example: it
needed a two-line addition to `router.py` (`_build_fast_brain`'s env-gated
tier switch), nothing in `policy.py` or `signals/`.

`routing/policy.py` is pure (no I/O, no brain calls) on purpose, so escalation
rules can be unit-tested directly. Keep it that way.

Both fast-brain seams now have real fillers (`NpuFastBrain`, `PhoneFastBrain`),
so `brains.py` is the worked example for a third: add a class to the **flat**
module, keep runtime imports lazy, and register it in `_build_fast_brain`
behind its own env var. Don't split it into a `routing/brains/` subpackage —
that was proposed once and the codebase went the other way.

`signals/difficulty.py` (the surface-feature heuristic) is still the signal for
any brain with `reports_confidence = False`, so it is not dead code — but that
set is now **only the stubs** (`LocalFastBrain`, `CloudDeepBrain`), i.e. the
default stdlib-only path. Both real brains self-rate. **It is
no longer the fallback for an unparseable confidence** — that used to be true
but isn't any more: `route()`'s confidence path now treats "no parseable
number" the same as "definitely not confident" (`difficulty = 1.0`), not a
second, different signal. See `docs/ORCHESTRATOR.md`'s "`confidence = None`
is not `confidence = 0.0`" section for why conflating the two was wrong.

---

## The `data/` receipts rule

**Nothing in `data/` is a guess without a receipt.** Every mocked file is
accompanied by a `_real_*_log.md` in the same directory recording the actual
tool calls attempted, their arguments, and the verbatim error strings that
forced the mock. Real captures are marked `_mock: false`.

If you add or update anything under `data/`:

- State whether it is real or mocked, and if mocked, **which real attempt
  preceded it and how it failed** — append to that tool's `_real_*_log.md`.
- Never fabricate plausible-looking numbers to make a demo run. A labeled stub
  with a logged reason is correct here; an invented latency figure is not.
- Keep the response *shape* faithful to the tool's real output, since
  `signals/loader.py` only knows the shape.

`signals/loader.DATA_DIR` can be pointed elsewhere with the
`TWO_BRAIN_DATA_DIR` env var — use that to run against real captures rather
than overwriting the logged mocks.

---

## Commands

`run.ps1` is the generic uv bootstrap (repo convention) and creates a local
`.venv` here:

```powershell
.\run.ps1                                     # one-time; installs the package with -e .
.venv\Scripts\python.exe -m two_brain_router  # demo: easy/local, hard/cloud, PII/cloud
.venv\Scripts\python.exe -m pytest tests\ -q
```

Without that venv, the parent repo's interpreter works if `src/` is on the
path:

```bash
PYTHONPATH=src ../../.venv/Scripts/python.exe -m pytest tests/ -q
PYTHONPATH=src ../../.venv/Scripts/python.exe -m two_brain_router
```

`tests/conftest.py` puts `src/` on `sys.path`, so the suite runs uninstalled.

Both real brains are **off by default** — the suite and the demo run
stdlib-only with stub brains unless you opt in:

```powershell
# Mobile tier against the real phone brain -- no phone needed, use the mock:
.venv\Scripts\python.exe src\phone_brain\mock_phone_brain_server.py --port 8000
$env:TWO_BRAIN_PHONE_BRAIN=1
.venv\Scripts\python.exe -m two_brain_router --tier mobile
```

`docs/ORCHESTRATOR.md` has the full env-var table. One trap worth repeating:
use **`127.0.0.1`, not `localhost`** — on this machine `localhost` costs ~2s
per call (IPv6 `::1` attempted first, server binds IPv4 only), which is enough
to blow the 3000ms routing budget on its own.

The demo transcript in the README is **byte-exact output**. If a change alters
it, update the README in the same commit — a stale transcript is worse than
none. The `--tier pc` transcript is the one pinned there; it is unaffected by
the mobile tier's brain, since `pc` uses the non-confidence path.

---

## Gaps — current blockers

Full writeup with error strings in `docs/GAPS.md`. What matters for planning
work here:

| # | Gap | Effect on this project |
|---|---|---|
| 1 | `hardware_detect` over MCP self-reports the *server's* container | use `quad-client detect` (client-local-static probe), not the raw tool |
| 2 (G8) | `quad.privacy` not in this checkout | `privacy/` is a working mock of its contract |
| 3 / 3b | `convert_model` broken on **two independent paths** — hosted server missing `libpython3.10.so.1.0`; local QAIRT hits a non-deterministic uninitialized-memory read in `ReshapeOp::calculateShape` | no real artifact → both brains are labeled stubs, difficulty is surface-feature-based |
| 4 | Cloud AI 100 has no platform/SDK/device value in any tool schema | cloud tier mocked end-to-end from the public datasheet |
| 5b | `quad-client detect --platform android` returns placeholder values | mobile hardware data came from direct `adb shell` queries |

Gaps 3/3b are not fixable from this client. Do not spend a session retrying
`convert_model` variations — attempts 1–5 are already logged in
`data/convert_model/_real_attempts_log.md` with the exact failure mode of each.
If you have a reason to believe the environment changed, verify that first.

**Bypass, not a fix, for 3/3b's AI PC tier — done.**
`superpowers/deploy-local-brain-npu.md` routed around `convert_model`
entirely via Qualcomm AI Hub's pre-built artifact — a different toolchain,
not a retry — and got a real model (`NpuFastBrain`) executing on this
machine's NPU, wired into the router. It doesn't close gap 3/3b (QUAD's own
compiler is still broken, and the Mobile tier still needs it or an
equivalent bypass); it just means the AI-PC fast-brain seam no longer has to
wait on that fix. See Branch state above.
