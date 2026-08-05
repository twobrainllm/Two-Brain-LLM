# CLAUDE.md — Two-Brain Privacy Router

Guidance for Claude Code working in
`QUAD-Client-main/samples/two_brain_privacy_router/`.

This file covers what is *specific to this sample* and non-obvious from the
code. The workspace-level `QUAD-Client-main/CLAUDE.md` still applies for
everything else (Python version, `run.ps1` convention, commit-message policy,
MCP connectivity triage). The `README.md` here is the user-facing description —
read it for the archetype, the real-vs-mocked table, and the layout; this file
is the working agreement.

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

## Invariants — do not break these

The point of this sample is a privacy guarantee. It lives in the *ordering*
inside `routing/router.py::TwoBrainRouter.route`, and any change to routing
must preserve all four:

1. **Mask first.** The query is masked *before* any routing decision is made —
   before difficulty scoring, before latency estimation. A decision made on raw
   text has already read the PII.
2. **The invariant check is fatal.** `assert_masked_token_invariant` runs on
   every request and raises. Never downgrade it to a warning, a log line, or a
   test-only assertion.
3. **Only masked text crosses the boundary.** `CloudDeepBrain.answer` receives
   masked query + masked, compressed context. The vault never leaves the
   process. Escalated *context* gets masked too, not just the query.
4. **Rehydrate last, on-device.** Only after the answer is back.

`tests/test_routing.py::test_escalated_pii_never_reaches_cloud_unmasked` is the
regression test for all of this. If you change the router, that test must still
pass unmodified — if it needs editing to pass, the guarantee changed and that
is a review-worthy decision, not a test fix.

---

## Where code goes

All source lives under `src/two_brain_router/`. The README has the full
file-by-file table; the short version is that each subpackage is a **seam** for
a piece that is currently mocked:

| Seam | Replace when | Contract to keep |
|---|---|---|
| `privacy/guard.py` | `quad.privacy` (gap G8) becomes available here | `mask` / `rehydrate` / invariant |
| `signals/difficulty.py` | a fast-brain artifact can emit real logprobs | `score(query) -> float` in `[0, 1]` |
| `routing/brains.py` | `convert_model` produces a real artifact | the `Brain` protocol → `BrainResponse` |

When swapping a mock for the real thing, **change only that module** — if the
swap forces edits in `router.py`, the seam was drawn in the wrong place.

`routing/policy.py` is pure (no I/O, no brain calls) on purpose, so escalation
rules can be unit-tested directly. Keep it that way.

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

The demo transcript in the README is **byte-exact output**. If a change alters
it, update the README in the same commit — a stale transcript is worse than
none.

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
