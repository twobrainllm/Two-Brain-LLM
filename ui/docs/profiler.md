# Plan: Live "profiler" panel for the chat UI

## Context

The chat UI (`ui/`) currently shows only the tier color (local/cloud) per
message. The Python router (`src/two_brain_router/`) already computes much
richer per-query telemetry — difficulty score, PII/privacy signal, estimated
latency, estimated cost, and the human-readable reasoning behind the routing
decision — but none of it surfaces in the UI. The goal is a live "profiler"
that exposes these numbers per query, in the same Dynamic-Island-style
interaction the user described: a small rounded-rect pill (not a true circle
— matches the robot avatar's `rx=10` and the composer's rounding, per
correction) that expands into a full metrics card on click.

Chat replies aren't wired to the real `TwoBrainRouter` yet (see
`ui/README.md`) — the reply and tier are still a manual mock toggle. So the
profiler's numbers must also come from the client for now. The important
constraint, matching this project's own real-vs-mocked honesty rule: wherever
a Python formula already exists (difficulty scoring, PII regex detection,
latency/cost estimates, escalation reasoning text), **port that exact
formula/data into JS** rather than inventing new numbers. That keeps the
profiler's numbers *algorithmically real*, even though no real model or
network call backs them yet — the same spirit as `privacy/guard.py` and
`routing/policy.py` being real, working logic around mocked model artifacts.

Checked the `p4-eval-demo` branch's evaluation harness
(`evaluation/benchmark.py`, `evaluation/summarize.py`) to confirm there's no
additional "official" metric set hiding elsewhere in the project — it reports
exactly `RouteDecision`'s five fields (`tier_answered`, `difficulty_score`,
`est_latency_ms`, `est_cost_usd`, `pii_entities_masked`). The profiler's scope
below matches that plus the router's `notes` reasoning strings, which the
eval harness doesn't surface but are valuable for an interactive UI.

## What gets surfaced, and where it's ported from

| Metric | Source (Python) | How it's ported |
|---|---|---|
| Tier answered | already in `message.tier` | no change |
| Difficulty score (0–1) + escalate threshold (0.55) | `signals/difficulty.py:11-27` (`DifficultyEstimator.score`, `HARD_QUERY_MARKERS`) | verbatim port to JS |
| Privacy / PII count + entity types | `privacy/patterns.py:12-17` (`PATTERNS`: EMAIL, PHONE, SSN, CREDIT_CARD) + `privacy/guard.py:33-40` (`PIIGuard.detect`) | verbatim regex + detection loop port |
| Estimated latency | `routing/policy.py:44-48` (local) + `routing/brains.py:86-94` (cloud, includes network RTT) | port formulas, using **real captured constants**: `data/profile_workload/pc_3b.json` (`ttft_mean=142`, `per_token_mean=94.5`) and `cloud_large.json` (`network_rtt_mean=45`, `ttft_mean=320`, `per_token_mean=12.4`) |
| Estimated cost | `routing/brains.py:95` | port formula; `token_cost_usd_per_1k`: `0.0` local (`pc_3b.json`), `1.8` cloud (`cloud_large.json`) |
| Escalation reasoning | `routing/policy.py:56-68` (`escalation_note`/`local_note`) + `routing/router.py:70-75` (masking note) + `router.py:117-118` (compression note) | port text templates |
| Device context (static footer) | `data/hardware_detect/ai_pc.json` | real captured strings: "Snapdragon X Elite X1E80100 · Hexagon NPU v73 @ 45 TOPS · 12 cores" |
| Actual latency | the mock "thinking" choreography's own measured wall-clock duration | labeled clearly as sim timing, not the estimate |

## UI / interaction design

- **Collapsed pill**: replaces the current static `.mock-badge` in
  `.main-header` (`ui/index.html` around line 76), top-right, fixed within
  the header so it doesn't scroll with the chat. Rounded-rect (~14px radius,
  matching the robot avatar/composer, not a 999px circle pill). Shows a
  glance state: tier-colored dot (reusing `--brain-local`/`--brain-cloud`)
  + short readout, e.g. "● Local · 620ms".
- **Expanded card**: click grows the pill in place (transform-origin
  top-right, width/height transition — the Dynamic Island effect) into a
  larger rounded-rect card that drops down over the chat area
  (`position: absolute`, doesn't reflow layout). Contents: difficulty as a
  small gauge/bar with the 0.55 threshold marked, privacy line (entity
  count + types, or "No PII detected"), latency (estimate vs. actual),
  cost, the reasoning notes list, and the device-context footer.
- Click again / click outside / Escape collapses back to the pill.
- Always reflects the **latest** assistant message's metrics (per the
  scope decision) — updates the moment a reply lands, same trigger point
  as today's post-landing `happy` expression in `handleSend`.
- Empty state (no assistant message yet): pill renders in a neutral idle
  style ("—"), no expand target.

## Data model change

Each assistant message object gains a `metrics` field, computed once in
`handleSend` (`ui/app.js`) at the same point the mock answer is generated:

```js
{ difficulty, piiEntities: [{type, count}], piiCount, estLatencyMs,
  actualLatencyMs, estCostUsd, notes: [...] }
```

Persisted to `localStorage` exactly like today's messages — old messages
without `.metrics` just render the profiler's idle state gracefully, no
migration needed.

## New/changed files

- **`ui/profiler.js`** (new) — pure functions only: ported difficulty
  scorer, PII pattern table + detector, latency/cost formulas with the real
  captured constants above, and note-text templates. Kept separate from
  `app.js`'s UI/DOM code, mirroring how the Python side keeps `signals/`,
  `privacy/`, and `routing/policy.py` as pure modules apart from
  `router.py`'s orchestration.
- **`ui/app.js`** — call `profiler.js` functions in `handleSend` to compute
  and store `metrics` on the assistant message; render/update the pill and
  card; wire click-to-expand/collapse.
- **`ui/index.html`** — replace the `.mock-badge` span with the pill +
  expanded-card markup.
- **`ui/styles.css`** — pill/card styling, expand/collapse transition,
  difficulty gauge.
- **`ui/README.md`** — extend the real-vs-mocked section: profiler formulas
  and constants are real (ported/copied with sources cited above); their
  *invocation* is still mock (no real model or network call backs them
  yet).

No Python files under `src/two_brain_router/` are touched — this is
entirely additive within `ui/`.

## Verification

- Headless-Chromium pass (same Playwright pattern used throughout this
  session): send an easy query, a hard/multi-clause query, and a
  PII-laden query; expand the pill each time; screenshot; confirm the
  displayed difficulty/PII-count/latency numbers match hand-computed
  expected values from the ported formulas.
- Confirm collapse/expand animation and that clicking outside collapses it.
- Confirm the pill shows the idle state before any message exists, and
  updates correctly after each new reply (including across a local→cloud
  tier switch).
- No Python test suite impact (`tests/` untouched) since no `src/` files
  change.
