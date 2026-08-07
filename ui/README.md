# Two-Brain Chat UI

A Claude-style chat interface for the Two-Brain router: sidebar with chat
history + search, a main chat panel, and an "emo robot" avatar that swaps
color to show which brain answered — **Lochmara `#007CC0`** for the local
fast brain, **Supernova `#FFC20E`** for the Cloud AI 100 deep brain.

Plain HTML/CSS/JS, no build step, no dependencies. Lives entirely under
`ui/` — nothing under `src/`, `data/`, `docs/`, or `tests/` was touched, so
this merges cleanly regardless of what lands on `main` in the meantime.

## Running it

**Live** — the chat talks to the real `TwoBrainRouter`:

```
UI_TEST=1 TWO_BRAIN_GPU_BRAIN=1 TWO_BRAIN_CLOUD_BRAIN=1 python ui/server.py
```

Open `http://localhost:8000`. Brains build lazily on the first request, so the
first reply is slow — a `llama-server` child has to start.

Each flag does something distinct:

| Flag | Effect if unset |
|---|---|
| `UI_TEST=1` | the local/cloud switch stops being authoritative. `route()` honours a forced tier only under this flag, so the policy decides and the switch becomes a preference — the sidebar says so rather than pretending otherwise. |
| `TWO_BRAIN_GPU_BRAIN=1` | the fast brain stays a stub and **image attachment is disabled**, since no vision model is loaded |
| `TWO_BRAIN_CLOUD_BRAIN=1` | the deep brain stays a labelled stub instead of reaching Cirrascale |

**Standalone** — front-end only, no backend:

```
cd ui && python -m http.server 8000
```

`/api/health` fails, the UI falls back to `mockRespond()`, and the sidebar
shows "Mock mode". This still works on purpose: the front-end was built to demo
without the X-Elite box, and that has not been taken away.

## Images

The `+` beside the composer attaches a PNG/JPEG/WebP (12 MB cap); pasting an
image into the input works too.

**The image never leaves the device.** That is not a policy choice but a
property of the deployment — the cloud tier is a text-only LLM and the service
has no vision model at all. So an image-bearing query is either answered
locally by the VLM, or, if it escalates, the local VLM first converts the image
to *words*; that description is masked like any other text and only the
description crosses. A physics diagram becomes "block on an incline, angle …",
which the text-only 70B can then reason over.

## Real vs. mocked

**Real:** the whole UI shell — sidebar, date-grouped history, search, delete,
composer, image attachment, avatar colour/animation. And now the reply itself:
`ui/server.py` runs the same `TwoBrainRouter` the CLI uses, with the same
masking order, so answers come from the real GPU fast brain or the real
Cirrascale deep brain.

**Mocked:** only `mockRespond()`, and only when the backend is unreachable.

Which colour a reply gets is now a *real* routing outcome. The server returns
`tier_answered` and the UI paints that, not whatever the switch was set to —
so with `UI_TEST=0` you will see the policy overrule the switch.

Each message stores its own `tier` at send time, so switching later does not
repaint history.

## Query profiler

The pill in the top-right of the header (click to expand, Dynamic-Island
style) shows per-query telemetry: difficulty score against the escalation
threshold, PII/privacy detection, estimated latency and cost, the router's
own escalation reasoning, and the AI PC's real captured hardware context.

**Real:** the formulas and constants. `ui/profiler.js` is a line-for-line
port of `signals/difficulty.py`'s `DifficultyEstimator`,
`privacy/patterns.py`'s `PATTERNS` + `privacy/guard.py`'s `PIIGuard.detect`,
`routing/policy.py`'s latency estimate and `routing/brains.py`'s cost
formula, and `routing/policy.py`'s escalation/local note text — with the
same profiled constants (`data/profile_workload/pc_3b.json`,
`cloud_large.json`) and the same captured hardware string
(`data/hardware_detect/ai_pc.json`), cited inline in `profiler.js`. Given
the same query, it produces the same difficulty score, PII count, and
latency/cost estimate the real router would.

**Mocked:** the *invocation*. Nothing in `profiler.js` calls the Python
router — it's a JS re-implementation run against whatever tier the sidebar
toggle is set to, not a real routing decision. The "actual" latency shown
alongside the estimate is the mock thinking choreography's own measured
wall-clock duration, not a real model's.

Each assistant message stores its own `metrics` snapshot (same pattern as
`tier`), so the profiler always reflects whichever message last answered —
switching chats or tiers later doesn't recompute history.

## Wiring it to the real router — done

This section used to be a plan. It is now history, kept because the plan's
last two items are still open.

- **Done:** `ui/server.py` is the API server (stdlib `http.server`, no
  FastAPI/Flask — the base package is stdlib-only and this stays consistent
  with it). `mockRespond()` is now a fallback rather than the path.
- **Done differently:** the plan said *remove the sidebar toggle once tier is
  a real routing decision*. Instead the toggle became real, gated behind
  `UI_TEST=1` — being able to force either brain is genuinely useful for
  demonstrating the two tiers side by side. Outside that flag it is inert and
  the policy decides, so it cannot masquerade as routing behaviour.
- **Still open:** `profiler.js` keeps its own ported copies of the difficulty
  and latency formulas. The `/api/chat` response already carries the real
  `difficulty_score`, `est_latency_ms`, `est_cost_usd`, `pii_entities_masked`
  and `notes`, and the UI now prefers those when the backend is live — but the
  JS formulas remain as the mock-mode fallback, so there are still two sources
  of truth. Retiring them is worthwhile.
- **Still true:** keep the per-message tier badge/dot. Colour alone is not an
  accessible signal.

## Persistence

Chats are stored in the browser's `localStorage` under
`twoBrainChats` — on-device, no server, no database. They persist across
closing the tab/browser and are only removed when you explicitly delete a
chat (trash icon on hover in the sidebar) or clear that browser's site
data. Note this is per-browser-profile: a different browser or a private/
incognito window won't see the same history.
