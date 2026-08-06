# Two-Brain Chat UI

A Claude-style chat interface for the Two-Brain router: sidebar with chat
history + search, a main chat panel, and an "emo robot" avatar that swaps
color to show which brain answered — **Lochmara `#007CC0`** for the local
fast brain, **Supernova `#FFC20E`** for the Cloud AI 100 deep brain.

Plain HTML/CSS/JS, no build step, no dependencies. Lives entirely under
`ui/` — the backend it talks to (`src/two_brain_router/api.py`) is one new
file, so this still merges cleanly; nothing existing under `src/` was
changed to accommodate it.

**Now wired to the real router**, verified end-to-end in a browser
(real routing, real PII masking display, escalation, and the offline
fallback below) — this section used to describe a wiring plan; that plan is
done, see "How it actually works" below for what shipped instead of what
was proposed.

## Running it

Two servers, both stdlib-only:

```powershell
# 1. The router API (from the sample root, not ui/)
.venv\Scripts\python.exe -m two_brain_router.api          # pc tier, stub brains
# add --tier mobile, or set TWO_BRAIN_NPU_BRAIN=1 / TWO_BRAIN_PHONE_BRAIN=1
# first -- see ../docs/ORCHESTRATOR.md

# 2. The UI itself, in a second terminal
cd ui
python -m http.server 8000
```

Then open `http://localhost:8000`. The sidebar's "● Live — routing as \<tier\>"
line confirms the API is reachable. **The API server is optional** — see
"Offline fallback" below; the UI still works, clearly labeled, without it.

## How it actually works

**Real, end-to-end:** sending a message calls
`TwoBrainRouter.route(query, context)`
(`src/two_brain_router/routing/router.py`) through `/route`
(`src/two_brain_router/api.py`) and renders the actual `RouteDecision` --
which tier answered, the real answer text, the real difficulty score,
latency, cost, and escalation notes, and (per query) which PII entity types
were masked before anything left the device. The "thinking" avatar animation
now paces off the real call's actual duration instead of a canned guess.

**Also real, but a *different* real thing than the router:** the sidebar's
color-swap logic, chat history/search/persistence, and the robot avatar
itself were already real before this was wired up and are unaffected.

## Offline fallback

If `/route` is unreachable (server not started, wrong port, killed
mid-session), the UI does **not** error -- it falls back to a simulated
reply, labeled "(offline preview)" in that message's badge, with the
sidebar status line switching to "○ API offline". This is `mockRespond()` /
`profiler.js`'s `computeMetrics()` from the UI's original mock-only version
-- kept intentionally rather than deleted once the real path landed, so the
UI still demos cleanly with no backend running (e.g. showing it to someone
without starting Python). The sidebar's "Preview tier" toggle only affects
this fallback path; it has no effect at all when the API is reachable, since
tier is then always the server's real decision.

Each message remembers whether it was answered live or by the fallback
(`msg.live`), so a mid-session backend restart never repaints history --
verified: an offline-preview reply keeps its label even after the API comes
back and the next message answers for real.

## Query profiler

The pill in the top-right of the header (click to expand, Dynamic-Island
style) shows per-query telemetry: difficulty score against the escalation
threshold, PII/privacy detection, estimated latency and cost, the router's
own escalation reasoning, and the answering tier's real captured hardware
context.

**Live path:** every field is the real `RouteDecision`, adapted into this
card's shape by `metricsFromRouteResponse()` in `app.js` -- not
recomputed. The one addition beyond what `RouteDecision` itself carries is
`pii_entities` (a per-type breakdown), which `api.py` computes separately,
read-only, purely for this display -- see its docstring for why that can't
affect routing.

**Offline-fallback path:** `ui/profiler.js`'s ported formulas -- still a
line-for-line port of `signals/difficulty.py`, `privacy/patterns.py` +
`privacy/guard.py`, and `routing/policy.py`/`routing/brains.py`'s cost
formula, cited inline in `profiler.js`. Both paths produce the exact same
field shape, so `renderProfiler()` doesn't know or care which one answered.

Each assistant message stores its own `metrics` snapshot (same pattern as
`tier`/`live`), so the profiler always reflects whichever message last
answered -- switching chats later doesn't recompute history.

## Persistence

Chats are stored in the browser's `localStorage` under
`twoBrainChats` — on-device, no server, no database. They persist across
closing the tab/browser and are only removed when you explicitly delete a
chat (trash icon on hover in the sidebar) or clear that browser's site
data. Note this is per-browser-profile: a different browser or a private/
incognito window won't see the same history.
