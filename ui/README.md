# Two-Brain Chat UI

A Claude-style chat interface for the Two-Brain router: sidebar with chat
history + search, a main chat panel, and an "emo robot" avatar that swaps
color to show which brain answered — **Lochmara `#007CC0`** for the local
fast brain, **Supernova `#FFC20E`** for the Cloud AI 100 deep brain.

Plain HTML/CSS/JS, no build step, no dependencies. Lives entirely under
`ui/` — nothing under `src/`, `data/`, `docs/`, or `tests/` was touched, so
this merges cleanly regardless of what lands on `main` in the meantime.

## Running it

```
cd ui
python -m http.server 8000
```

Then open `http://localhost:8000`. (Opening `index.html` directly by
double-clicking also works in most browsers — the app only uses
`localStorage`, no `fetch` calls yet.)

## Real vs. mocked

**Real:** the entire UI shell — sidebar, date-grouped chat history (Today /
Yesterday / Previous 7 Days / Previous 30 Days / Older), live search over
titles and message text, per-chat delete, the composer, and the robot
avatar's color/animation logic.

**Mocked:** the reply itself. `mockRespond()` in `app.js` returns a canned
string; no model runs and no query leaves the browser tab. Which color the
next reply's avatar gets is **not** a routing decision — it's whatever the
"Simulated brain" toggle in the sidebar footer is set to. This was a
deliberate call, not a placeholder we forgot to wire up: this UI was built
in an environment that can't reach the X-Elite box's NPU runtime, so there
was nothing real to call. The toggle exists to let you preview the color
swap without a backend.

Each message stores its own `tier` at send time, so switching the toggle
later doesn't repaint history — old messages keep the color of whatever
tier "answered" them, same as the real router would.

## Wiring it to the real router

`TwoBrainRouter.route(query, context)`
(`src/two_brain_router/routing/router.py`) already returns exactly the
signal this UI needs: a `RouteDecision` with `tier_answered: "local" |
"cloud"` and `answer`. Nothing about the routing/masking logic needs to
change for this UI — it's a pure consumer.

To wire it up on the target device:

1. **Add a thin API server** (new file, e.g. `ui/server.py` or
   `src/two_brain_router/api.py` — a new module either way, not an edit to
   an existing one). A single `FastAPI`/`Flask` endpoint is enough:

   ```python
   router = TwoBrainRouter(tier="pc")  # set TWO_BRAIN_NPU_BRAIN=1 for the real NPU brain

   @app.post("/route")
   def route(body: dict):
       decision = router.route(body["query"], body.get("context", ""))
       return {"tier": decision.tier_answered, "answer": decision.answer}
   ```

2. **Replace `mockRespond()`** in `app.js` with a `fetch("/route", ...)`
   call, and use the response's `tier` field instead of
   `state.currentTier`.
3. **Remove the sidebar toggle** once tier is a real routing decision
   instead of a manual override — at that point it's dead UI, not a
   feature.
4. Keep the per-message `tier_badge`/dot — it's useful even once real,
   since color alone isn't an accessible signal.

Until step 1 exists on a machine that can actually run it, this stays
labeled mock, per this repo's own rule: nothing gets to look real without
a receipt.

## Persistence

Chats are stored in the browser's `localStorage` under
`twoBrainChats` — on-device, no server, no database. They persist across
closing the tab/browser and are only removed when you explicitly delete a
chat (trash icon on hover in the sidebar) or clear that browser's site
data. Note this is per-browser-profile: a different browser or a private/
incognito window won't see the same history.
