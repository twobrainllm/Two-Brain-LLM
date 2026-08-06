# The Orchestrator

How `TwoBrainRouter` decides, now that both local tiers have a real brain
behind them.

There is no separate orchestrator component to build — **`TwoBrainRouter`
(`src/two_brain_router/routing/router.py`) is O.** The phone-brain docs refer
to L (fast brain), O (orchestrator), and C (deep brain); the mapping is:

| Contract name | This codebase |
|---|---|
| **L** — fast brain | `NpuFastBrain` (AI PC) / `PhoneFastBrain` (Mobile) / `LocalFastBrain` (stub) |
| **O** — orchestrator | `TwoBrainRouter` + `RoutePolicy` |
| **C** — deep brain | `CloudDeepBrain` (stub — gap #4) |

---

## The two decision shapes

The privacy ordering never changes: **mask → assert → decide → answer →
rehydrate.** What changes per tier is *where the difficulty signal comes
from*, and that determines whether the fast brain is called before or after
the decision.

A brain declares which shape it needs via `Brain.reports_confidence`.

### Shape A — brain does not self-rate (`reports_confidence = False`)

Used by `LocalFastBrain` (stub) and `NpuFastBrain`. This is the original
flow: score the query from surface features, decide, and only pay for a local
inference if the decision was "stay local".

```
mask ─▶ assert ─▶ heuristic score ─▶ escalate? ─┬─ no ──▶ fast brain ──▶ rehydrate
                                                └─ yes ─▶ cloud ───────▶ rehydrate
```

### Shape B — brain self-rates (`reports_confidence = True`)

Used by `PhoneFastBrain`. Per `src/phone_brain/L_INTERFACE_CONTRACT.md`, L
answers *and* rates its own confidence in **one** call, and O — never L —
applies the threshold. Since the signal arrives attached to the answer, the
brain must be asked before the decision:

```
mask ─▶ assert ─▶ budget pre-check ─┬─ over budget ──▶ "not confident" ─▶ …
                                    │
                                    └─ ok ─▶ fast brain (answer+conf)
                                                    │
                                       confidence parseable?
                                          │              │
                                         yes             no
                                          │              │
                                 difficulty = 1−conf   difficulty = 1.0
                                          │              │
                                          └──── escalate? ────┘
                                            │              │
                                            no             yes
                                            │              │
                                   use that answer   "not confident" ─▶ …
                                       ─▶ rehydrate
```

"Not confident" (via either the budget pre-check or the threshold) is where
Shape B used to always go to the cloud. **It no longer always does** — see
"The escalation brain" below.

Three details in Shape B are load-bearing:

1. **The latency budget is checked *before* the call, never after.** Its job
   is to avoid *starting* an inference that can't finish in time. Re-applying
   it once the answer is in hand would be actively harmful — escalating at
   that point adds the next brain's latency *on top of* local time already
   spent, so it could only make the total worse. If the profiled estimate
   already exceeds the budget, the fast brain is skipped entirely rather than
   called and discarded.
2. **A discarded local answer is still reported.** `RouteDecision.est_latency_ms`
   on a "not confident" decision includes the time spent on the local answer
   that lost, and a note says so. Speculation is not free and the audit trail
   shouldn't pretend it is.
3. **One threshold, not two.** `confidence_estimator.py` carries its own
   `confidence_threshold=0.5` and a `should_escalate` field; O ignores both.
   Confidence is inverted to a difficulty (`signals/confidence.py`) and the
   existing `RoutePolicy.escalate_threshold` decides. This is what keeps the
   contract's "O owns the decision every time" true in code, and it is why
   `routing/policy.py` needed no changes at all — not for Shape B originally,
   and not for the escalation brain either (below).

### `confidence = None` is not `confidence = 0.0` — and neither gets a heuristic

- **`None`** — the model ignored the `CONFIDENCE:` output format. A real,
  expected failure mode of a small quantized model (`PHONE_DEPLOYMENT_GUIDE.md`
  Part 8). O does **not** fall back to a different signal (the surface-feature
  heuristic) for this — it treats "no number" as `difficulty = 1.0`, maximally
  uncertain, and says so in the notes. A brain that formats badly is not the
  same claim as "the surface features say this is hard"; conflating the two
  would score the same query two different ways depending on an unrelated
  formatting accident.
- **`0.0`** — the brain or its transport is reporting it cannot answer
  (timeout, non-200). Per the L contract's error-handling section that is a
  definite escalate, which `1 − 0.0 = 1.0` produces the same value as `None`
  by construction, not by coincidence.

Both land on the same "not confident" path as a low-but-parsed number. What
differs is only the note text, so the audit trail still says *why*.

---

## The escalation brain

**"Not confident" no longer means "the cloud."** When a second, better
*local* opinion is configured — today: mobile's `PhoneFastBrain` escalating
to the AI PC's own `NpuFastBrain` — that model is asked directly instead:

```
"not confident" ─┬─ escalation brain configured? ── yes ──▶ NpuFastBrain(masked_query) ─▶ rehydrate
                 └─ no ─────────────────────────────────▶ CloudDeepBrain (as before)
```

`TwoBrainRouter.escalation_brain` (`_build_escalation_brain` in `router.py`)
is `None` unless **both** `TWO_BRAIN_PHONE_BRAIN=1` and `TWO_BRAIN_NPU_BRAIN=1`
are set on the `mobile` tier — same two flags each brain already used
individually, no third switch invented. It's `None` on the `pc` tier
unconditionally: `pc`'s own fast brain already *is* this model when the flag
is on (`_build_fast_brain`), and this hardware doesn't support two live Genie
sessions at once (`tests/test_npu_brain.py`'s `npu_brain` fixture).

What this buys, and what it costs:

- **Still `tier_answered = "local"`, not a new tier value.** `NpuFastBrain`
  runs in-process on this machine (`docs/npu-deployment.md`) — nothing about
  this path reaches the cloud boundary `CloudDeepBrain` represents, which is
  the invariant that actually matters here. The audit trail (`notes`) still
  says which brain answered; only the enum stayed binary.
- **Two real local inferences on one query, by design.** The phone answers
  (or the budget pre-check skips it), and if that's not confident, the AI PC
  answers too. Nothing here optimizes for latency — `RouteDecision.est_latency_ms`
  bills both, same accounting the discarded-local-answer case already used.
- **Masking still comes first, for every brain.** `route()`'s step 1-2 run
  before any brain is called, so the escalation brain — same as the phone,
  same as the cloud — only ever sees `masked_query.masked_text`.
- **Fails loudly, not silently, without the real hardware/runtime stack.**
  Constructing `NpuFastBrain` without `onnxruntime_qnn` installed (or the
  Genie artifact) raises immediately from `TwoBrainRouter.__init__` — the
  exact same failure mode the `pc` tier's own real brain already has, not a
  new one. There is no silent fallback to a stub here; if you set both flags,
  you need the real stack.

Cleanup: `TwoBrainRouter.close()` closes `fast_brain` and `escalation_brain`,
whichever are real — `api.py`'s `serve()` calls it in `finally`, and anything
constructing a router directly should too.

---

## Per-tier signal

| Tier | Fast brain | "Not confident" goes to | Transport | Self-rates? | Difficulty signal |
|---|---|---|---|---|---|
| AI PC (`pc_3b`) | `NpuFastBrain` — Phi-3.5-mini-instruct on Hexagon NPU | Cloud (Shape A has no escalation brain) | in-process `ctypes`/Genie | No | surface-feature heuristic |
| Mobile (`mobile_1b`) | `PhoneFastBrain` — Llama-3.2-3B on a Galaxy S25 | **The AI PC's `NpuFastBrain`** (if `TWO_BRAIN_NPU_BRAIN=1`), else cloud | HTTP to loopback (`adb reverse`) | **Yes** | model's own self-report |
| Cloud | `CloudDeepBrain` (stub) | — | — | No | n/a — escalation target |

`NpuFastBrain` does not self-rate for two concrete reasons: Genie exposes no
logprobs through the C API it uses, and adding a self-report suffix would
change the prompt that the real numbers in `data/profile_workload/pc_3b.json`
were measured against. Giving it Shape B is a real option — it needs a
prompt change plus a re-profile, not a router change.

---

## Running it

```powershell
# AI PC tier, real NPU brain (needs .venv-npu + the Genie artifact)
$env:TWO_BRAIN_NPU_BRAIN=1
.venv-npu\Scripts\python.exe -m two_brain_router --tier pc

# Mobile tier, real phone brain -- against the mock, no phone needed:
python src\phone_brain\mock_phone_brain_server.py --port 8000
$env:TWO_BRAIN_PHONE_BRAIN=1
.venv\Scripts\python.exe -m two_brain_router --tier mobile

# Mobile tier, phone + AI PC as the escalation brain (needs .venv-npu +
# the Genie artifact -- see "The escalation brain" above):
python src\phone_brain\mock_phone_brain_server.py --port 8000
$env:TWO_BRAIN_PHONE_BRAIN=1
$env:TWO_BRAIN_NPU_BRAIN=1
.venv-npu\Scripts\python.exe -m two_brain_router --tier mobile
```

Against the real device, the only change is that the server is the phone
(`adb reverse tcp:8000 tcp:8000` first) — which is the property
`L_INTERFACE_CONTRACT.md` exists to guarantee.

| Env var | Default | Meaning |
|---|---|---|
| `TWO_BRAIN_NPU_BRAIN` | unset | `1` enables the real NPU brain (pc tier) |
| `TWO_BRAIN_PHONE_BRAIN` | unset | `1` enables the real phone brain (mobile tier) |
| `TWO_BRAIN_PHONE_URL` | `http://127.0.0.1:8000` | where L is served |
| `TWO_BRAIN_PHONE_MODEL` | `llama-3.2-3b-instruct` | model id sent in the request |
| `TWO_BRAIN_PHONE_ALLOW_REMOTE` | unset | `1` permits a non-loopback L (see below) |

Both brains are **off by default**, so the base package stays stdlib-only and
the test suite never depends on hardware or a served endpoint being up.

### Use `127.0.0.1`, not `localhost`

Measured on this machine, same server and prompt: **`localhost` 2778–3117 ms
vs. `127.0.0.1` 742–1153 ms.** `localhost` resolves to `::1` first, the
server binds IPv4 only, and the failed IPv6 attempt costs ~2 s before falling
back.

Elsewhere that is an annoyance; here it is a correctness bug. The budget is
3000 ms and the router decides local-vs-cloud on this exact number, so the
phantom 2 s escalates queries the phone could comfortably have answered. Hence
the `127.0.0.1` default.

### The on-device host check

`PhoneFastBrain` refuses a non-loopback base URL unless `allow_remote=True`.
The router only ever hands it *masked* text, so this is not the thing standing
between the user and a leak — but the difference between "the model runs on my
phone" and "the model runs on someone's server" is one typo in a plain string,
and it is a different privacy posture than this project advertises. It takes a
deliberate opt-in rather than a silent default.

---

## The chat UI

`ui/` (merged from `origin/(hollowbyte)-feat/chat_app`) is a plain HTML/JS
chat interface, now wired to a real `/route` endpoint
(`src/two_brain_router/api.py`) instead of its original mock. See
`ui/README.md` for how to run both pieces and how the offline fallback
works (the API is optional -- the UI still demos cleanly without it, just
clearly labeled as a preview). Verified end-to-end in a real browser: live
routing, PII masking visible in the profiler, cloud escalation, offline
fallback, and recovery after a backend restart, all without repainting
chat history that already rendered under a different state.

---

## What this changed, and what it didn't

**Unchanged:** all four invariants in `../CLAUDE.md`, `routing/policy.py`
(untouched — as `WALKTHROUGH.md` next-step #4 asked, and still true after
the escalation brain), and the `--tier pc` demo transcript in the README
(byte-identical). The original nine tests still pass unmodified; two mobile
confidence-path tests were rewritten (not the original nine) when the
heuristic fallback was removed — see below.

**Changed:** `BrainResponse` gained optional `confidence` and `error` fields;
`Brain` gained `reports_confidence`; `route()` grew the Shape B branch, then
grew the escalation-brain branch inside it. `TwoBrainRouter` gained
`escalation_brain` and a `close()` that releases both real brains.

**Removed, not just changed:** the surface-feature heuristic fallback for an
unparseable confidence (`signals/difficulty.py`'s `DifficultyEstimator`) is
gone from the mobile confidence path — an unparseable confidence is now
`difficulty = 1.0`, full stop, not a second signal. `DifficultyEstimator`
itself is unaffected and still used for Shape A (the `pc`/stub tiers).

**Worth knowing:** on the mobile tier with a confident fast brain, the demo's
PII query now stays **local** — where the surface heuristic escalated it. That
is the archetype working as intended: it is a genuinely easy request, and the
better signal keeps the PII on-device entirely instead of masking it and
shipping it to the cloud. `WALKTHROUGH.md` predicted this exact misfire
("query 3 scores 0.40 for a genuinely easy request"). With an unconfident
phone and the escalation brain configured, that same PII stays on-device even
further — it never reaches the cloud at all, since the AI PC answers instead.

**Mock artifact, not a bug:** `mock_phone_brain_server.py` echoes the first 60
characters of the prompt back, which now includes part of the self-report
instruction, so its canned answers trail off mid-sentence with "After
answering, on a new line". A real model does not echo its instructions. Only
the `CONFIDENCE: <n>` line itself is parsed and stripped.
