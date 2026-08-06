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
mask ─▶ assert ─▶ budget pre-check ─┬─ over budget ──────────────▶ cloud ─▶ rehydrate
                                    │
                                    └─ ok ─▶ fast brain (answer+conf)
                                                    │
                                       difficulty = 1 − confidence
                                                    │
                                            escalate? ─┬─ no ──▶ use that answer ─▶ rehydrate
                                                       └─ yes ─▶ discard it, cloud ─▶ rehydrate
```

Three details in Shape B are load-bearing:

1. **The latency budget is checked *before* the call, never after.** Its job
   is to avoid *starting* an inference that can't finish in time. Re-applying
   it once the answer is in hand would be actively harmful — escalating at
   that point adds cloud latency *on top of* local time already spent, so it
   could only make the total worse. If the profiled estimate already exceeds
   the budget, the fast brain is skipped entirely rather than called and
   discarded.
2. **A discarded local answer is still reported.** `RouteDecision.est_latency_ms`
   on an escalation includes the time spent on the local answer that lost, and
   a note says so. Speculation is not free and the audit trail shouldn't
   pretend it is.
3. **One threshold, not two.** `confidence_estimator.py` carries its own
   `confidence_threshold=0.5` and a `should_escalate` field; O ignores both.
   Confidence is inverted to a difficulty (`signals/confidence.py`) and the
   existing `RoutePolicy.escalate_threshold` decides. This is what keeps the
   contract's "O owns the decision every time" true in code, and it is why
   `routing/policy.py` needed no changes at all.

### `confidence = None` is not `confidence = 0.0`

- **`None`** — the model ignored the `CONFIDENCE:` output format. A real,
  expected failure mode of a small quantized model (`PHONE_DEPLOYMENT_GUIDE.md`
  Part 8). O falls back to the surface-feature heuristic and says so in the
  notes rather than inventing a number.
- **`0.0`** — the brain or its transport is reporting it cannot answer
  (timeout, non-200). Per the L contract's error-handling section that is a
  definite escalate, which `1 − 0.0 = 1.0` produces naturally.

Collapsing the two would either silently force-escalate every query a model
formats badly, or silently trust a value that was never reported.

---

## Per-tier signal

| Tier | Fast brain | Transport | Self-rates? | Difficulty signal |
|---|---|---|---|---|
| AI PC (`pc_3b`) | `NpuFastBrain` — Phi-3.5-mini-instruct on Hexagon NPU | in-process `ctypes`/Genie | No | surface-feature heuristic |
| Mobile (`mobile_1b`) | `PhoneFastBrain` — Llama-3.2-3B on a Galaxy S25 | HTTP to loopback (`adb reverse`) | **Yes** | model's own self-report |
| Cloud | `CloudDeepBrain` (stub) | — | No | n/a — escalation target |

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
(untouched — as `WALKTHROUGH.md` next-step #4 asked), the nine original tests
(passing unmodified), and the `--tier pc` demo transcript in the README
(byte-identical).

**Changed:** `BrainResponse` gained optional `confidence` and `error` fields;
`Brain` gained `reports_confidence`; `route()` grew the Shape B branch.

**Worth knowing:** on the mobile tier with a confident fast brain, the demo's
PII query now stays **local** — where the surface heuristic escalated it. That
is the archetype working as intended: it is a genuinely easy request, and the
better signal keeps the PII on-device entirely instead of masking it and
shipping it to the cloud. `WALKTHROUGH.md` predicted this exact misfire
("query 3 scores 0.40 for a genuinely easy request").

**Mock artifact, not a bug:** `mock_phone_brain_server.py` echoes the first 60
characters of the prompt back, which now includes part of the self-report
instruction, so its canned answers trail off mid-sentence with "After
answering, on a new line". A real model does not echo its instructions. Only
the `CONFIDENCE: <n>` line itself is parsed and stripped.
