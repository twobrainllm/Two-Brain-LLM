# The Orchestrator

How `TwoBrainRouter` decides, now that both local tiers have a real brain
behind them.

There is no separate orchestrator component to build — **`TwoBrainRouter`
(`src/two_brain_router/routing/router.py`) is O.** The phone-brain docs refer
to L (fast brain), O (orchestrator), and C (deep brain); the mapping is:

| Contract name | This codebase |
|---|---|
| **L** — fast brain | `NpuFastBrain` / `GpuLocalBrain` (AI PC, either real) / `PhoneFastBrain` (Mobile, real) / `LocalFastBrain` (stub) |
| **O** — orchestrator | `TwoBrainRouter` + `RoutePolicy` |
| **C** — deep brain | `CirrascaleDeepBrain` (real) / `CloudDeepBrain` (stub — gap #4 unaffected either way) |

---

## The three decision shapes

The privacy ordering is: **detect → answer on-device → mask at the boundary →
assert → send → rehydrate.** What changes per tier is *where the difficulty
signal comes from*, and that determines whether the fast brain is called before
or after the decision — and, in Shape C, whether the decision is "which brain"
at all.

> **The ordering changed.** It used to be *mask → assert → decide → answer →
> rehydrate*: the query was masked before anything touched it, including the
> local model. That was the wrong place for it. The AI PC's fast brain executes
> on this machine, so masking its input bought no privacy and measurably cost
> answer quality — a model asked to draft a reply to `[PII_EMAIL_1]` writes a
> worse reply than one that can see the address. See **The trust boundary**
> below.

A brain declares which shape it needs with two flags:

| | `reports_confidence` | `reports_gaps` | Brains |
|---|---|---|---|
| **Shape A** | `False` | `False` | `LocalFastBrain`, `CloudDeepBrain` (the stubs) |
| **Shape B** | `True` | `False` | `PhoneFastBrain` |
| **Shape C** | `True` | `True` | `NpuFastBrain`, `GpuLocalBrain` (both AI-PC, both default-on) |

`reports_gaps` implies `reports_confidence` and never the reverse — a brain
that can say *which part* it couldn't do can necessarily say *how sure* it is.

---

## The trust boundary

A third flag, `Brain.trusted_with_raw_pii`, decides something more important
than any of the above: **what the `query` argument to `answer()` actually
contains.**

| Brain | Trusted | Why |
|---|---|---|
| `NpuFastBrain` | **Yes** | in-process, `ctypes` → `Genie.dll`, this machine's NPU |
| `GpuLocalBrain` | **Yes** | a `llama-server` child process this class started, bound to `127.0.0.1` |
| `LocalFastBrain` (stub) | **Yes** | in-process; a stub that got different input from the real thing would make the default path a bad rehearsal |
| `PhoneFastBrain` | **No** | a physically separate device. `adb reverse` makes the hop *look* like loopback, which is exactly why this is declared and not inferred from the URL |
| `CloudDeepBrain` / `CirrascaleDeepBrain` | **No** | the boundary itself |

Trusted brains get the query **exactly as the user typed it**. Everything else
gets masked text, and every string that leaves is masked immediately before it
does — via `_Request.mask_for_boundary`, which pairs the mask with the fatal
invariant assert. Every path to `deep_brain.answer` goes through that one
method, so a missing mask/assert is greppable rather than a matter of reading
carefully.

The flag is read as `getattr(brain, "trusted_with_raw_pii", False)`. **A brain
that forgets to declare it gets masked input**, because forgetting to opt *in*
costs answer quality while forgetting to opt *out* would leak.

### Watching it happen: the trace

`RouteDecision.notes` says *what the router decided and why*. The trace
(`src/two_brain_router/trace.py`) prints **the payloads** — the exact strings
handed to each brain and returned by each one. That distinction matters here:
"did the raw address reach the cloud?" is not answerable from a note reading
"masked 3 entities", only from seeing the bytes.

It is **on by default when you run `python -m two_brain_router.api`**
(`--no-trace` to silence it) and opt-in for the CLI (`--trace`; off under
`--json`, and off by default so the README's pinned transcript stays exact).
Real output, abridged:

```
- CALL  fast brain | NpuFastBrain   [ON-DEVICE -- raw text]
  query      : My email is jane.doe@example.com -- ...tell me the exact ISBN of the 1813 first edition...
- RETURN  NpuFastBrain  5871ms
  text       : Your SSN 123-45-6789 was found in a backup, and the exact ISBN ... is [ISBN here].
  confidence : 0.90
  unknown    : The exact ISBN of the 1813 first edition of Pride and Prejudice
- BOUNDARY  -> CloudDeepBrain   [OFF-DEVICE -- masked]
    masked  'jane.doe@example.com' -> [PII_EMAIL_1]
    masked  '123-45-6789' -> [PII_SSN_1]
  context    : A smaller on-device model has already answered part of this question:
               Your SSN [PII_SSN_1] was found in a backup, ...
- REHYDRATE  2 placeholder(s) restored on-device
- DECISION  hybrid  difficulty=0.10  7350ms  $0.00004
  PII        : 2 detected -- 2 masked before leaving
```

The `[ON-DEVICE]` / `[OFF-DEVICE]` banners are driven by the `crossing` flag at
the call site, not guessed from the brain's type. Note the same sentence
appearing twice: the local model wrote `Your SSN 123-45-6789`, and what crossed
says `Your SSN [PII_SSN_1]`. That is Shape C's masking of the partial answer,
visible.

Two implementation notes worth keeping:

- **Every `answer()` call in the router goes through `_ask`**, which is what
  emits the trace. A new branch that calls a brain directly would silently drop
  out of it — and the trace is how anyone verifies the privacy claim.
- **ASCII only.** Box-drawing characters crashed the server outright on a
  Windows console (`cp1252` cannot encode them, and `print` raises). A trace
  that kills the process it is tracing is worse than a plain-looking one.

**The trace prints raw PII**, necessarily — you cannot verify that PII stayed
on-device without seeing it was there. It goes to stdout, never to a file, and
`api.py` says so at startup rather than letting someone find out mid-screenshare.

### Showing the local half first: `/route/stream`

A split has two very differently-priced halves. Measured on real hardware with
the real Cirrascale endpoint, on one query:

```
[  9.80s] PROGRESS  local : "Pride and Prejudice is a novel set in the English countryside..."
                    gap   : "the exact ISBN of the 1813 first edition"
[ 21.00s] RESULT    tier=hybrid  cost=$0.000275
```

The local answer was finished and displayable at 9.8s; plain `POST /route`
holds it until 21.0s and shows nothing for the intervening 11 seconds. So
there is a second endpoint that hands it over as soon as it exists:

| | `POST /route` | `POST /route/stream` |
|---|---|---|
| Response | one JSON object | NDJSON, one object per line |
| Local half arrives | with everything else | as soon as the local model finishes |
| Final result | — | last line, **byte-identical** to `/route`'s body |

Lines are `{"type": "progress"|"result"|"error", ...}`. There is at most one
`progress` line, and only when something is about to cross the boundary:

- `phase: "local_answer"` — a usable partial exists and the deep brain is
  about to be asked about the named gap. Carries `local_answer` (rehydrated,
  display-ready) and `gap`.
- `phase: "escalating"` — nothing usable came back locally, so the whole query
  is going to the cloud. `local_answer` and `gap` are `null` — *"there is no
  partial"* and *"the partial was blank"* are different states and the UI says
  different things for them. Emitted anyway, because "escalating, nothing
  usable locally" is a much better thing to show for 15s than a bare spinner.

A locally-answered query emits **no** progress line at all: nothing is being
waited for, so there is nothing to announce.

Underneath, `route()` takes an optional `on_progress` callback and calls it
once, at the moment the local half is settled and rehydrated but before the
deep brain is asked. Four properties are pinned by tests because each is a way
this could be quietly wrong:

- **Ordering is the whole feature**, so it is asserted on ordering, not on the
  callback merely firing — `test_the_local_half_is_handed_out_before_the_cloud_is_called`
  checks the deep brain's call log is still empty *inside* the callback, and
  `test_stream_hands_out_the_local_half_before_the_cloud_call_finishes` uses a
  deliberately slow fake cloud so "arrived early" is measured rather than
  inferred. A version that emitted both lines at the end would pass every
  other test.
- **`local_answer` is rehydrated**, so a UI can paint it directly without a
  `[PII_EMAIL_1]` reaching the screen. Safe to do early because that text came
  from a brain given `request.view`, whose vault is complete before the cloud
  call and does not depend on anything the cloud returns.
- **The final line equals `/route`'s body.** Streaming changes *when* you learn
  things, never *what* — if those diverge, the UI is showing something the
  audit trail doesn't.
- **A callback that raises cannot break the request.** The realistic cause is a
  client disconnecting mid-stream; that must not turn a working answer into a
  500 or abandon a cloud call already paid for. Exceptions are swallowed: a
  dead listener is not a routing failure.

`ui/app.js` uses this by default and falls back to `/route` on a 404, so a UI
newer than its server degrades to *slower* rather than to the offline preview
(which would wrongly claim nothing ran). The partial is painted into the live
DOM row only — never into `chat.messages` — so a reload can't restore a
half-finished answer as if it were real.

### Conversation history

`route(query, context)` has always had a `context` argument, but it only ever
fed the *cloud* — every `_ask(self.fast_brain, …)` call omitted it. So the local
model received a follow-up like "explain in more detail" as a standalone
sentence with no referent, and answered something unrelated. It now gets the
history too.

**The router stays stateless, deliberately.** `api.py` builds one
`TwoBrainRouter` and shares it across every request and every browser tab (the
NPU's Genie session is far too expensive to rebuild per request), so history
held there would leak between unrelated chats. The client owns it — `ui/app.js`
assembles recent turns from `chat.messages` and sends them as `context` on each
request.

Where it goes follows the same trust rule as everything else:

| | Gets history |
|---|---|
| On-device fast brain (`trusted_with_raw_pii`) | **raw**, via `_Request.context_view` |
| Off-device fast brain (`PhoneFastBrain`) | masked, same guard as the query |
| Cloud, on an escalation | masked and compressed into `context`; the gap is the `query` and is never truncated |

Bounded in `ui/app.js` (6 messages / 300 chars each / 1200 total, newest-first)
because an unbounded history costs three ways: the local model's prompt grows
against a finite compiled context length, its 4–12s latency grows with it, and
every extra turn is more text eligible to leave the device. `compress_context`
caps the cloud side at 800 chars regardless.

Two bugs this turned up, both fixed and both regression-tested:

- **A placeholder minted from the history didn't rehydrate.** An answer can echo
  `[PII_EMAIL_1]` that came from the *context* rather than the query, so
  rehydration now merges both vaults (`_Request.local_vault`). Safe to merge
  because one `PIIGuard` masked both and maps a repeated value to one
  placeholder.
- **The audit trail undercounted.** `pii_entities_detected` and
  `pii_entities_masked` were computed from the query alone, so a request whose
  PII lived only in the history reported *"0 detected / 0 masked"* while the
  trace plainly showed an address being masked at the boundary. The masking was
  correct; the reporting wasn't — and on a privacy demo the profiler saying "No
  PII detected" about a request that shipped a masked address is the worse
  failure. Both counts now cover query + context, de-duplicated by value so an
  address written in both places reads as one entity rather than as
  "2 detected, 1 masked".

**Not yet verified:** whether Phi-3.5-mini actually *uses* the history it is now
given. The prompt plumbing is confirmed (the rendered prompt contains the prior
turns), but the model's use of it needs a run on the real NPU.

### Token-level streaming: `/route/sse`

`/route/stream` hands over the local half when it is *complete*. `/route/sse`
hands over each token as it is generated, so a bubble fills in like a chat
app rather than appearing all at once. Measured on the real NPU:

```
first visible text at  0.83s
stream ended at        4.15s
```

A 3.3-second head start on a 4-second answer, and proportionally larger on the
long ones — this model has taken 11s+ on a single reply.

**The hard part is that the local model does not generate prose.** It generates
`{"solution": "...", "confidence": 90, "unknown": "..."}`, one token at a time.
The first raw token off the real NPU is literally `` {" ``. Forwarding that
would show a brace, a quoted key and a colon before any answer, then routing
metadata after it. `signals/structured.SolutionStreamer` walks the growing
buffer and releases **only the decoded contents of `solution`**, while keeping
the raw text intact for the real parse at the end.

> The `(hollowbyte)-feat/chat_app` branch declined to stream the local half for
> exactly this reason — *"it arrives as JSON — streaming it would show the user
> the scaffolding."* Solving it is what makes token streaming usable on the
> tier where it matters most, since the on-device model is the slow one.

Three cases the streamer has to survive, all pinned in
`tests/test_solution_streaming.py`:

- **Chunk boundaries fall anywhere.** `"solu` / `tion": "Par` / `is"` is normal,
  so nothing can be matched against a single chunk; the whole buffer is
  rescanned each feed.
- **A `\uXXXX` escape split in half is withheld**, not guessed at. There is no
  edit in a stream, only append — a broken character can never be taken back.
- **A model that emits no JSON at all streams verbatim.** `parse_structured`
  already degrades to prose; the stream degrades the same way, so a
  badly-behaved model produces a badly-formatted answer rather than a blank
  screen.

Frames are SSE, named rather than type-tagged:

| Event | When | Payload |
|---|---|---|
| `meta` | once, before any text | `{tier, streaming}` |
| `delta` | repeatedly | `{text, tier}` — **tier-attributed**, so a split renders as two bubbles from the data rather than from ordering |
| `tier` | a split opening the cloud's bubble | `{tier, gap}` |
| `done` | once | the full `RouteDecision` plus `/route`'s display extras |
| `error` | in-band | the status line is long sent, so a failure cannot be an HTTP code |

`EventSource` is unusable here — it is GET-only and this request carries a JSON
body with an optional image — so `ui/app.js` reads the `fetch` body as a stream
and parses frames itself.

**Streaming changes nothing about the privacy ordering.** Detection, the local
view, and (for an escalated image) describing it on-device and masking that
description all happen before the first delta; only *generation* is
incremental. `_answer_hybrid` and `_answer_hybrid_streaming` share one
`_prepare_gap_escalation`, so the masking cannot differ between them — two
copies of privacy-critical code is how one ends up a fix behind. Deltas are
rehydrated per-frame so a `[PII_EMAIL_1]` never reaches the screen.

A brain that cannot stream (`streams_tokens`), or a shape this does not
implement (Shape B, which discards its local answer when it routes away — so
streaming it would mean showing text about to be retracted), falls back to one
`answer()` call emitted as a single delta. The client renders both identically.

### Image attachment

`POST /route` and `/route/sse` accept `image` as a `data:image/...;base64,...`
URL. `api.py` decodes it to a temp file, hands the `Path` to `route()`, and
deletes it in a `finally` that covers every exit — including the
invariant-violation path, which returns early. An image is the most sensitive
thing a user can hand this system; a copy left in the temp directory after an
error is exactly the quiet residue this project exists to avoid.

Capped at `MAX_IMAGE_CHARS` (~3.4 MB of image bytes) and checked client-side
too, so an oversized file is refused before it is base64'd and pushed over the
wire. An unbounded body is a trivial memory-exhaustion lever on a loopback
server that deliberately has no other auth.

The image itself **never leaves the machine** — that is not a policy choice,
it is forced: the cloud tier has no vision model at all
(`data/cloud_ai100/_real_endpoint_log.md`). An image-bearing query that
escalates is described on-device first, and only that masked description
crosses.

### Detected ≠ masked

`RouteDecision` carries both counts, and it has to. A PII-heavy query answered
entirely on-device now reports `pii_entities_masked = 0` — which is the *good*
outcome, not a missing measurement, but reads as "no PII here" on its own.
`pii_entities_detected` is what the query contained regardless, and the gap
between the two numbers is the demo: *2 detected — none left the device.*

### What this costs

The routing decision is now made on raw text. That is acceptable precisely
because the thing making it is on-device: the difficulty heuristic is a pure
local function, and the confidence/gap signals come from the local model, which
was already trusted with the query by the time it produced them. Nothing about
the decision is transmitted.

The real cost lands in Shape C, and it is worth being blunt about: because the
local model sees raw PII, **its `solution` and `unknown` can contain raw PII
too** — a real address, not a placeholder. Those strings cross the boundary on
a split. Masking them there is not defence-in-depth; it is the only thing
standing between the user and a leak, because query-level masking never saw
them.

### Two thresholds, not one

`RoutePolicy` carries two OR-triggers that both read as "the query is too hard,
escalate" and are **not interchangeable**:

| | `escalate_threshold` | `confidence_escalate_threshold` |
|---|---|---|
| Feeds | `signals/difficulty.py`'s heuristic (Shape A) | a brain's own self-report (Shape B/C) |
| Scale | arbitrary 0–1, weighted keyword/length features | `1 - confidence`, direct from the model |
| Default | 0.55 | 0.10 |
| In words | escalate below ~45% heuristic confidence | escalate at 90% reported confidence or below |

They used to be one field, deliberately (`needs_gap_fill`'s original docstring:
*"the same `escalate_threshold` every other path uses. Still one threshold, not
two."*). That held until real numbers showed the two signals don't share a
scale: Phi-3.5/Qwen self-reports measured in a narrow, over-confident band,
0.85–1.00 (see "Confidence is weak on this tier" below), so 0.55 as a
*difficulty* threshold — confidence ≤ 0.45 — essentially never fires against
that band. The symptom was "it never escalates to the cloud." Lowering the
**shared** field to fix that would have also made the heuristic path (image
queries, the stub-only default demo, the pinned README transcript) escalate on
almost everything — a difficulty scale tuned separately has no reason to share
a boundary with a confidence scale.

So `confidence_escalate_threshold` is its own field, read only by
`confidence_says_escalate`, `needs_gap_fill`, and `_route_on_confidence` — the
Shape A heuristic path in `route()` still reads `escalate_threshold` alone,
unchanged, which is why the README's pinned `--tier pc` transcript (stub
brains, Shape A) stayed byte-identical across this change.

**A real bug found getting here, worth keeping in mind for any future
threshold:** `confidence_to_difficulty` used to return `1.0 - confidence`
un-rounded. `1.0 - 0.90` in IEEE 754 is `0.09999999999999998`, not `0.1` — so a
threshold set to exactly `0.10` to mean *"confidence 0.90 or below escalates"*
let `0.90` itself silently through, because `0.09999999999999998 >= 0.10` is
`False`. `confidence_to_difficulty` now rounds to 6 decimal places, which
removes the binary-float artifact while keeping far more precision than a
2-decimal-place self-report (`n/100` for integer `n`) ever carries. Any
threshold landing on a round confidence value is exposed to this; round at the
conversion, not per comparison site, or the next threshold rediscovers it.

### Shape A — brain does not self-rate (`reports_confidence = False`)

Used by the stubs, `LocalFastBrain` and `CloudDeepBrain` -- i.e. the default
stdlib-only path with no real brain configured. `NpuFastBrain` used to be a
Shape A example too; it moved to Shape B (below) once it started self-rating
-- see "Per-tier signal" further down. This is the original flow: score the
query from surface features, decide, and only pay for a local inference if
the decision was "stay local".

```
mask ─▶ assert ─▶ heuristic score ─▶ escalate? ─┬─ no ──▶ fast brain ──▶ rehydrate
                                                └─ yes ─▶ cloud ───────▶ rehydrate
```

### Shape B — brain self-rates (`reports_confidence = True`)

Used by both real brains: `PhoneFastBrain` (mobile) and `NpuFastBrain` (AI
PC) -- see "Per-tier signal" below for when the latter moved here. Per
`src/phone_brain/L_INTERFACE_CONTRACT.md`, L answers *and* rates its own
confidence in **one** call, and O — never L — applies the threshold. Since
the signal arrives attached to the answer, the brain must be asked before the
decision:

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
3. **One *owner*, not two — `L_INTERFACE_CONTRACT.md`'s claim, still true.**
   `confidence_estimator.py` carries its own `confidence_threshold=0.5` and a
   `should_escalate` field; O ignores both entirely and decides for itself.
   This is what keeps the contract's "O owns the decision every time" true in
   code. (Not the same claim as "one *number*" — see "Two thresholds, not one"
   further down, where O's own decision later grew a second number for a
   second kind of signal. O still never defers to L's opinion of itself.)

### `confidence = None` is not `confidence = 0.0` — and neither gets a heuristic

- **`None`** — the model ignored the `CONFIDENCE:` output format. A real,
  expected failure mode of a small quantized model (`PHONE_DEPLOYMENT_GUIDE.md`
  Part 8), and observed on the AI PC tier too: one of eight real queries came
  back `'Jane Austen, 95'`, the number supplied but the label dropped. O does
  **not** fall back to a different signal (the surface-feature
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

### Shape C — brain names its gaps (`reports_gaps = True`)

Used by both AI-PC brains. **This is the flow the AI PC actually runs today.**

Shapes A and B both answer the same question — *which brain handles this?* —
and throw one brain's work away. Shape C stops asking that. The local model is
asked to solve what it can *and to write down the part it can't*, in one call,
as JSON:

```json
{"solution": "Paris, 2.1 million", "confidence": 95,
 "unknown": "The specific population on 3 March 2019"}
```

The `solution` is kept and shown. The `unknown` is what the deep brain is
asked about — a second, **narrower** call, not a redo:

```
mask ─▶ assert ─▶ ceiling pre-check ─▶ fast brain (solution + confidence + gap)
                                                 │
                                    needs_gap_fill(difficulty, gap)?
                                        │                    │
                                       no                   yes
                                        │                    │
                                   tier=local        usable solution?
                                   ─▶ rehydrate        │          │
                                                      yes         no
                                                       │          │
                                              mask the gap    tier=cloud
                                              + the partial   (ordinary
                                                       │      escalation)
                                              deep brain(gap)
                                                       │
                                              tier=hybrid, merge
                                                 ─▶ rehydrate
```

Four things here are load-bearing:

1. **A named gap escalates on its own, whatever the confidence says.**
   `RoutePolicy.needs_gap_fill` ORs "gap is non-empty" with the usual
   threshold. This is the whole point. Measured on real hardware: *"capital of
   France, and its population on 3 March 2019"* self-rates **0.95** →
   difficulty 0.05, far below the 0.55 threshold. Under Shape B it stays local
   and answers "Paris", silently dropping half the question. The model knew the
   other half was missing all along; there was nowhere to say so.
2. **The gap is the question asked, not a note attached to one.** The deep
   brain's `query` is the masked gap; the original query, history and partial
   answer are background in `context`. This is a regression-tested fix for a
   real refusal — with the whole query as the ask and the gap merely mentioned
   in context, a live Cirrascale call answered the *entire* placeholder-laden
   request and declined it:

   > "I cannot provide you with a reply that includes your personal information."

   …when all that was needed was an ISBN. Asking narrowly also makes the gap
   structurally immune to `compress_context`, which only ever trims `context`.

3. **The gap and the partial are masked before they cross — the sharpest edge
   here.** Both are *newly generated* text the guard has never seen, written by
   a model that was handed the **raw** query, so they can quote a real address
   verbatim. Query-level masking cannot help: these strings did not exist when
   the query was read. Both go through `mask_for_boundary`, and their vaults
   are merged for rehydration.
   `tests/test_structured_routing.py::test_the_raw_partial_answer_is_masked_before_it_crosses`
   and `…::test_pii_the_local_model_invented_in_the_gap_is_masked_before_it_crosses`
   are the regression tests.
4. **A split needs two halves.** No usable `solution` (empty, or the brain
   errored) means this is an ordinary escalation and is reported as
   `tier_answered="cloud"`, not as a split with one side missing.
5. **The latency ceiling is a different number here** — see below.

#### Why Shape C has its own latency ceiling

Shape B skips the fast brain when the profiled estimate exceeds
`local_latency_budget_ms` (3000 ms), on the stated grounds that *"a local answer
would have been discarded anyway"*.

**That premise is false in Shape C**, where a usable local answer is always
kept. Reusing the budget measurably broke the feature: on the first real
hardware run, two of the four demo queries never reached the fast brain,
including the PII query — which the local model then turned out to answer
*fully* on-device. Escalating it sent masked PII to the cloud for nothing.

So Shape C uses `RoutePolicy.local_partial_budget_ms` (15000 ms), a runaway
guard rather than a preference, sized at ~2.3x the slowest measured structured
call. A query between the two numbers is answered locally anyway, with a note
in the audit trail saying so. `local_latency_budget_ms` is unchanged for
Shapes A and B.

#### `tier_answered="hybrid"`

A third value, not a flavour of `"cloud"`. For *"did anything cross the
boundary?"* treat it exactly like `"cloud"` — it did. What it adds is that
something also **didn't**, and `RouteDecision` carries the split:
`local_answer`, `cloud_answer`, `gap` (all rehydrated), with `answer` as the
merged text.

#### What Shape C does not do yet

**Images.** `route(query, context, image)` still takes the Shape A heuristic
path whenever an image is present, even on a Shape C brain. Neither Shape B nor
C has anywhere to *put* an image — both call `answer(masked_text)` with no image
argument — so routing one through would silently drop it and answer the text
alone. Image behaviour is therefore exactly as it was. Extending Shape C to
images is the deliberate next step.

**The phone.** `PhoneFastBrain` is untouched and stays Shape B, so the mobile
tier behaves exactly as before.

---

## The escalation brain

**"Not confident" no longer means "the cloud."** When a second, better
*local* opinion is configured — today: mobile's `PhoneFastBrain` escalating
to whichever real AI-PC brain this machine is running — that model is asked
directly instead:

```
"not confident" ─┬─ escalation brain configured? ── yes ──▶ escalation_brain(masked_query) ─▶ rehydrate
                 └─ no ─────────────────────────────────▶ CloudDeepBrain / CirrascaleDeepBrain (as before)
```

`TwoBrainRouter.escalation_brain` (`_build_escalation_brain` in `router.py`)
does **not** hardcode `NpuFastBrain`. It delegates to
`_build_fast_brain("pc", ...)` — the exact function the `pc` tier itself
uses to pick its own fast brain — and uses whatever comes back, as long as
it isn't the stub. Concretely, on the `mobile` tier with
`TWO_BRAIN_PHONE_BRAIN=1`:

| `TWO_BRAIN_GPU_BRAIN` | `TWO_BRAIN_NPU_BRAIN` | `escalation_brain` |
|---|---|---|
| `1` | `1` | `GpuLocalBrain` — GPU wins, same preference `_build_fast_brain` has for its own tier |
| `1` | unset | `GpuLocalBrain` |
| unset | `1` | `NpuFastBrain` |
| unset | unset | `None` — falls back to the cloud, exactly as before |

This is deliberate, not incidental: the AI-PC tier's own fast-brain choice
and the mobile tier's escalation target must never be able to drift apart,
and delegating means a third AI-PC backend (or a change to which one wins)
is automatically correct here too, with no escalation-brain-specific edit.
`tests/test_orchestrator.py::test_escalation_brain_prefers_gpu_over_npu_when_both_are_configured`
pins this.

It's `None` on the `pc` tier unconditionally: `pc`'s own fast brain already
*is* this model when the flag is on (`_build_fast_brain`), and — specifically
for `NpuFastBrain` — this hardware doesn't support two live Genie sessions at
once (`tests/test_npu_brain.py`'s `npu_brain` fixture).

What this buys, and what it costs:

- **Still `tier_answered = "local"`, not a new tier value.** Both
  `NpuFastBrain` (Genie, in-process `ctypes`) and `GpuLocalBrain`
  (`llama-server`, loopback HTTP) run on this machine — nothing about this
  path reaches the cloud boundary `CloudDeepBrain`/`CirrascaleDeepBrain`
  represents, which is the invariant that actually matters here. The audit
  trail (`notes`) still says which brain answered; only the enum stayed
  binary.
- **Two real local inferences on one query, by design.** The phone answers
  (or the budget pre-check skips it), and if that's not confident, the AI PC
  answers too. Nothing here optimizes for latency — `RouteDecision.est_latency_ms`
  bills both, same accounting the discarded-local-answer case already used.
- **Masking still comes first, for every brain.** `route()`'s step 1-2 run
  before any brain is called, so the escalation brain — same as the phone,
  same as the cloud — only ever sees `masked_query.masked_text`. No image can
  reach it either: only `GpuLocalBrain.for_vision()` instances `can_see`, and
  nothing in the escalation path passes an image (mobile's `PhoneFastBrain`
  isn't vision-capable, so `route()` never has one to forward — see "Image
  routing" below).
- **Fails loudly, not silently, without the real hardware/runtime stack.**
  Constructing `NpuFastBrain` without `onnxruntime_qnn` installed, or
  `GpuLocalBrain` without the `llama-server` OpenCL build / GGUF weights,
  raises immediately from `TwoBrainRouter.__init__` — the exact same failure
  mode the `pc` tier's own real brain already has, not a new one. There is no
  silent fallback to a stub here; if you set the flags, you need the real
  stack.

Cleanup: `TwoBrainRouter.close()` closes `fast_brain` and `escalation_brain`,
whichever are real (including `GpuLocalBrain`'s `llama-server` child process)
— `api.py`'s `serve()` calls it in `finally`, and anything constructing a
router directly should too.

---

## Real cloud, real GPU, and image routing (from `main`)

Independent of this branch's confidence-routing work, `main` made two more
seams real and closed a fourth:

- **`CirrascaleDeepBrain`** — a real Cloud AI 100 endpoint (Cirrascale AI
  Suite), behind `TWO_BRAIN_CLOUD_BRAIN=1`. Replaces `CloudDeepBrain`'s
  datasheet-guess stub with real measured latency and real published
  per-model pricing. Building it raises immediately if
  `INFERENCE_CLOUD_ENDPOINT`/`INFERENCE_CLOUD_API_KEY` aren't set — no
  silent fallback, same posture as every other real brain here.
- **`GpuLocalBrain`** — a real Adreno GPU brain (`llama-server`, OpenCL),
  behind `TWO_BRAIN_GPU_BRAIN=1`. One class serves both plain LLM and VLM
  weights; `GpuLocalBrain.for_vision()` loads a multimodal projector.
  Does not self-rate (Shape A) — a real option for later, not a backend
  limitation.
- **Image routing** — `route(query, context, image)`. An image never leaves
  the device: the cloud tier has no VLM at all, so an image-bearing query
  that needs to escalate is first *described* on-device
  (`GpuLocalBrain.describe_image`), and only that description — masked,
  same as any other text — is eligible to cross the boundary. Requires a
  vision-capable `fast_brain` (`can_see`); `route()` raises if an image is
  supplied to one that isn't.
- **A real, still-open privacy gap, deliberately pinned in the suite:**
  `privacy/patterns.py` only covers regex-shaped identifiers — person names
  are not masked. `tests/test_privacy.py::test_known_gap_person_names_are_not_masked`
  asserts the current broken behavior on purpose, so this stays visible
  rather than hidden behind `test_escalated_pii_never_reaches_cloud_unmasked`
  (which only ever checks an email). If you fix name masking, that test
  should start failing — invert it, don't delete it.

None of this needed a change to the confidence-routing work above, and
vice versa: `_build_deep_brain`/`_build_fast_brain` compose with
`_build_escalation_brain` exactly because each only reasons about its own
piece (which brain answers this tier) — see "Where code goes" in
`../CLAUDE.md` for why that seam discipline is the point.

---

## Per-tier signal

| Tier | Fast brain | Shape | "Not confident" goes to | Transport | Difficulty signal |
|---|---|---|---|---|---|
| AI PC (`pc_3b`) | `NpuFastBrain` — Phi-3.5-mini-instruct on Hexagon NPU | **C** | Cloud, and only for the named gap (no escalation brain on this tier) | in-process `ctypes`/Genie | self-report **+ named gap** |
| AI PC (`pc_3b`) | `GpuLocalBrain` — GGUF on the Adreno GPU, wins if both are set | **C** | as above | `llama-server` on loopback | self-report **+ named gap** |
| Mobile (`mobile_1b`) | `PhoneFastBrain` — Llama-3.2-3B on a Galaxy S25 | **B** | **The AI PC's brain** (if `TWO_BRAIN_GPU_BRAIN`/`NPU_BRAIN=1`), else cloud | HTTP to loopback (`adb reverse`) | model's own self-report |
| Cloud | `CloudDeepBrain` (stub) / `CirrascaleDeepBrain` (real) | — | — | — | n/a — escalation target |

### Confidence is weak on this tier; the gap field is not

`NpuFastBrain` self-rates as of the Shape B change. As predicted, that took a
prompt change plus a re-profile and **no** router change — `route()`,
`policy.py`, and `signals/confidence.py` were untouched; only `brains.py` and
`data/profile_workload/pc_3b.json` moved. `LocalFastBrain` and `CloudDeepBrain`
still use Shape A, so `signals/difficulty.py` remains the signal for the default
stdlib-only path.

Two honest caveats, both measured rather than assumed (receipts:
`data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md`, Attempt 5):

- **It is a prompted self-report, not a logprob.** Genie still exposes no token
  probabilities through the C API this brain uses. The number is the model's own
  claim about itself.
- **Its discrimination is weak.** Across 8 real queries the parsed values were
  0.85–1.00 — a merge-sort derivation self-rated the same 0.95 as "What is the
  capital of France?". Inverted, that is difficulty 0.00–0.15, all far below the
  0.55 threshold, so in practice this tier's escalations are driven almost
  entirely by the latency budget pre-check rather than by the confidence. The
  signal reliably separates "produced a number" from "didn't"; it does not yet
  separate easy from hard. This is the calibration risk `WALKTHROUGH.md`
  next-step #4(b) flagged, now confirmed on real hardware instead of predicted.

The threshold was deliberately **not** retuned to compensate — moving *the
heuristic's* threshold to fit seven samples of an unrelated signal would have
hidden the finding rather than fixed it. What changed instead, later, is that
confidence-based routing (Shape B/C) got its **own** threshold,
`RoutePolicy.confidence_escalate_threshold` (0.10, i.e. escalate at confidence
≤ 0.90, stay local only at ≥ 0.95) — see "Two thresholds, not one" below. That
is not the retune this paragraph declined to do: it is a second number for a
second signal, not a new value for the same one.

**Shape C is what actually addressed it** — not by improving the number, but by
asking for something else alongside it. Across ten real structured calls
(receipts: `data/npu_model/phi-3.5-mini-instruct/_real_structured_inference_log.md`)
the confidence stayed in its usual uninformative 0.95–1.00 band while the
`unknown` field cleanly separated the queries with a real missing piece from the
ones without:

| Query | confidence | `unknown` | Routed |
|---|---|---|---|
| "What time zone is Tokyo in?" | 1.00 | *(empty)* | local |
| "capital of France, and its population on 3 March 2019?" | 0.95 | "The specific population on 3 March 2019" | **hybrid** |
| "Pride and Prejudice, and the exact ISBN of the 1813 first edition" | 0.95 | "The exact ISBN of the 1813 first edition…" | **hybrid** |
| the PII demo query | 1.00 | *(empty)* | local |

Format compliance was **10/10** — a well-formed single-line JSON object every
time, nothing falling through to `parse_structured`'s lower rungs. That was the
main risk going in, since Genie's C API exposes no grammar hook the way
`llama-server`'s `response_format` does. It did not materialise on this
artifact; the degradation ladder exists in case it does on another.

Three real defects were found and fixed in the process, all prompt-shaped
rather than code-shaped — `unknown` is a place a small model can put the wrong
thing in three distinct ways, and each needed its own line in
`STRUCTURED_SUFFIX`:

1. **A caveat, not a gap.** The model used `unknown` for a disclaimer about an
   answer it had already given ("This response assumes the user's
   authority…"), which split a fully-answered PII query and sent masked PII to
   the cloud for nothing.
2. **An invented follow-up, not a gap.** For "explain about stable diffusion"
   the model answered fully, then put *"How does stable diffusion specifically
   apply to environmental science or economic models?"* in `unknown` — neither
   field was in the query. It widened the question rather than reporting a
   hole in its own answer, triggering a real ~15.6s Cirrascale call for a
   question nobody asked.
3. **A guess dressed as a complete answer — this one was self-inflicted.** The
   first fix for #2 ("only a literal part of the question") over-corrected: on
   the very next real run, the population half of the France/population query
   — a genuine gap this feature was originally validated against — came back
   as `{"solution": "Paris, France's population on 3 March 2019", "confidence":
   100, "unknown": ""}`. The model echoed the question phrase back as the
   "answer" instead of naming the gap, at claimed full confidence. Worse than
   #2: a silently wrong "complete" answer instead of an honest split.

Fixed by checking both directions together rather than one at a time: the ban
on inventing gaps (#2), plus an explicit instruction that not-knowing a fact
must be *named* in `unknown`, never guessed or echoed (#3). Validated on real
hardware across four categories — the original disclaimer/scope-creep/PII
cases (still fixed), a genuinely unknowable control fact (still named as a
gap, so the fix isn't just suppressing everything), and the France/population
query, which now answers `"Paris, 2,148,000"` with no gap. That number matches
the real INSEE 2019 estimate (2,148,271) closely enough to be genuine recall
rather than the #3 bug recurring — a correct avoided cloud call, not a
regression, but it means that query stopped being a reliable *test probe* even
though nothing about the behaviour is wrong (the test suite moved to the ISBN
query instead, which has no real answer to recall). Full account, including
the reverted attempt, in
`data/npu_model/phi-3.5-mini-instruct/_real_structured_inference_log.md`,
Run 4.

Whether the model's line between "I know this" and "I should say I don't" is
reliable in general is still **unmeasured** — one accurate recall is
reassuring, not a calibration. Check both directions again before the next
prompt edit, not just the one that motivates it.

Getting the prompt to behave was itself measured, not guessed. Left alone the
model emits the answer, the `CONFIDENCE:` line, and then paragraphs of
unrequested rationale until the token cap — tripling latency and leaving
truncated prose in the answer. `NpuFastBrain._STOP_SEQUENCES` therefore includes
`"\n\n"`, which is safe only because `_render_prompt`'s system message forbids
blank lines inside the reply; a blank line can then only follow the confidence
number. Asking the model to lead with the confidence was faster still but
sometimes returned a number and **no answer at all**, so it was rejected.

---

## Running it

```powershell
# AI PC tier, real NPU brain, Shape C (needs .venv-npu + the Genie artifact)
$env:TWO_BRAIN_NPU_BRAIN=1
$env:PYTHONPATH="src"
.venv-npu\Scripts\python.exe -m two_brain_router --tier pc

# ...the same thing behind the chat UI. Two terminals:
#   1) the router API   -- first request pays a ~12s Genie cold load
$env:TWO_BRAIN_NPU_BRAIN=1; $env:PYTHONPATH="src"
.venv-npu\Scripts\python.exe -m two_brain_router.api --tier pc
#   2) the static UI, from ui/
..\..\..\.venv\Scripts\python.exe -m http.server 5500   # then open http://127.0.0.1:5500

# Same tier, back on Shape B's bare CONFIDENCE: format, for A/B:
$env:TWO_BRAIN_STRUCTURED=0

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
| `TWO_BRAIN_NPU_BRAIN` | unset | `1` enables the real NPU brain (pc tier, or mobile's escalation target) |
| `TWO_BRAIN_GPU_BRAIN` | unset | `1` enables the real GPU brain (pc tier, or mobile's escalation target) — wins over NPU if both are set |
| `TWO_BRAIN_PHONE_BRAIN` | unset | `1` enables the real phone brain (mobile tier) |
| `TWO_BRAIN_PHONE_URL` | `http://127.0.0.1:8000` | where L is served |
| `TWO_BRAIN_PHONE_MODEL` | `llama-3.2-3b-instruct` | model id sent in the request |
| `TWO_BRAIN_PHONE_ALLOW_REMOTE` | unset | `1` permits a non-loopback L (see below) |
| `TWO_BRAIN_CLOUD_BRAIN` | unset | `1` enables the real Cirrascale cloud brain (needs `INFERENCE_CLOUD_ENDPOINT`/`INFERENCE_CLOUD_API_KEY`) |
| `TWO_BRAIN_TRACE` | off as a library, **on** for `api.py`; `--trace` for the CLI | Print every call's input and output, and what was masked at each crossing. See below. |
| `TWO_BRAIN_TRACE_REDACT` | unset | `1` shows placeholders instead of real values in the `[ON-DEVICE]` sections too — for demoing this in front of an audience |
| `TWO_BRAIN_TRACE_MAX` | `1200` | Per-field character budget before a value is elided |
| `TWO_BRAIN_STRUCTURED` | `1` (on) | `0` puts the AI-PC brains back on Shape B's bare `CONFIDENCE:` format. Only affects brains that are already enabled, so the stdlib-only default path is untouched either way. The isolation tool when this tier misbehaves: it separates "the model is bad at JSON" from "the model is bad at this question" without giving up the real hardware. |

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

**Changed again, for Shape C:** `BrainResponse` gained `unknown` and
`model_masked_output`; `Brain` gained `reports_gaps`; `RouteDecision` gained
`"hybrid"` plus `local_answer`/`cloud_answer`/`gap`; `RoutePolicy` gained
`needs_gap_fill()`, `local_partial_budget_ms` and `send_partial_to_cloud`.
`signals/structured.py` is new, and `routing/brains.py` grew a structured mode
on both AI-PC brains. `routing/policy.py` is **no longer untouched** — the two
fields and one pure predicate above are the first additions to it since the
original build. It is still pure (no I/O, no brain calls), which was the actual
constraint; "never edit this file" was never the rule.

**Changed a third time, and this one is an inversion rather than an addition:**
masking moved from the front door to the boundary. `Brain` gained
`trusted_with_raw_pii`; `route()` no longer masks before deciding;
`RouteDecision` gained `pii_entities_detected` alongside `pii_entities_masked`;
`router.py` gained `_Request`/`_LocalView` and `mask_for_boundary`, the single
method every crossing goes through. The README's pinned transcript moved with
it — its first note is now "detected 3 PII entities in the query" instead of
"masked 3 PII entities before any routing decision".

**Not changed, deliberately:** `PhoneFastBrain` and the whole mobile tier
(still Shape B, still masked — it is off-device); `signals/confidence.py`;
image routing.

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
