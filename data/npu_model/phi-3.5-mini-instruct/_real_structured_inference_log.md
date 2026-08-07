# Real structured-output runs — Phi-3.5-mini-instruct on the Hexagon NPU

Receipts for the Shape C ("solve what you can, name what you can't") reply
format — `signals/structured.py` + `NpuFastBrain(structured=True)`. Companion
to `_real_inference_smoke_log.md`, which covers the earlier bare
`CONFIDENCE:` format (Shape B) and the three defects fixed getting there.

Everything below is **real output from the real artifact on this machine's
NPU**, not a mock. Nothing here is `_mock: true`.

- Artifact: `raw/phi_3_5_mini_instruct-genie-w4a16-qualcomm_snapdragon_x_elite`
- Runtime: `Genie.dll` via `ctypes`, from `onnxruntime_qnn` 2.4.0 in `.venv-npu`
- Date: 2026-08-06
- Deep brain: `CloudDeepBrain` (stub). `INFERENCE_CLOUD_ENDPOINT` /
  `INFERENCE_CLOUD_API_KEY` are **not set on this machine**, so the *cloud leg*
  of every split below is a labeled stub. The local leg, the gap detection and
  the routing decision are all real; the cloud round trip is not. Re-run with
  `TWO_BRAIN_CLOUD_BRAIN=1` once credentials exist.

---

## Headline: the gap field discriminates where the confidence number does not

`_real_inference_smoke_log.md` (Attempt 5) recorded that this artifact's
self-reported confidence "separates 'produced a number' from 'didn't'; it does
not yet separate easy from hard" — 0.85–1.00 across eight queries, a merge-sort
derivation rating itself the same as "What is the capital of France?".

That finding still holds. **The `unknown` field does not share it.** Across the
runs below the model reliably names the specific sub-question it cannot do,
while rating itself 0.95–1.00 overall:

| Query | confidence | `unknown` | Routed |
|---|---|---|---|
| "What time zone is Tokyo in?" | 1.00 | *(empty)* | local |
| "capital of France, and its population on 3 March 2019?" | 0.95 | "The specific population on 3 March 2019" | **hybrid** |
| "Summarise Pride and Prejudice, and give the exact ISBN of the 1813 first edition" | 0.95 | "The exact ISBN of the 1813 first edition of Pride and Prejudice." | **hybrid** |

The middle row is the one worth staring at. Confidence 0.95 inverts to
difficulty 0.05 — *far* below the 0.55 threshold — so under Shape B this query
stays local and answers "Paris", silently dropping half of what was asked. The
model knew the other half was missing the whole time; there was simply nowhere
to say so. That is the capability this format adds, and it is why
`RoutePolicy.needs_gap_fill` treats a named gap as sufficient on its own,
regardless of the confidence number.

## Format compliance: 10/10

Two runs of five queries each, via `NpuFastBrain.answer()`:

- **10/10** parsed as `source="json"` — a well-formed single-line JSON object
  every time, no code fences, no prose around it.
- **0** fell through to the `CONFIDENCE:` rung, **0** to the raw rung.
- **0** `BrainResponse.error` set.

This was the main risk going in — a 3.8B w4a16 model asked for machine-readable
output, with no grammar constraint available (Genie's C API exposes no GBNF
hook, unlike `llama-server`'s `response_format`). It did not materialise on this
artifact. It may still on another; `parse_structured`'s degradation ladder
exists for that and is unit-tested even though nothing exercised it here.

Latency, per call: **2213–11011 ms** (10 calls). Higher than Shape B's ~1.8 s
mean, as expected — JSON scaffolding plus a prose `unknown` field is simply more
tokens. `_STRUCTURED_MAX_NEW_TOKENS = 320` (up from 96) and the `"\n\n"` stop
sequence still ends generation first in every observed call.

---

## Run 1 — original prompt wording

`STRUCTURED_SUFFIX` as first written: *"the specific part you cannot answer, or
an empty string if you answered fully"*. Cold load 12675 ms.

| Q | ms | confidence | `unknown` |
|---|---|---|---|
| Tokyo time zone | 2558 | 1.00 | *(empty)* |
| merge sort vs quicksort | 6501 | 0.95 | "Explanation of the trade-offs …, including stability, space requirements, and practical performance" |
| capital of France + population on a date | 2562 | 1.00 | "The population of Paris on 3 March 2019" |
| Pride and Prejudice + 1813 ISBN | 5169 | 0.95 | "The exact ISBN of the 1813 first edition is not provided as it requires specific historical publication data." |
| PII draft (pre-masked input) | 4702 | 1.00 | *(empty)* |

### The defect this run exposed

Routed end-to-end (`TwoBrainRouter(tier="pc")`), the PII demo query came back
`tier=hybrid` with this in `unknown`:

> "This response assumes the user's authority to address the security matter and
> does not involve retrieving or handling personal identification numbers
> directly."

That is a **disclaimer about an answer the model did give**, not a part it
failed to give. `needs_gap_fill` cannot tell the difference — a non-empty string
is a gap — so a query the local model had answered *completely* was split, and
its masked PII was sent to the cloud for no reason. On this project's flagship
privacy query, that is the exact round trip the whole sample exists to avoid.

Not a parser bug and not a router bug: the model answered a question the prompt
had accidentally asked. Fixed at the prompt, in `STRUCTURED_SUFFIX` and
`STRUCTURED_SYSTEM_PROMPT`, by naming the misuse explicitly — *"Caveats,
assumptions, disclaimers and notes about the answer you did give do not belong
there."*

## Run 2 — after the caveat ban

Same five queries, same session shape. Cold load 11314 ms.

| Q | ms | confidence | `unknown` |
|---|---|---|---|
| Tokyo time zone | 2213 | 1.00 | *(empty)* |
| merge sort vs quicksort | 11011 | 0.95 | *(empty)* — answered the trade-offs inline instead |
| capital of France + population on a date | 3075 | 0.95 | "The specific population on 3 March 2019" |
| Pride and Prejudice + 1813 ISBN | 5572 | 0.95 | "The exact ISBN of the 1813 first edition of Pride and Prejudice." |
| PII draft (pre-masked input) | 4026 | 1.00 | *(empty)* |

The PII query now stays **fully local** — `tier_answered="local"`, nothing
crosses the boundary. The two genuine knowledge gaps (a population on a
specific date, a 1813 ISBN) are still named. Both are the desired direction.

**Honest caveat on this fix.** The merge-sort query stopped naming a gap
between the two runs and answered the trade-offs itself, taking 11011 ms to do
it. One sample either side of a prompt change is not evidence that the ban made
the model *better* at deciding what a gap is — only that it stopped putting
disclaimers there. Whether the caveat ban also suppresses real gaps is
**unmeasured**, and it is the first thing to check if splits start looking too
rare.

---

## The latency-budget interaction, and why `local_partial_budget_ms` exists

In Run 1, routed end-to-end, **two of the four demo queries never reached the
fast brain at all**: the profiled estimates (3222 ms and 4855 ms, from
`data/profile_workload/pc_3b.json`) exceeded `local_latency_budget_ms = 3000`,
so the Shape B pre-check skipped the local model and escalated.

That pre-check is correct in Shape B, where its stated premise holds — "a local
answer would have been discarded anyway". In Shape C the premise is false: a
usable local answer is always kept. Applying it anyway meant the PII query was
sent to the cloud without the local model being consulted, and the local model
then turned out (Run 2) to answer it fully on-device.

Hence `RoutePolicy.local_partial_budget_ms = 15000`, used only on the Shape C
path. The value is derived from the measurements above — ~2.3x the slowest
observed call (11011 ms) — and is a runaway guard, not a target. Queries between
3000 ms and it are answered locally anyway, with a note in the audit trail
saying so.

`local_latency_budget_ms` is unchanged at 3000 ms and Shape A/B still use it.

---

---

## Run 3 — after masking moved to the boundary

The local model is now handed the query **unmasked** (it runs on this machine;
`Brain.trusted_with_raw_pii`). Masking happens only where text crosses.

Two things this changed, both observed on the PII demo query:

**1. The answer got better, which was the point.** Given placeholders, the model
had written:

> "Your SSN [PII_SSN_1] was found in an old backup. Please rotate it
> immediately to maintain security."

Given the real text:

> "We have identified your Social Security Number (SSN) 123-45-6789 within an
> old backup. It is highly recommended to rotate this sensitive information
> immediately to maintain your security."

**2. The local model's own output now contains raw PII, and that output crosses
the boundary on a split.** This is the hazard the change introduces, so it was
checked against the actual bytes rather than reasoned about. Recording the raw
`(query, context)` pair handed to the deep brain on that same query:

```
tier: hybrid | detected: 3 | masked: 3
Local answer (stays on device): We have identified your Social Security Number (SSN) 123-45-6789 within an old backup...

--- cloud call 0: 555 chars crossed
    LEAKED: none
    placeholders present: ['[PII_EMAIL_1]', '[PII_PHONE_1]', '[PII_SSN_1]']
    context: A smaller on-device model has already answered part of this question:
             We have identified your Social Security Number (SSN) [PII_SSN_1] within
             an old backup. It is highly recommended to rotate this sensitive
             information immediately to maintain your security.
```

The same sentence, holding `123-45-6789` on-device and `[PII_SSN_1]` on the
wire. Query-level masking could not have produced that: this sentence did not
exist when the query was read. `_answer_hybrid`'s `mask_for_boundary` on the
partial answer is the only thing standing between the user and a leak here, and
`tests/test_structured_routing.py::test_the_raw_partial_answer_is_masked_before_it_crosses`
is its regression test.

**A note on the gap quality in this run.** The gap named was *"Potential reasons
or implications of not rotating the SSN were not discussed."* — arguably a real
missing part, arguably a caveat wearing a gap's clothing. It is on the fuzzy
side of the Run 1 defect, and it caused a split where Run 2 stayed local.
Another data point for the still-open question of whether the caveat ban is
calibrated right; not enough on its own to change anything.

---

## Run 4 — `unknown` was widening scope to topics never asked, and the first fix over-corrected

Trigger: a real user query through the live UI/API, `TWO_BRAIN_CLOUD_BRAIN=1`
against the real Cirrascale endpoint (`CirrascaleDeepBrain`, not the stub — the
first fully real end-to-end split observed).

```
query: "explain about stable diffusion"
local : "Stable diffusion refers to a process where a state of equilibrium is
         maintained, often in a physical or chemical context..." (wrong
         definition, but a complete answer to what was asked)
gap   : "How does stable diffusion specifically apply to environmental
         science or economic models?"
```

Neither "environmental science" nor "economic models" appears anywhere in the
query. The model answered fully (if incorrectly) and then *invented a
follow-up question* to put in `unknown`, which triggered a real ~15.6s,
$0.000374 `CirrascaleDeepBrain` call for a question nobody asked.

This is a different species from Run 1's defect (a disclaimer about the given
answer) — here the model isn't hedging on its own answer, it's widening the
question to a new topic. Same failure shape (something-that-isn't-a-gap read
as one), different mechanism, so it needed its own prompt line rather than
folding into the existing caveat ban.

### Attempt A — "only a literal part of the question" (reverted)

`STRUCTURED_SUFFIX`/`STRUCTURED_SYSTEM_PROMPT` were tightened to: put
something in `unknown` *only* if the user's own question contains an
unanswered part; never a new topic, application, or follow-up.

Result on the trigger query: fixed, 3/3 clean reruns, no invented topic.

**But it broke a working case.** Re-running the query this whole feature was
originally validated against:

```
query: "What is the capital of France, and what was its population on 3 March 2019?"
RAW    : {"solution": "Paris, France's population on 3 March 2019",
          "confidence": 100, "unknown": ""}
```

3/3 identical (this artifact appears to decode near-deterministically given
fixed input — every repeated probe in this file has returned byte-identical
output). The model echoed the *phrase* "France's population on 3 March 2019"
back as if it were the number, at claimed full confidence, and named no gap.
That is worse than the pre-fix behaviour: a silently wrong "complete" local
answer instead of an honest split. Caught before shipping by testing the
*two* failure directions together (invented gap vs. erased gap) rather than
just the one that motivated the change — the general lesson recorded inline
in `structured.py` now, so the next prompt edit checks both.

### Attempt B — ship (current)

Kept the scope-creep ban, and added an explicit instruction for the failure
Attempt A introduced: if the model does not actually know a fact the user
asked for, it must name that in `unknown` — not guess, not echo the question
back. `STRUCTURED_SUFFIX`'s worked example now shows both directions in one
place (a real literal gap vs. an out-of-scope topic) rather than one.

Validated against four categories, real hardware, `n=2`–`3` each (all
byte-identical across repeats):

| Query | Result |
|---|---|
| "explain about stable diffusion" (the trigger) | No gap, answers fully, stays **local**. Fixed, 3/3. |
| PII draft (email/phone/SSN) | No gap (disclaimer-shaped or otherwise), stays **local**. 3/3, matches Run 1's fix. |
| Pride and Prejudice + 1813 ISBN (no real answer exists — ISBNs postdate 1970) | Names the gap every time: `"the exact ISBN of the 1813 first edition"`. 3/3. |
| "…exact serial number on my desk, and what is 12×12?" (control: one trivial, one truly unknowable) | Answers `144` inline, names `"the exact serial number printed on the laptop"` as the gap. 2/2. Proves the fix didn't just suppress *all* gaps. |
| Capital of France + population on a date | No longer names a gap — answers `"Paris, 2,148,000"`. |

That last row needed a second check before trusting it: is `2,148,000` a real
recalled fact or a second instance of the Attempt-A bug wearing a plausible
number? INSEE's published Paris population estimate for 2019 is **2,148,271**
— the model's figure is accurate to within a few hundred. Combined with the
control row (which shows the model *will* name a gap for something it truly
cannot know, in the same call shape), this reads as genuine recall, not
confabulation: the local model already knew this fact, so not escalating it is
the correct call and a real avoided cloud round-trip, not a regression.
`tests/test_npu_brain.py`'s structured-gap tests were moved off this query for
exactly this reason — it stopped being a reliable *probe* even though nothing
about it is wrong.

**Still an open question, not a new one:** the France/population case shows
this artifact's line between "I know this" and "I should say I don't" is
sensitive to exact wording in ways not fully mapped. The control row is
reassuring but is one example, not a calibration. Treat any future prompt
change the same way this one eventually was — checked against a
known-gap query and a known-complete query together, not just the one that
motivated the edit.

---

## Run 5 — the deep brain refused a query it should never have been asked

Real end-to-end through the UI: real NPU local brain, real `CirrascaleDeepBrain`,
two turns, PII in both. Masking was flawless -- 4 entities detected, 4 masked, 4
rehydrated, nothing raw on the wire. The *routing* was not.

The local model split correctly, naming a clean gap:

```
unknown : The exact ISBN of the 1813 first edition of Pride and Prejudice
```

But what the cloud was actually **asked** was the whole original query:

```
query   : My email is [PII_EMAIL_1], my phone is [PII_PHONE_1], my card is
          [PII_CREDIT_CARD_1] and my SSN is [PII_SSN_1]. Draft a short reply
          about the backup breach, and give me the exact ISBN ...
context : ... Answer only the remaining part it could not: The exact ISBN ...
```

and it declined:

> "I cannot provide you with a reply that includes your personal information."

**Root cause: the gap was in the wrong argument.** `_answer_hybrid` passed
`masked_query.masked_text` as the deep brain's `query` and buried the gap in
`context`. A chat model follows the user message, so it attempted the entire
placeholder-laden drafting request and refused on safety grounds -- when the
only thing needed from it was an ISBN, which is innocuous.

Not a masking failure and not a model failure. The masking did its job; the
model behaved reasonably given what it was handed. The bug was that it was
handed the wrong question, and it contradicted the spec this feature was built
to: *"what needs to be addressed goes into a second call."*

**Fix:** the gap becomes the `query`; the original query, the conversation
history and the partial answer become `context` background. Re-run of the exact
same two turns:

```
gap   : the exact ISBN of the 1813 first edition
cloud : "The 1813 first edition of Pride and Prejudice does not have an ISBN, as
         the International Standard Book Number (ISBN) system was not introduced
         until the 20th century. ISBNs were first used in 1966."
```

Correct, and factually right. Still `4 detected / 4 masked`.

**A second property came free.** `compress_context` only ever trims `context`,
so with the gap living there a long partial answer plus a small budget could
truncate away the very instruction saying what the deep brain was for. As the
`query` it cannot be trimmed at all -- the guarantee is now structural rather
than a matter of keeping it last in a list.
`tests/test_structured_routing.py::test_the_gap_is_the_question_the_deep_brain_is_asked`
and `::test_the_gap_cannot_be_lost_to_context_compression` pin both.

### The follow-up defect that fix exposed

A second real run of the same scenario -- this time with no conversation
context -- had the NPU decline outright:

```
text       : I am unable to provide the exact ISBN for the 1813 first edition...
confidence : 0.00
unknown    : (absent)
```

Confidence 0.00 **and no gap named**. `needs_gap_fill` fires on the confidence
alone, so this still reached `_answer_hybrid` -- which, now that the gap is the
deep brain's `query`, asked the cloud **the empty string**. Reproduced
directly: `cloud was asked: ''`.

Invisible before the gap-as-query change, because the old code sent the whole
query as the ask regardless of whether a gap existed. A fix in one place turned
a latent nonsense case into a live one.

**Fix:** no gap named means there is nothing gap-shaped to ask about, so it is
an ordinary escalation -- whole query, local answer discarded -- rather than a
split on nothing. Verified on the same input: `tier=cloud`, cloud asked
`"My SSN is [PII_SSN_1]. Draft a reply and give me the exact ISBN..."`.

Note what *doesn't* change: a named gap still splits at **any** confidence,
including 0.00. "No gap named" is the trigger, not "low confidence" -- a model
that can say what it is missing is worth splitting on however unsure it is
overall. Both pinned:
`::test_an_empty_gap_never_becomes_an_empty_question_to_the_cloud` and
`::test_a_named_gap_still_splits_even_at_zero_confidence`.

**Worth watching:** the cloud still saw the drafting request as background and
appended a sentence about it ("Regarding the backup breach, I must inform you
that your SSN was found in an old backup"). Harmless and masked, but it shows
background is not inert -- a future tightening could withhold the original query
entirely and send only gap + partial.

---

## What is still not verified

- **`GpuLocalBrain`'s structured path.** The `llama-server` OpenCL build and
  GGUF weights are not present here, so its `response_format: {"type":
  "json_object"}` path has been exercised only against the parser, never
  against a real Adreno run.
- **Image input.** Shape C is text-only by construction right now — an
  image-bearing query is routed down the heuristic path instead (see
  `router.py::route`). Deliberate, and the next step.
- **How far the France/population finding generalises** — see Run 4's closing
  note. One accurate recall on one fact is not evidence the model reliably
  knows when it knows something.

## What is now verified that wasn't

- **The cloud leg is real.** Run 4's trigger query went through
  `CirrascaleDeepBrain` against the live Cirrascale endpoint, not the stub —
  real latency (~15.6s for a 70B-class completion), real cost ($0.000374),
  real response text. Superseded: the "cloud leg… was not [verified]" line
  from Runs 1–3.
