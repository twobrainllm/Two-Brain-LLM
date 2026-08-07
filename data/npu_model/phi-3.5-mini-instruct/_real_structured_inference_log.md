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

## What is still not verified

- **The cloud leg.** No `INFERENCE_CLOUD_ENDPOINT`/`INFERENCE_CLOUD_API_KEY` on
  this machine, so every split above filled its gap from `CloudDeepBrain`'s
  labeled stub. The gap text that *would* cross was checked (masked, invariant
  asserted); the round trip itself was not.
- **`GpuLocalBrain`'s structured path.** The `llama-server` OpenCL build and
  GGUF weights are not present here, so its `response_format: {"type":
  "json_object"}` path has been exercised only against the parser, never
  against a real Adreno run.
- **Image input.** Shape C is text-only by construction right now — an
  image-bearing query is routed down the heuristic path instead (see
  `router.py::route`). Deliberate, and the next step.
- **Whether the caveat ban suppresses real gaps** — see Run 2's caveat above.
