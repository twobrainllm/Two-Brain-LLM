# Cloud AI 100 deep brain via Cirrascale — real endpoint log

`_mock: false` for everything measured below. This is the first time the cloud
tier in this project has been anything other than a datasheet stub.

Credentials live in `secrets.txt`, which is **gitignored** and must stay so.
`CirrascaleDeepBrain` reads `INFERENCE_CLOUD_ENDPOINT` / `INFERENCE_CLOUD_API_KEY`
from the environment, never from a tracked file.

## What is real now

Endpoint: `https://aisuite.cirrascale.com/apis/v2`, OpenAI-shaped.

`GET /models` — real response:

```json
{"text_to_image":["stabilityai/sdxl-turbo"],
 "llm":["Llama-3.1-8B"],
 "embedding":["BAAI/bge-base-en-v1.5","BAAI/bge-large-en-v1.5"]}
```

`POST /chat/completions` returns a standard `chat.completion` with a real
`usage` block. A real answer to "What is gravity?" came back in 596 ms.

### Real measured latency (n=2 per point, very linear)

| max_tokens | completion_tokens | total wall time |
|---|---|---|
| 16 | 16 | 276.7 / 279.2 ms |
| 128 | 128 | 1380.1 / 1393.2 ms |

Derived: **~9.9 ms/token (~101 tok/s), ~119 ms fixed overhead** (network RTT +
prefill for a 44-token prompt). This should replace the hand-authored numbers
in `data/profile_workload/cloud_large.json`, which were datasheet guesses.

For contrast, on-device measured in the same session: **25.8 tok/s** sustained
on the Adreno GPU. The cloud tier is ~4x faster per token.

## What is still NOT real — stated plainly

1. **The silicon is unverified from here.** Cloud AI 100 attribution comes from
   Cirrascale's service documentation, not from anything observable through the
   API. `/models`, `/health` (200) and the completion response were all
   checked; none report device, backend, or accelerator. `/`, `/info` and
   `/hardware` are 404. **We cannot prove this ran on a Cloud AI 100** — only
   that it ran on Cirrascale's hosted inference service.
2. **Gap #4 is not closed.** `hardware_detect` still has no cloud platform
   value and `convert_model` no cloud `target_sdk`. This bypasses QUAD entirely,
   the same way the NPU brain bypassed `convert_model`.
3. ~~**Cost is half-real.**~~ **SUPERSEDED** — the account owner supplied the
   vendor catalogue shortly after this was written, so pricing is now real.
   See "Correction + real catalogue and pricing" below. The suspicion recorded
   here (that 1.8 USD/1k "is almost certainly wrong") was correct, and
   understated: it was off by ~16,000x.

## ~~Architectural finding: the tiers are closer than the design assumes~~ (SUPERSEDED)

**Superseded** by the 70B measurements at the end of this file. The finding
below was true while `Llama-3.1-8B` was the only reachable model; with
`Llama-3.3-70B` answering, escalation buys ~18x the fast brain's parameters
*and* ~2x its throughput. Kept for the record, not as current guidance.

The deep brain is **8B**; the local fast brain is Phi-3.5-mini at **3.8B**, and
the same Adreno GPU has already run an **8B** VLM locally. So escalation
currently buys roughly 2x parameters over the fast brain and **nothing** over
what the device could run itself.

What it does buy is **~4x throughput** (101 vs 25.8 tok/s). That is a real,
measured benefit — but it means the escalation policy is currently trading
privacy exposure for *speed*, not for *capability*. Any threshold tuning in
`routing/policy.py` should be done with that in mind, and it is a strong
argument for provisioning a 70B.

## 🔴 Real privacy failure found on the first real escalation

Escalating a PII-laden query through the **real** endpoint, with the outbound
payload intercepted, this is what actually left the device:

```
My name is Sarah Chen, email [PII_EMAIL_1], SSN [PII_SSN_1], phone [PII_PHONE_1].
Analyse in depth the legal and tax implications of ...
```

Email, SSN and phone masked correctly. **The person's name was sent in plain
text to a third party.**

Root cause: `privacy/patterns.py` covers only regex-shaped identifiers —
`EMAIL`, `PHONE`, `SSN`, `CREDIT_CARD`. Person names are an open class and are
not detected at all. This is not a bug in the new brain; it is a pre-existing
limitation of the mocked guard (gap G8 — `quad.privacy` is unavailable here)
that was **harmless while the cloud brain was a stub and is not harmless now**.

`tests/test_routing.py::test_escalated_pii_never_reaches_cloud_unmasked` passes
green throughout, because it only ever asserts on an email. The guarantee is
narrower than that test's name implies.

**Consequences, in order:**

- `TWO_BRAIN_CLOUD_BRAIN=1` must **not** be used with real user data until
  names are handled. It is off by default, and should stay off.
- `tests/test_privacy.py::test_known_gap_person_names_are_not_masked` now pins
  this deliberately, asserting the *current broken* behaviour so it is visible
  in the suite. When name masking works, that test starts failing — invert it,
  do not delete it.
- Regex cannot solve this properly; names need either `quad.privacy` (G8) or a
  local NER pass. That is the decision to make before this tier carries real
  data.

One fabricated name ("Sarah Chen") was sent to Cirrascale during this test. No
real personal data was involved.

---

## Correction + real catalogue and pricing (same day)

The vendor catalogue, supplied by the account owner, corrects two earlier
conclusions in this file.

| Size | Model | Context | $/1M input | $/1M output |
|---|---|---|---|---|
| 8B | `Llama-3.1-8B` | 8K | 0.02 | 0.22 |
| 32B | `Qwen-QwQ-32B` | 8K | 0.08 | 0.36 |
| 70B | `Llama-3.3-70B` | 8K | 0.19 | 0.69 |
| 70B | `DeepSeek-R1-Distill-Llama-70B` | 8K | 0.19 | 0.69 |

**Correction 1 — a 70B does exist; the earlier probe used a wrong name.**
This file previously read the 70B failure as an account provisioning limit.
That was wrong. `Llama-3.1-70B` (the name tried) simply does not exist. The
service's two error messages are diagnostic and distinguish the cases:

- `"Invalid model/rate limits not configured for this model"` → **name not in
  the catalogue** (returned for `Llama-3.1-70B`, and for `Llama3.1-8B` without
  the dash).
- `"Internal Server Error: Models Busy/Unavailable"` (HTTP 500) → **valid name,
  model not currently loaded** (returned for `Llama-3.3-70B`,
  `DeepSeek-R1-Distill-Llama-70B`, `Qwen-QwQ-32B`).

**Correction 2 — the cost figure was wrong by four orders of magnitude.**
Recomputed on the real usage block from the live 8B call (prompt 46,
completion 38):

| | Cost for that one call |
|---|---|
| Old mocked rate (1.8 USD/1k) | $0.15120000 |
| Real, `Llama-3.1-8B` | **$0.00000928** |
| Real, `Llama-3.3-70B` | $0.00003496 |

**Overstated by ~16,293x.** It did **not** affect routing — `policy.py` decides
on difficulty and `local_latency_budget_ms`, and never reads cost — so the
damage was confined to reporting. `CirrascaleDeepBrain.MODEL_PRICING` now bills
input and output separately at the published rates, and
`data/profile_workload/cloud_large.json` is `_mock: false` for latency and
pricing.

The practical consequence is that **cloud escalation is essentially free at
these volumes** (~$0.00003 for a 70B answer). Any future policy that weighs
cost against privacy exposure should start from that, not from the old figure.

## Service availability is real and intermittent

Mid-session the whole service degraded: `Llama-3.3-70B`, both other large
models, **and eventually `Llama-3.1-8B`** all returned HTTP 500
`Models Busy/Unavailable`, and `GET /models` — which had earlier returned the
full catalogue — began returning `{}` while still answering HTTP 200.

This is load state, not provisioning: the 8B had been answering normally
minutes earlier. It means **the deep brain must be assumed intermittently
unavailable**, which is now handled: `CirrascaleDeepBrain` tries
`Llama-3.3-70B` and falls back to `Llama-3.1-8B`, surfacing the service's own
message if both fail.

**Still unverified because of that outage:** a live end-to-end call against a
70B. The fallback chain was exercised and behaved correctly (tried both, raised
with the real service message), and the pricing maths was verified against real
captured token counts — but no 70B has yet answered on this account. Re-run
when the service recovers, and re-measure `cloud_large.json`'s latency against
it, since a 70B will be slower per token than the 8B those numbers came from.

---

## 🔴 There is no VLM on this service — the vision path cannot escalate

Checked against the account's own catalogue once the outage cleared. The API's
taxonomy is exactly three categories, and none is vision-understanding:

```json
{"llm":["DeepSeek-R1-Distill-Llama-70B","Llama-3.1-8B","Llama-3.3-70B","Qwen-QwQ-32B"],
 "text_to_image":["stabilityai/sdxl-turbo"],
 "embedding":["BAAI/bge-base-en-v1.5","BAAI/bge-large-en-v1.5"]}
```

The two Stable Diffusion models run the **opposite direction** — text in, image
out. A VLM is image in, text out. Nothing here reads an image.

Confirmed behaviourally, not just from the listing. Sending OpenAI multimodal
content-parts (`type: image_url`, base64 PNG) to `Llama-3.3-70B` **does not
error** — it returns HTTP 200 and answers:

> "I don't see an image. As a text-based AI model, I don't have the capability
> to visually access or analyze images."

**The image is silently dropped.** That is a worse failure mode than a rejection:
a caller who assumed vision support would get a confident, plausible, entirely
image-blind answer with no signal that anything was lost.

### Consequence for the architecture

**The local Qwen3-VL-4B on the Adreno GPU is the only vision capability in the
entire system.** There is no cloud deep brain for images, so an image-bearing
query has no escalation target.

This settles the open question in [`../../docs/timeline.md`](../../docs/timeline.md).
That file framed three options for image input and recommended option 2
(escalate a masked *textual* derivative rather than the image) on privacy
grounds. It is now also the **only technically available** option:

- Option 1 (images never escalate) is no longer a choice — it is the default
  state, forced by the catalogue.
- Option 3 (mask and send the image itself) has nowhere to send it.
- **Option 2 is the only path by which an image query can benefit from the
  cloud at all**: the local VLM describes the image, that description is masked
  by `PIIGuard` like any other text, and only masked text escalates to the 70B.

The privacy-preferred design and the only feasible design turn out to be the
same one. Worth noting the caveat that already applies: the description is
generated by the local VLM, so whatever it chooses to mention is what escapes —
if it describes a face or reads text off a document, that lands in the text
path and is only as safe as `PIIGuard` is, which currently does not mask names.

## Real 70B measurements (outage cleared)

`Llama-3.3-70B` answered. Superseding the 8B-derived figures:

| Model | 16 tok | 128 tok | Derived |
|---|---|---|---|
| **Llama-3.3-70B** | 748.5 / 385.7 ms | 2718.4 / 2674.3 ms | **~19.0 ms/token (~52.6 tok/s)**, ~263 ms overhead |
| Llama-3.1-8B | 276.7 / 279.2 ms | 1380.1 / 1393.2 ms | 9.9 ms/token (101 tok/s), ~119 ms overhead |

The 16-token 70B runs varied 2x (748.5 vs 385.7 ms), so the fixed-overhead
figure is approximate; the per-token slope is stable.

**This repairs the escalation premise.** With the 8B the deep brain was only ~2x
the local fast brain and this device already runs an 8B locally, so escalation
bought throughput and nothing else. At 70B it is ~18x the fast brain's
parameters **and** still ~2x faster than the local GPU's 25.8 tok/s. Escalation
now buys capability, which is what the architecture always assumed.

## Real rate limits (account dashboard)

| Model | req/min | req/hour | req/day |
|---|---|---|---|
| Llama-3.3-70B | 20 | 1000 | — |
| DeepSeek-R1-Distill-Llama-70B | 15 | 750 | 7500 |
| Qwen-QwQ-32B | 15 | 800 | 8000 |
| Llama-3.1-8B | 70 | 3500 | 35000 |

**20 requests/minute on the default deep brain** is a real operational ceiling —
one escalation every 3 seconds sustained. Not a problem for interactive use, but
it rules out batch evaluation against the 70B without throttling, and it is
another argument for the 8B fallback path already implemented.
