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
3. **Cost is half-real.** Token counts are real, from `usage`. The *rate* is
   still the datasheet-derived `token_cost_usd_per_1k: 1.8` in
   `cloud_large.json`, and it is almost certainly wrong — it yields **$1.03 for
   a single query**, which no hosted 8B costs. Cirrascale publishes no
   per-token price through this API. Treat `cost_usd` as unusable until a real
   rate is obtained.
4. **`Llama-3.1-8B` is the only model reachable on this key.** `Llama-3.1-70B`
   and `meta-llama/Llama-3.1-70B-Instruct` both return
   `{"message":"Too many requests. Invalid model/rate limits not configured for
   this model.","status":"error"}` — which reads as an **account provisioning
   limit, not a platform limit**. Worth asking Cirrascale to enable a larger
   model; `TWO_BRAIN_CLOUD_MODEL` overrides the default with no code change.

## Architectural finding: the tiers are closer than the design assumes

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
