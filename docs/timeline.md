# Timeline — how this got here, and what's next

Chronological. For the *prioritised backlog* see
[`WALKTHROUGH.md`'s Next steps](WALKTHROUGH.md#next-steps); for *what runs on
which device right now* see
[`local-inference-status.md`](local-inference-status.md). This file exists to
answer "what happened, in what order, and what is immediately next" — it does
not restate either of those.

---

## 2026-08-04 — the architecture, all brains mocked

`af45f1b` → `95a3010`. Router, privacy guard, difficulty scoring, escalation
policy — all real, working Python. Both brains labeled stubs, because
`convert_model` is broken on two independent paths (gaps 3/3b) and no artifact
existed. `src/` restructured into explicit swap-in seams, so a real runtime
could be dropped into `routing/brains.py` without touching the router.

## 2026-08-06 — a real fast brain on the NPU

`ca97bdd`, merged to `main` as PR #1. `NpuFastBrain` runs Qualcomm's pre-built
Phi-3.5-mini-instruct Genie/QNN artifact in-process via `ctypes`, confirmed
executing on the Hexagon NPU. **Bypassed gap 3/3b rather than waiting on it**,
by downloading a pre-compiled artifact instead of converting one. ~10.6 tok/s.
Env-gated on `TWO_BRAIN_NPU_BRAIN=1`; the base package stays stdlib-only.

## 2026-08-06 — VLMs, and the search for a second device

Research first concluded a VLM on the NPU was not possible: no downloadable
QAIRT artifact for any VLM, and llama.cpp's Hexagon backend carried a
documented silent-CPU-fallback bug. The Adreno GPU sidestepped both.

- `0f980cc` — **Qwen3-VL-4B verified on the Adreno GPU**: 37/37 layers, KV
  cache, *and* vision encoder on `GPUOpenCL`, `graph splits = 1`. The 8B was
  ruled out — its vision tower is head_dim 72 and OpenCL's flash-attention
  kernels cover only 64/128, forcing a 4.58 GB single-tensor allocation
  against a 2048 MB device limit.
- `e8b40f7` — **GenieX `hybrid` (GGUF on Hexagon) does not work here.** The NPU
  really initialises and really takes 1950 MiB of repacked weights, then dies
  in the FastRPC transport. Notably *not* the silent-fallback bug — it fails
  loudly. Same run showed CPU and GPU are 2.4–3x faster than the NPU brain,
  which remains inconclusive because **power was never measured**.
- `0adeb1d` — **GenieX cannot drive a VLM in 0.3.18**; its own mtmd reports
  zero media markers in a prompt demonstrably containing two.
- `40c95ba` — status page collecting all of the above.

## 2026-08-06 — a real fast brain on the GPU

`b659df1`. `GpuLocalBrain` fills the same seam as `NpuFastBrain`, gated on
`TWO_BRAIN_GPU_BRAIN=1`. One class covers LLM and VLM weights, since the
difference is a projector argument, not a backend. Verified `using device
GPUOpenCL` / `offloaded 33/33 layers to GPU`.

Building it surfaced a defect worth remembering: with server logs discarded, a
silent CPU fallback would have been undetectable, because `llama-server` hides
device lines below `-v`. Hence `verify_gpu_placement()` and a test that asserts
placement from a captured log rather than inferring it from throughput.

**Current state: the GPU is the only device that runs both LLMs and VLMs, and
the router can use it for text today.**

---

# Next step — image input

Everything above routes *text*. The VLM is loaded, verified, and vision-capable,
but `GpuLocalBrain.answer()` accepts text only. Closing that gap is the next
piece of work.

To be precise about what is missing: these models are **image-text-to-text**, so
text-only *output* is inherent and `BrainResponse` never needs to change. It is
the **input** path that is unbuilt.

## This is a privacy decision before it is an engineering task

The router's entire purpose is the guarantee in
[`../CLAUDE.md`](../CLAUDE.md)'s Invariants. Images do not currently participate
in it:

- `PIIGuard.mask` masks **text**. It produces a `MaskResult` with a vault of
  text entities.
- `assert_masked_token_invariant` inspects **text**.
- Therefore an image containing a face, a document, a screen showing an email
  address, or EXIF GPS **would cross to `CloudDeepBrain` completely untouched,
  while every existing invariant check still passed.**

That is invariant #3 ("only masked text crosses the boundary") violated in
substance while green in test. It is the same shape of defect `CLAUDE.md`
already warns about for `phone_brain`, whose self-reported confidence sends the
raw, unmasked prompt off-device.

**So the first task is not plumbing an `image` parameter. It is deciding what
"mask first" means for an image.** Wiring the parameter first would silently
answer that question with "nothing".

## The decision, framed

Three defensible options, in increasing order of effort:

1. **Images never escalate.** A query carrying an image is answered locally or
   refused. Preserves the guarantee absolutely, costs the cloud tier's
   capability on exactly the queries most likely to need it.
2. **Escalate a redacted derivative, never the image.** The local VLM describes
   the image, that *text* goes through `PIIGuard` like any other text, and only
   the masked description escalates. Fits the existing invariant with no new
   masking machinery — the vault and the invariant check work unchanged. Costs
   fidelity, and the description itself may leak what a mask would have caught.
3. **Mask the image itself** — face/text detection and redaction before it
   crosses. Highest fidelity, and by far the most work: it needs a vision
   pipeline the project does not have, plus a way to extend
   `assert_masked_token_invariant` to pixels, which is a genuinely hard
   assertion to write.

Option 2 looks strongest for this codebase, precisely because it reuses the
existing text guarantee rather than inventing a parallel one for pixels — but
that is a recommendation, not a decision, and it should be made explicitly.

## Then, and only then, the engineering

Sequenced so nothing pre-empts the decision above:

1. **Extend the `Brain` protocol** with optional image input —
   `answer(masked_query, context="", images=None)`. This *will* touch
   `router.py`, which `CLAUDE.md` normally flags as a seam drawn wrong. That
   flag is correct and should be respected: adding a modality is a genuine
   protocol change, not a backend swap, and it deserves to be visible in the
   diff rather than hidden inside a brain.
2. **Carry images through `route()`** in whatever shape the decision above
   implies — for option 2, they never reach `_escalate` at all.
3. **Extend the invariant test.** `test_escalated_pii_never_reaches_cloud_unmasked`
   is the regression test for the whole guarantee; it needs an image-carrying
   sibling. Per `CLAUDE.md`, if the existing test has to be *edited* to pass,
   the guarantee changed and that is a review-worthy decision.
4. **Implement in `GpuLocalBrain`.** The smallest part: pass `--mmproj` (already
   supported), post `image_url` content parts to `llama-server`'s
   OpenAI-shaped endpoint. Use Qwen3-VL-4B — the 8B needs
   `--no-mmproj-offload` and is slower and no more accurate.
5. **Set expectations on accuracy.** Both models get colour and shape right and
   **misplace position** at the default 256 image tokens. Anything positional
   needs `--image-min-tokens 1024`, which roughly quadruples prompt tokens and
   halves decode. Fine for description, classification, OCR-ish and Q&A; not
   for grounding. See
   [`local-inference-status.md`](local-inference-status.md).

## Not blocking this

Perf-per-watt (NPU vs GPU vs CPU) is the other open question. It is
**independent** — it decides which gate should win for the *text* path, and
does not gate image work, which is GPU-only regardless since no VLM runs on
the NPU at all.
