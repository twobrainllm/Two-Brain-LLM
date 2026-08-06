# VLM-on-NPU research (4-8B range) -- blocked on NPU, solved on GPU

> **STATUS UPDATE -- the NPU conclusion below still holds, but it is no longer
> the end of the story.** After this research pass ruled out the NPU, the
> **Adreno GPU via llama.cpp's OpenCL backend** was tried instead, and it
> works:
>
> - **Qwen3-VL-4B-Instruct: verified running on the Adreno X1-85**, 37/37
>   layers *and* the vision encoder on GPU, `graph splits = 1` (no silent CPU
>   fallback) -- receipts in
>   [`data/vlm_gpu_model/qwen3-vl-4b-instruct/_real_inference_smoke_log.md`](../data/vlm_gpu_model/qwen3-vl-4b-instruct/_real_inference_smoke_log.md).
> - **Qwen3-VL-8B-Instruct: also runs**, LLM fully on GPU, but the vision
>   encoder must be forced to CPU (`--no-mmproj-offload`) or it segfaults, and
>   it measured **worse than the 4B on every axis tested** -- receipts in
>   [`data/vlm_gpu_model/qwen3-vl-8b-instruct/_real_inference_smoke_log.md`](../data/vlm_gpu_model/qwen3-vl-8b-instruct/_real_inference_smoke_log.md).
> - **Recommendation: Qwen3-VL-4B on Adreno GPU.** The 8B is not worth its
>   cost on this device today.
>
> Crucially, the silent-CPU-fallback risk that killed the `GENIEX_LLAMACPP`
> Hexagon path (see "Why the llama.cpp/Hexagon path is a real risk" below) is
> **closed, not assumed away**, on the OpenCL backend: llama.cpp reports
> per-layer device assignment, KV-cache device, and graph-split counts, and
> declares unsupported ops loudly rather than quietly running them on CPU. On
> the 8B it did exactly that -- it named the unsupported CLIP ops and crashed
> instead of pretending.
>
> Why not QUAD's own Adreno path? QUAD does expose one
> (`generate_code(runtime="gpu")` -> `libQnnGpu`, and `QnnGpu.dll` is present
> in this machine's QAIRT 2.38.0.250901 SDK), but it cannot serve a VLM: it
> needs a `.dlc` from `convert_model` (gap 3/3b, broken), pre-built LLM/VLM
> artifacts are Genie **context binaries** which are NPU-only by design, and
> Genie's backend is hard-fixed to `QnnHtp` (confirmed in this repo's own
> `genie_config.json`) so there is no GPU LLM runtime to target. `qnn-net-run
> --backend QnnGpu.dll` is a single-graph runner with no KV cache, no decode
> loop and no tokenizer. QUAD's own `hardware_detect` for this device reports
> `available_runtimes: ["cpu", "npu"]` -- no GPU.

## Re-check 2026-08-06, prompted by Qualcomm's GenieX slides

Checked live against the two links on the slides. Three results, one of which
opens a genuinely new door.

**The slide's repo link is wrong.** `github.com/qualcomm/genie` is a **404**.
The real repo is [`github.com/qualcomm/GenieX`](https://github.com/qualcomm/GenieX)
(PyPI `geniex` **0.3.18**, released 2026-07-31, still "Developer Preview",
**source-only sdist -- no binary wheels**; the installer auto-provisions the
native SDK for Windows arm64 / Linux aarch64).

**1. The AI Hub QAIRT blocker has NOT cleared.** The Qwen3-VL-4B-Instruct
Compute page still reads, verbatim:

> "This model is currently not supported on any Compute chipset."

Snapdragon X Elite / X2 Elite are still listed as targets under GenieX-QAIRT,
and the only download offered is a generic Windows CLI app, not a
chipset-specific artifact. **The entire NPU conclusion below stands unchanged.**

**2. GenieX's GPU path is the path this project already uses.** Its
`device_map` aliases resolve as:

| `device_map` | What it actually is |
|---|---|
| `cpu` | pure CPU |
| **`gpu`** | **"Adreno via OpenCL with layer offloading"** -- i.e. `ggml-opencl`, device `GPUOpenCL` |
| `npu` | pinned single-session HTP; "deterministic, slower on LLMs" |
| `hybrid` | "llama_cpp per-tensor HTP+CPU scheduler" |
| `auto` | defaults to NPU |

So GenieX's `gpu` is **the same `ggml-opencl` backend, on the same
`GPUOpenCL` device**, that
`data/vlm_gpu_model/*/_real_inference_smoke_log.md` already verifies. GenieX
is a wrapper over it, not a different or better backend. **This is
confirmation, not a change of direction** -- and it means the 8B vision-encoder
blocker (OpenCL flash-attention has no head_dim 72 kernel) would be *identical*
under GenieX. Its docs say nothing about VLM/mmproj offload, so there is no
`--no-mmproj-offload` equivalent documented.

Also confirmed: **Q4_0 is the officially optimized quant for this backend**
("optimized for weights using the Q4_0 quantization scheme", per
[Qualcomm's own OpenCL-backend announcement](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu)) --
retroactively validating the quant choice already made here.

**3. The interesting find: `hybrid` runs GGUF on the Hexagon NPU with no
`convert_model` at all.** It "inspects each tensor and assigns it to whichever
registered backend supports the op (HTP for computable ops, CPU for
fallbacks)", and GenieX's notes report **~90 tok/s prefill vs ~60 tok/s for
pinned NPU**. If that holds on this machine it routes around **gap 3/3b
entirely** -- the broken-`convert_model` blocker that forced the text brain onto
a pre-built Qualcomm context binary (Phase 2a) and that blocks every other
model here. It would apply to the *text* brain first, not VLMs.

**Do not adopt it on those numbers.** This is precisely the
`GENIEX_LLAMACPP`/Hexagon path this document already flags as risky, and
`hybrid`'s *design* -- silently assigning unsupported ops to CPU -- is the
documented silent-fallback failure mode with a friendly name. The quoted
throughput is Qualcomm's own, on unstated hardware. It needs the same evidence
standard already applied twice in this repo: per-layer device assignment,
graph-split counts, and a real op-level breakdown showing what actually landed
on HTP versus CPU. That is a bounded, worthwhile experiment -- and unlike
`convert_model`, it is not a known-dead end.

---

Research pass only -- no code changed. Answers the question "can we add a
vision-language model to the NPU fast-brain path with similarly low friction
to Phi-3.5-mini-instruct (see
[`superpowers/deploy-local-brain-npu.md`](../superpowers/deploy-local-brain-npu.md))?"
Every claim below has a real source, checked live (not from training-data
memory) since this space moves fast. **Bottom line: not yet -- no VLM has a
verified, downloadable NPU artifact today. Revisit periodically, see
"What would change this."**

## Candidates checked

| Model | Params | Gated? | Backend (once available) | AI Hub Compute-chipset status |
|---|---|---|---|---|
| [Qwen3-VL-4B-Instruct](https://aihub.qualcomm.com/compute/models/qwen3_vl_4b_instruct) | 4B | No (Apache 2.0) | **GenieX + QAIRT** (NPU-native, same kind of path as Phi-3.5-mini-instruct) | Lists Snapdragon X Elite/X2 Elite as targets, but page states *"This model is currently not supported on any Compute chipset."* No downloadable artifact. |
| [Qwen3-VL-8B-Instruct](https://aihub.qualcomm.com/models) | 8B | No (Apache 2.0) | Same as above (per [v0.58.0 release notes](https://github.com/qualcomm/ai-hub-models/releases)) | Same "not supported" status as the 4B variant. |
| [Qwen2.5-VL-7B-Instruct](https://aihub.qualcomm.com/models) | 7B | No (Apache 2.0) | Unclear -- shown in a GenieX CLI example (`geniex infer ai-hub-models/Qwen2.5-VL-7B-Instruct`), but a [2025-09 feature request](https://github.com/qualcomm/ai-hub-models/issues/226) for the near-identical Qwen2-VL-7B was closed with Qualcomm saying VL variants "aren't currently underway" -- the CLI example may be aspirational docs, not a live bundle. | Not independently confirmed. |
| [Gemma-4-E2B-it](https://aihub.qualcomm.com/compute/models/gemma_4_e2b_it) / [E4B-it](https://aihub.qualcomm.com/compute/models/gemma_4_e4b_it) | 2B / 4B | No (Apache 2.0 model; Qualcomm GenAI usage terms apply) | **GenieX + llama.cpp** (confirmed via [v0.57.0 release notes](https://github.com/qualcomm/ai-hub-models/releases/tag/v0.57.0): listed under "VLMs & LLMs (GenieX accelerated by Llama CPP)", not the QAIRT section) | Same "not supported on any Compute chipset" disclaimer as Qwen3-VL, despite listing Snapdragon X Elite/X2 Elite as targets. |

**All four show the identical AI Hub pattern**: chipset listed as a target,
but the live site says "not supported on any Compute chipset." Read as: a
real recipe exists in the `qai_hub_models` repo (confirmed present --
`qai_hub_models.models.gemma_4_e4b_it` etc. -- and self-exportable via the
AI Hub cloud compiler, Phase 2b-style, per the existing deployment doc), but
**no pre-built, verified, downloadable artifact exists yet for any of
them** -- unlike Phi-3.5-mini-instruct, which shipped a ready QNN context
binary directly.

## The bigger finding: Qualcomm shipped a new SDK, GenieX, since the text brain was built

[GenieX](https://github.com/qualcomm/GenieX) (PyPI `geniex`, v0.3.18,
released 2026-07-31) is a real successor to the Genie SDK `NpuFastBrain`
already uses -- pip-installable, Windows ARM64 wheels, explicit LLM+VLM
framing, "Developer Preview" status badge. It exposes **two backends**:

- **`GENIEX_QAIRT`** -- the same NPU-native, compiled-context-binary
  approach this project already uses. No silent-CPU-fallback risk (Genie's
  backend is fixed to `QnnHtp`, same as documented in
  `data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md`).
- **`GENIEX_LLAMACPP`** -- runs GGUF models via llama.cpp's Hexagon NPU
  backend. Qwen3-VL/Gemma-4 GGUF weights are real and downloadable today
  (e.g. [`Qwen/Qwen3-VL-4B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)),
  so *something* runs today with zero AI Hub involvement -- but see the
  risk below before trusting what device it actually runs on.

## Why the llama.cpp/Hexagon path is a real risk, not just "less mature"

llama.cpp's own Hexagon-NPU backend docs mark it
**"(experimental)"** ([`docs/backend/snapdragon/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)),
document only text-only LLM examples (no VLM example at all, despite
GenieX pointing VLMs at this backend), and an independent, detailed
real-world report
([macroco.de, "The AI laptop that could not"](https://macroco.de/en/der-ki-laptop-der-keiner-ist-mein-snapdragon-x-elite-npu-debakel/))
found that `GGML_OP_MUL_MAT` isn't fully implemented on this backend on a
Snapdragon X Elite laptop: llama.cpp *reported* NPU/GPU offload while
actually running ~99% of inference on CPU. That's a **silent fallback**,
not a crash -- exactly the failure mode
[`deploy-local-brain-npu.md`'s R6](../superpowers/deploy-local-brain-npu.md)
flags as the proven trap on this hardware/toolchain family (same class of
bug `convert_model` attempt 3 caught for the text model), and exactly why
Phase 3 required two independent real-execution-proof signals
(`QnnGraph_execute` logs + a QNN profiler capture) before any router code
was written for `NpuFastBrain`. Trusting `GENIEX_LLAMACPP`'s reported
device placement without that same level of independent verification would
violate this project's own receipts rule.

## Bottom line

**No VLM currently clears this project's bar** (a real, inspectable,
verified-on-NPU artifact -- the same bar Phi-3.5-mini-instruct cleared).
Two independent reasons stack up: (1) no VLM has a downloadable QAIRT-native
artifact from AI Hub yet -- every candidate says "not supported on any
Compute chipset" live, regardless of backend; (2) the one backend that *does*
run something today without AI Hub (`GENIEX_LLAMACPP`) has a documented,
independently-reproduced silent-CPU-fallback bug on this exact chip family.

If a VLM is ever added, **Qwen3-VL-4B-Instruct is the model to revisit
first** -- it's the only 4-8B-range candidate on the safer `GENIEX_QAIRT`
track (per the v0.57.0 release notes), Apache-2.0/ungated, and already in
Qualcomm's own recipe repo waiting on either an AI Hub self-compile
(Phase 2b-equivalent) or a future pre-built binary drop (Phase 2a-equivalent).
Gemma-4-E2B/E4B-it should stay lower priority specifically because of the
llama.cpp backend risk, not because of size or licensing.

## What would change this assessment

- AI Hub actually publishing a downloadable QAIRT artifact for
  Qwen3-VL-4B-Instruct (watch its
  [Compute model page](https://aihub.qualcomm.com/compute/models/qwen3_vl_4b_instruct)
  for the "not supported" disclaimer to clear).
- An independent, real (not self-reported) confirmation that
  `GENIEX_LLAMACPP` actually executes on the Hexagon NPU on Windows
  Snapdragon X Elite for *any* model -- e.g. a `QnnGraph_execute`-equivalent
  log or profiler capture, the same standard of evidence Phase 3 already
  applied to the text brain -- which would de-risk the Gemma-4/Qwen3-VL
  llama.cpp path too.
- GenieX graduating out of "Developer Preview."
