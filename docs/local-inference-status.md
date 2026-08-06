# Local inference status -- what runs where, as of 2026-08-06

A single place to see which model/hardware combinations are **verified
working** on this machine (Snapdragon X Elite X1E80100 / Adreno X1-85 /
Hexagon v73), which are **broken**, and what is **still unproven**.

Every claim here traces to a receipt under `data/`. Nothing in this file is
projected, estimated, or taken from a vendor datasheet.

---

## The short answer

**The Adreno GPU runs both LLMs and VLMs. It is the only device that runs
both.**

| | CPU | **GPU (Adreno X1-85)** | NPU (Hexagon v73) |
|---|---|---|---|
| **LLM** | works, fastest | **works** | works, but only via a pre-compiled Qualcomm artifact |
| **VLM** | works, slow | **works** | no path at all |

---

## LLM -- Phi-3.5-mini-instruct (3.8B)

Same model on all four rows. Same prompt, same machine.

| Path | Artifact | Short (~35 tok) | Sustained (512 tok) | Cold load | TTFT |
|---|---|---|---|---|---|
| **CPU** | GGUF Q4_0 | **39.8 tok/s** | **32.1 tok/s** | -- | 0.1 s |
| **GPU** | GGUF Q4_0 | 28.7 tok/s | 25.8 tok/s | 3.0 s | 0.2-0.3 s |
| **NPU** (`NpuFastBrain`, in use today) | Genie w4a16 context binary | -- | **10.6 tok/s** (n=2180) | 13.1 s | ~142 ms |
| NPU via llama.cpp | GGUF Q4_0 on HTP0 | **crashes** | **crashes** | -- | -- |

Receipts: `data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md`
(the Genie path) and `_real_geniex_hybrid_log.md` (the GGUF paths).

**The uncomfortable headline: the NPU brain currently in the router is ~3x
slower than running the same model as a GGUF on the CPU.** Two things stop that
from being a conclusion:

1. **Power was never measured**, and power is the NPU's whole justification in a
   power-aware design. A 3x throughput deficit at a fraction of the wattage may
   well be the right trade for a background fast brain.
2. The quantizations differ (w4a16 weight-shared vs generic Q4_0), so this is
   not a controlled comparison of silicon.

**Do not change the router's device default until perf-per-watt is measured.**

---

## VLM -- Qwen3-VL

### Qwen3-VL-4B-Instruct -- the recommended VLM

Fully on the GPU: **37/37 layers, the KV cache, and the vision encoder** all on
`GPUOpenCL`, with `graph splits = 1` over 766 nodes.

| | Prefill | Decode | Vision encode |
|---|---|---|---|
| GPU, warm | 68.96 tok/s | 21.17 tok/s | 2742 ms |
| GPU, cold | 49.13 tok/s | 17.23 tok/s | 4652 ms |
| CPU control | 43.33 tok/s | 13.73 tok/s | 4289 ms |

The GPU win over CPU is real but **modest** -- roughly 1.25x decode cold,
~1.5x warm. The value of this path is not raw speed; it is that it works, is
*verifiably* on an accelerator, and leaves the CPU free.

### Qwen3-VL-8B-Instruct -- runs, but a downgrade

Needs `--no-mmproj-offload` (vision encoder on CPU) or it **segfaults**.
45.45 tok/s prefill, 13.26 tok/s decode. Slower than the 4B, needs a workaround
flag, loses GPU vision acceleration entirely, and was no more accurate.

Root cause, pinned exactly: the 8B vision tower is **head_dim 72**
(`n_embd 1152 / n_head 16`), and llama.cpp's OpenCL flash-attention kernels
cover 64 and 128 but not 72. Without FA, vision attention materializes
`SOFT_MAX f32 [8464 8464 16 1]` -- a **4.58 GB single tensor** against the
device's hard 2048 MB max-alloc. The 4B's vision tower is head_dim 64, which is
exactly why it works.

Receipts: `data/vlm_gpu_model/qwen3-vl-{4b,8b}-instruct/_real_inference_smoke_log.md`.

### Shared VLM limitation -- grounding

Both models get colour and shape right and **misplace position**. Ground truth
measured from the pixels: the test card's red circle is centred at
(0.50W, 0.50H); the 4B calls it "lower center", the 8B "lower-right".

Fine for description, classification, OCR-ish and Q&A. **Not trustworthy for
grounding** (bounding boxes, "which side", "point at X") without
`--image-min-tokens 1024`, which is expensive -- on the 8B it took prompt
tokens 270 -> 1045 and halved decode to 6.46 tok/s.

---

## How to actually run them

Both need the Adreno OpenCL ICD, which is not registered system-wide (this
avoids needing admin rights):

```bash
export OCL_ICD_FILENAMES="<repo>/.llama-cpp-opencl/extracted/OpenCL_adreno.dll"
```

**VLM** -- via `llama-mtmd-cli` (build 10291 / `803b7fcae`):

```bash
./llama-mtmd-cli.exe \
  -m       data/vlm_gpu_model/qwen3-vl-4b-instruct/raw/Qwen3-VL-4B-Instruct-Q4_0.gguf \
  --mmproj data/vlm_gpu_model/qwen3-vl-4b-instruct/raw/mmproj-Qwen3VL-4B-Instruct-f16.gguf \
  --image <image.png> -p "Describe this image." \
  -c 4096 -ngl 99
```

**LLM** -- either `llama-cli` directly, or GenieX:

```bash
geniex-py chat <model.gguf> --device gpu -p "..." --n-ctx 4096
```

**`-c` / `--n-ctx` is mandatory.** Without it, llama.cpp's auto-fit sizes
context to 81920, consumes ~12 GB of KV cache, then fails a 316 MB
compute-buffer allocation with a misleading error.

**Q4_0 is deliberate.** Qualcomm's OpenCL backend is specifically optimized for
Q4_0. It is why the 4B LLM weights come from unsloth rather than Qwen's own
GGUF repo, which ships only Q4_K_M/Q8_0.

---

## What is broken, and why not to retry it

| Thing | Status | Signature to re-test against |
|---|---|---|
| `convert_model` (gaps 3/3b) | broken, two independent paths | see `data/convert_model/_real_attempts_log.md` -- do not retry |
| GGUF on Hexagon NPU (GenieX `hybrid` / `llama_cpp:HTP0`) | **crashes, 3/3 deterministic** | `dspqueue_read failed: 0x00000072` at `ggml-hexagon.cpp:1583` |
| GenieX driving a **VLM** | **broken in 0.3.18**, both 4B and 8B | `mtmd_tokenize: number of media markers in text (0)` despite markers present in `prompt_utf8` |
| 8B vision encoder on GPU | segfault | `failed to allocate 4372.52 MiB`; needs OpenCL FA at head_dim 72 |
| VLM on NPU via AI Hub/QAIRT | no artifact exists | AI Hub still: *"This model is currently not supported on any Compute chipset."* |
| QUAD's own Adreno path (`libQnnGpu`) | unusable for LLM/VLM | needs a `.dlc` from the broken `convert_model`; Genie is HTP-fixed; `qnn-net-run` has no KV cache or decode loop |

Notably, **the Hexagon crash is not the silent-CPU-fallback bug** the research
doc feared. The NPU really initializes (v73 session, domain-id 3) and really
takes 1950 MiB of repacked weights before dying loudly in the FastRPC
transport. The backend is honest; it is just unreliable here.

---

## Integration status -- the real gap

**None of the GPU work is wired into the router.** It is hardware verification
only.

- `routing/brains.py` has exactly one real-hardware `Brain`: `NpuFastBrain`.
- `routing/router.py:41-44` still selects `NpuFastBrain` for the AI PC tier.
- There is no `GpuLlmBrain`, no `GpuVlmBrain`, and no test coverage for either.
- The router has no notion of an image input at all, so a VLM cannot yet be
  routed even if the seam existed.

---

## Open questions, in priority order

1. **Perf-per-watt for NPU vs GPU vs CPU.** Blocks every device-default
   decision, including whether `NpuFastBrain` should stay the router's choice.
2. **A `Brain` seam for the GPU path** (`GpuLlmBrain` first -- it is a drop-in
   for the existing contract; `GpuVlmBrain` needs a router that can carry an
   image).
3. **Multi-chat serving.** `llama-server -np N` with continuous batching exists
   and is testable on the verified GPU path today. Genie's one-dialog-at-a-time
   C API has no equivalent, and a second concurrent `NpuFastBrain` currently
   fails at the C level (`err 1002`) with no guard in Python to prevent it.
4. Re-test GenieX VLM and the Hexagon backend on future releases, against the
   signatures in the table above.
