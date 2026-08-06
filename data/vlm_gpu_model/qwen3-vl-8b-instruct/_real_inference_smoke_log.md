# Qwen3-VL-8B-Instruct on Adreno GPU -- real verification log

`_mock: false` throughout. Same machine, same binaries, same test card as
`../qwen3-vl-4b-instruct/_real_inference_smoke_log.md` -- read that first; this
file only records what differs at 8B.

**Result: it runs, and the LLM is fully on the Adreno GPU -- but only with the
vision encoder forced onto CPU (`--no-mmproj-offload`). Without that flag it
segfaults.** And the 8B is *worse* than the 4B at the one thing being checked.

## Files -- real downloads, both byte-exact

| File | Bytes | `curl` result |
|---|---|---|
| `raw/Qwen3-VL-8B-Instruct-Q4_0.gguf` | 4787333984 | `http=200 size=4787333984 time=160.9s` |
| `raw/mmproj-F16.gguf` | 1159030336 | `http=200 size=1159030336 time=196.0s` |

Both from `unsloth/Qwen3-VL-8B-Instruct-GGUF`, fetched with:

```
curl -L --fail -o <file> https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/<file>
```

Server-reported `Content-Length` was checked *before* download and matches the
bytes on disk exactly for both files. **Better provenance than the 4B**, whose
LLM file is 384 bytes off its upstream (see that log). Still not
revision-pinned -- `?revision=<sha>` should be used in future.

`Q4_0` again deliberate (Adreno OpenCL has optimized Q4_0 paths). Qwen's own
GGUF repo ships only `Q4_K_M`/`Q8_0` at 8B, hence unsloth.

Aside: `hf download` was tried first and **failed silently, exit 1, no output
whatsoever**. Cause: `hf.exe` is on `PATH` from `Python313\Scripts`, but that
interpreter has no `huggingface_hub` installed, so the console entry point dies
before printing anything. Not a network or auth problem. `curl` was used
instead -- which also matches how the phi-3.5 NPU artifact was fetched.

## Attempt 1 -- FAILED: segfault in the CLIP graph (exit 139)

Exactly the 4B's working command (`-c 4096 -ngl 99`). The **LLM half loaded
fine** -- this is not a model-too-big failure:

```
llama_prepare_model_devices: using device GPUOpenCL (Qualcomm(R) Adreno(TM) X1-85 GPU) - 15137 MiB free
load_tensors: offloaded 37/37 layers to GPU
llama_context: n_ctx = 4096
sched_reserve:     OpenCL compute buffer size =   304.75 MiB
sched_reserve: graph splits = 2
```

The **vision encoder** is what breaks:

```
clip_ctx: CLIP using OpenCL backend
reserve_compute_meta: graph splits = 55, nodes = 853
ggml_backend_opencl_buffer_type_alloc_buffer: failed to allocate 4372.52 MiB
ggml_gallocr_reserve_n_impl: failed to allocate OpenCL buffer of size 9468981504
warmup: WARNING: the CLIP graph uses unsupported operators by the backend
warmup:          list of unsupported ops (backend=OpenCL):
warmup:         SOFT_MAX: type = f32, ne = [8464 8464 16 1]
warmup:             CONT: type = f32, ne = [8464 72 16 1]
warmup:          PERMUTE: type = f32, ne = [72 8464 16 1]
warmup:             ROPE: type = f32, ne = [72 16 8464 1]
warmup: flash attention is disabled
warmup: please report this on github as an issue
warmup: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118
```

**Root cause, precisely:** llama.cpp's OpenCL backend does not implement flash
attention for this CLIP graph, so vision attention materializes the full
score matrix. `SOFT_MAX f32 [8464 8464 16 1]` is
`8464 x 8464 x 16 x 4 B = 4.58 GB` **in a single tensor** -- more than double
the device's hard `max mem alloc size: 2048 MB`, before the 9.47 GB total
buffer is even considered. The 55 graph splits (vs 1 for the LLM) are the same
story: many CLIP ops have no OpenCL kernel and bounce to CPU.

This is the *good* version of the silent-fallback risk `docs/vlm-npu-research.md`
warned about: llama.cpp declares the unsupported ops loudly instead of quietly
running them on CPU while claiming GPU.

**The failure is a segfault, not a clean error.** llama.cpp does not recover
from the failed allocation -- the process dies with exit 139. Any `Brain`
implementation must pass the flag up front; it cannot catch this and retry.

## Attempt 2 -- FAILED: `--image-max-tokens 1024` does not help

Hypothesis: the 8464-token figure is a *warmup* graph sized to the projector's
maximum, so capping image tokens should shrink it. **Wrong** -- real result,
byte-identical failure:

```
clip_ctx: CLIP using OpenCL backend
reserve_compute_meta: graph splits = 55, nodes = 853
ggml_backend_opencl_buffer_type_alloc_buffer: failed to allocate 4372.52 MiB
```

Same 4372.52 MiB, same segfault. The CLIP warmup graph is sized from the
projector's own config, not from `--image-max-tokens`. Recorded so nobody
retries this.

## Attempt 3 -- SUCCEEDED with `--no-mmproj-offload`

```
llama-mtmd-cli.exe \
  -m   data/vlm_gpu_model/qwen3-vl-8b-instruct/raw/Qwen3-VL-8B-Instruct-Q4_0.gguf \
  --mmproj data/vlm_gpu_model/qwen3-vl-8b-instruct/raw/mmproj-F16.gguf \
  --image data/vlm_gpu_model/qwen3-vl-8b-instruct/test_assets/test_image.png \
  -p "Describe this image." -c 4096 -ngl 99 --no-mmproj-offload
```

Real on-device evidence -- **LLM still fully on GPU**, only the vision encoder moved:

```
llama_prepare_model_devices: using device GPUOpenCL (Qualcomm(R) Adreno(TM) X1-85 GPU) - 15137 MiB free
load_tensors: offloaded 37/37 layers to GPU
clip_ctx: CLIP using CPU backend
reserve_compute_meta: graph splits = 1, nodes = 853
```

`graph splits = 1` over the LLM graph -- no op fell back. Real output:

> This image contains two simple geometric shapes on a plain white background.
> - In the upper-left portion of the image, there is a solid blue square.
> - In the lower-right portion of the image, there is a solid red circle.
> The shapes ... are positioned diagonally from each other.

### Accuracy -- the 8B is WORSE than the 4B here

Measured ground truth (pixels, not eyeballed): blue square centroid
**(0.16W, 0.16H)**, red circle centroid **(0.50W, 0.50H)** -- dead centre.

| Model | image tokens | Red circle called | Correct? |
|---|---|---|---|
| 4B | 256 (default) | "lower center" | wrong in 1 axis (it is centred) |
| **8B** | 256 (default) | **"lower-right", "diagonally"** | **wrong in 2 axes** |
| 8B | 1024 (`--image-min-tokens 1024`) | "lower-center ... below and to the right of the square" | wrong in 1 axis |

Both models get colour and shape right and both misplace the circle. **Going
from 4B to 8B did not buy accuracy on this probe -- it lost some.** Raising the
8B to the recommended 1024 image tokens only claws it back to where the 4B
already was at 256.

Caveat, stated plainly: this is **one prompt on one deliberately simple test
card**, and "centre" vs "lower-centre" is a soft judgement. It is enough to say
the 8B shows no accuracy win here; it is *not* enough to conclude the 8B is a
worse model. A real comparison needs a proper eval set.

### Real timings

| Config | prefill | decode | vision encode |
|---|---|---|---|
| 8B, 256 img tokens, vision on CPU | 5940.34 ms / 270 tok = **45.45 tok/s** | 6714.23 ms / 89 = **13.26 tok/s** | ~1.5 s |
| 8B, 1024 img tokens, vision on CPU | 32924.13 ms / 1045 tok = **31.74 tok/s** | 11148.33 ms / 72 = **6.46 tok/s** | **20851 ms** |
| *4B, 256 img tokens, vision on GPU (warm)* | *68.96 tok/s* | *21.17 tok/s* | *2742 ms* |

Cold-load time ~10.6 s for the 8B (vs ~5.9 s warm for the 4B).

The 1024-image-token row is the expensive one: vision encoding alone is **20.9
seconds on CPU**, and decode more than halves. That combination -- vision
forced to CPU *and* grounding needing 4x the image tokens -- is what makes 8B
grounding impractical on this device today.

## Attempt 4 -- root cause pinned exactly: vision head_dim 72

Why does CLIP-on-GPU work for the 4B and not the 8B? Real `load_hparams` from
both runs:

| | 4B vision tower | 8B vision tower |
|---|---|---|
| `n_embd` | 1024 | 1152 |
| `n_head` | 16 | 16 |
| **head_dim** | **64** | **72** |
| `n_layer` | 24 | 27 |
| Flash attention | `flash attention is enabled` | `flash attention is disabled` |

The 4B log shows the kernel being built for exactly that head dim:

```
ggml_opencl: lazy-compiling flash_attn prepass for DK=64 DV=64
```

llama.cpp's OpenCL backend compiles FA kernels for **DK=DV=64** (the 4B vision
tower) and **128** (both models' LLM). **72 is not supported**, so the 8B's
vision attention falls back to a materialized score matrix.

The 4.58 GB tensor then follows arithmetically. Also real, from the log:

```
get_dummy_batch: warmup with image size = 1472 x 1472
```

1472 / patch_size 16 = 92 patches per side, 92^2 = **8464** -- the exact figure
in `SOFT_MAX [8464 8464 16 1]`. The warmup image size derives from the mmproj's
own `image_max_pixels: 4194304`, **not** from `--image-max-tokens` (verified:
that flag left `get_dummy_batch` at 1472 x 1472, which is why Attempt 2
failed identically).

## Attempt 5 -- `--no-warmup` "works" and MUST NOT be used

Skipping warmup skips the oversized reserve. It genuinely runs:

```
clip_ctx: CLIP using OpenCL backend
mtmd batch encoding done in 5543 ms
```

exit 0, correct output, vision encoder on the GPU. **This is a trap.** The
allocation was never made smaller -- it was only deferred to the first real
image big enough to need it. Probed with a generated 1536x1536 image
(`test_assets/test_image_large.png`, 96x96 patches):

```
clip_ctx: CLIP using OpenCL backend
image_tokens->nx = 48 / ny = 48
ggml_backend_opencl_buffer_type_alloc_buffer: failed to allocate 5184.00 MiB
ggml_gallocr_reserve_n_impl: failed to allocate OpenCL buffer of size 11197366272
-> Segmentation fault (exit 139)
```

`--no-warmup` converts a **deterministic startup failure** into a
**data-dependent crash in production**, triggered by nothing worse than a user
supplying a larger photo. That is strictly worse than the honest failure. Do
not use it to "fix" this.

## Attempt 6 -- even when GPU vision works, it is SLOWER

Same 512x512 image, same everything, only the vision encoder's device differs:

| Vision encoder | encode time | CLIP graph splits |
|---|---|---|
| **CPU** (`--no-mmproj-offload`) | **4011 ms** | **1** |
| GPU (`--no-warmup`, OpenCL) | 5543 ms | **55** |

And end-to-end, GPU vision was worse on prefill too (30.53 vs 45.45 tok/s) with
a slower load (18.1 s vs 10.7 s). The 55 splits explain it: without FA, most of
the CLIP graph's ops have no usable OpenCL kernel at these shapes and bounce
back to CPU anyway -- so the "GPU" vision path pays constant round-trip cost to
do most of the work on CPU regardless.

**There is no performance prize waiting behind this bug.** Even if the memory
limit were lifted, GPU vision on the 8B would still be slower until the FA
kernel gap is closed.

For completeness, the CPU vision path does handle the large image correctly
(exit 0, correct "upper left" square / "lower right" circle -- both genuinely
correct for that generated card), but took **101716 ms** to encode it. Large
images are impractical on the 8B on this device by either route.

## Can the GPU vision encoder be fixed? Not from here.

Ranked, with real reasons:

1. **The actual fix is upstream: add head_dim 72 to ggml-opencl's flash-attention
   kernels.** Everything else is a workaround. Note this is **not** a
   "just upgrade llama.cpp" situation -- the binaries in use are build
   **10291 (`803b7fcae`)**, dated the same day as this testing, so we are
   already on a current build and the gap is live upstream.
2. **Shrink `image_max_pixels` in the mmproj GGUF metadata** (4194304 today).
   This is the only lever reachable locally without recompiling, and it would
   shrink the warmup dummy image quadratically. **Not recommended**: it caps
   real image resolution too, which directly worsens grounding -- already this
   model's weakest axis -- and buys a vision path that Attempt 6 shows is
   slower than CPU anyway.
3. **`--no-warmup`** -- rejected, see Attempt 5.

**Recommended: `--no-mmproj-offload`.** LLM on GPU, vision on CPU. It is
faster, has 1 graph split instead of 55, and fails predictably rather than
mid-request. Or simply use the 4B, whose vision tower is head_dim 64 and
therefore runs fully on the GPU with flash attention today.

## Bottom line

| | 4B | 8B |
|---|---|---|
| LLM on Adreno GPU | yes, 37/37 | yes, 37/37 |
| Vision encoder on GPU | **yes** | **no -- segfaults, CPU only** |
| Decode (256 img tokens) | 21.17 tok/s (warm) | 13.26 tok/s |
| Position accuracy | wrong in 1 axis | wrong in 2 axes |

**The 8B is a downgrade on this hardware on every axis measured**: slower,
requires a workaround flag, loses GPU vision-encoder acceleration entirely, and
was no more accurate on the one probe run. It works, and this log proves it
works, but there is currently no measured reason to prefer it over the 4B here.

If the 8B is wanted anyway, the blocker to fix upstream is now pinned to one
line: **llama.cpp's OpenCL flash-attention kernels support head_dim 64 and 128,
but the 8B vision tower is head_dim 72** (see Attempt 4). Adding that case
removes the 4.58 GB single-tensor allocation and would let the vision encoder
back onto the GPU -- though Attempt 6 shows the CLIP graph would *still* need
OpenCL kernels for its other ops before that is actually faster than CPU.

## Status

Hardware/toolchain verification only. No `Brain`-seam implementation in `src/`,
no tests, not wired into the router -- same as the 4B.
