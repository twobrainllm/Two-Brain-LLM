# Qwen3-VL-4B-Instruct on Adreno GPU -- real verification log

`_mock: false` throughout. Every number and quoted string below came from an
actual run on this machine; the raw logs are kept under
`.llama-cpp-opencl/extracted/vlm_smoke_*.log` (gitignored -- they sit next to
the binaries they were produced by).

This is the **GPU (Adreno / OpenCL)** path, *not* the NPU path. It exists
because `docs/vlm-npu-research.md` found no VLM clears the NPU bar: no
downloadable QAIRT artifact for any VLM, and the one NPU-adjacent backend that
runs today (`GENIEX_LLAMACPP` / llama.cpp Hexagon) has a documented
silent-CPU-fallback bug. **Adreno OpenCL sidesteps both**: it is a mature,
non-experimental llama.cpp backend, and -- as recorded below -- its device
placement is verifiable from llama.cpp's own scheduler output, so the
silent-fallback risk that blocked the Hexagon path is actually closed here
rather than assumed away.

## Toolchain

- llama.cpp **build 10291 (`803b7fcae`)**, prebuilt Windows arm64 OpenCL/Adreno
  release, "built with Clang 20.1.8 for Windows arm64" (`llama-cli --version`,
  real).
- Unpacked to `.llama-cpp-opencl/extracted/` (gitignored; not vendored).
- Device, as llama.cpp reports it:
  `GPUOpenCL (Qualcomm(R) Adreno(TM) X1-85 GPU) - 15137 MiB free`,
  driver `OpenCL 3.0 QUALCOMM build: 863.0 Compiler DX.50.39.00`,
  FP16 support true, **max mem alloc size 2048 MB**, global mem 16161 MB.

### OpenCL ICD loading (no admin rights)

The Adreno OpenCL ICD is not registered system-wide on this machine. Rather
than modifying the registry (needs admin), the runtime is pointed at a copy of
the driver's own ICD via the standard Khronos loader env var:

```
OCL_ICD_FILENAMES=<repo>\.llama-cpp-opencl\extracted\OpenCL_adreno.dll
```

with the Khronos `OpenCL.dll` loader and the Adreno driver DLLs
(`OpenCL_adreno.dll`, `qcgpuarm64compilercore.so`, `kcl.dll`, `CB.dll`,
`adreno_utils.dll`, ...) beside the executable. Without this, real enumeration
output was:

```
Available devices:
  (none)
```

(`out2.txt`, `out4.txt` -- real, from the failed attempts before the ICD was
placed).

## Model files -- real downloads, mixed provenance

| File | Bytes (local) | Source |
|---|---|---|
| `raw/Qwen3-VL-4B-Instruct-Q4_0.gguf` | 2375774112 | `unsloth/Qwen3-VL-4B-Instruct-GGUF` |
| `raw/mmproj-Qwen3VL-4B-Instruct-f16.gguf` | 836180256 | `Qwen/Qwen3-VL-4B-Instruct-GGUF` (official) |

Two honest caveats, recorded rather than smoothed over:

1. **Provenance of the LLM file is not exactly pinned.** A live `HEAD` against
   `unsloth/.../Qwen3-VL-4B-Instruct-Q4_0.gguf` today returns
   `Content-Length: 2375774496` -- **384 bytes larger** than the local copy.
   Unsloth re-uploads GGUFs with updated metadata routinely, so this is almost
   certainly a metadata-only revision drift, not a different model. But the
   local file was not downloaded against a pinned revision SHA, so it cannot be
   asserted byte-identical to any current upstream file. *Future downloads here
   should pin `?revision=<sha>`.* The mmproj **does** match its source exactly
   (836180256 == 836180256).
2. **The LLM and the mmproj come from different repackagers.** This pairing is
   not upstream-blessed; it works (verified below), but it is a real deviation
   worth knowing about if accuracy ever looks off.

`Q4_0` is deliberate, not incidental: llama.cpp's Adreno OpenCL backend has
optimized Q4_0 paths. The official Qwen GGUF repo ships only `Q4_K_M`/`Q8_0`,
which is why the LLM came from unsloth at all.

## Attempt 1 -- FAILED, bad CLI argument

```
error: invalid argument: --single-turn
```

Real, from `vlm_smoke_run1.log` (returncode 1, 0.1s). Not a hardware finding --
that flag does not exist in this build of `llama-mtmd-cli`. Recorded only so
the run numbering below matches the raw logs.

## Attempt 2 -- FAILED on GPU, real OOM: llama.cpp's auto-fit overcommits KV cache

Ran with default context (no `-c`). Real error:

```
ggml_backend_opencl_buffer_type_alloc_buffer: failed to allocate 301.75 MiB
ggml_gallocr_reserve_n_impl: failed to allocate OpenCL buffer of size 316407808
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
```

**This is not a plain "model too big" failure, and the misleading part is worth
recording**: 316 MB is far below the device's 2048 MB max-alloc limit and its
15137 MiB of free memory, so the error message reads like it should have
succeeded. The real cause is upstream of it, in llama.cpp's `-fit on`
(default) auto-sizing, visible in the verbose log:

```
llama_context: n_ctx = 262144
common_params_fit_impl: projected to use 39467 MiB of device memory vs. 15137 MiB of free device memory
common_params_fit_impl: cannot meet free memory target of 1024 MiB, need to reduce device memory by 25353 MiB
common_params_fit_impl: context size reduced from 262144 to 81920 -> need 25372 MiB less memory in total
common_params_fit_impl: entire model can be fit by reducing context
llama_context: n_ctx = 81920
```

The heuristic declared 81920 context "fits", then the ~12 GB KV cache that
implies (36 layers x 8 KV heads x 128 head dim x 2 (K+V) x f16) plus 2.4 GB of
weights consumed the budget, leaving too little for the 301.75 MiB compute
buffer. **The auto-fit heuristic overcommits on this device.** Fix: pass `-c`
explicitly. This generalizes to any larger model on this GPU and is the single
most important operational finding in this file.

## Attempt 3 -- SUCCEEDED on GPU with explicit `-c 4096`

```
llama-mtmd-cli.exe \
  -m   data/vlm_gpu_model/qwen3-vl-4b-instruct/raw/Qwen3-VL-4B-Instruct-Q4_0.gguf \
  --mmproj data/vlm_gpu_model/qwen3-vl-4b-instruct/raw/mmproj-Qwen3VL-4B-Instruct-f16.gguf \
  --image data/vlm_gpu_model/qwen3-vl-4b-instruct/test_assets/test_image.png \
  -p "Describe this image." -c 4096 -ngl 99
```

`test_assets/test_image.png` is a generated 512x512 test card. Ground truth,
measured from the pixels (not eyeballed -- PNG decoded and centroids computed):

| Shape | x extent | y extent | Centroid |
|---|---|---|---|
| Blue square | 30-130 | 30-130 | **(0.16W, 0.16H)** -- upper-left |
| Red circle | 156-356 | 156-356 | **(0.50W, 0.50H)** -- dead centre |

Real generated output:

> The image displays a blue square in the upper left and a red circle in the
> lower center against a white background.

Re-verified live on a later date with the same command and binaries:

> This is a minimalist graphic featuring two simple geometric shapes on a plain
> white background.
> - **A blue square**: Located in the upper-left quadrant of the image. [...]
> - **A red circle**: Positioned in the lower-center of the image. [...]

**Colour and shape are correct; position is not.** The red circle is centred at
exactly (0.50W, 0.50H) -- the model calls it "lower center" in both runs. That
is a real, reproducible spatial error, not a wording quibble, and it is
consistent with the `--image-min-tokens 1024` grounding warning below: these
runs use the default 256 image tokens, under the 1024 the model needs for
grounding.

This is enough to prove the **pipeline** is real -- a broken or
silently-CPU-stubbed run does not produce correct colours, correct shapes, and
correct relative arrangement. It is **not** enough to claim spatial accuracy.
See `../qwen3-vl-8b-instruct/_real_inference_smoke_log.md`, where the 8B is
measurably *worse* on this same axis.

Note for future work: this test card is a weak grounding probe, because
"centre" vs "lower-centre" is a soft judgement. A card with shapes at
unambiguous, well-separated positions would be a sharper regression test.

### On-GPU confirmation (the silent-fallback check, closed with real evidence)

This is the check `docs/vlm-npu-research.md` demanded before trusting any
llama.cpp device claim, and the reason the Hexagon path was rejected. Real
lines from the verbose log:

```
llama_prepare_model_devices: using device GPUOpenCL (Qualcomm(R) Adreno(TM) X1-85 GPU) - 15137 MiB free
load_tensors: layer 0..35 assigned to device GPUOpenCL, is_swa = 0
load_tensors: offloading output layer to GPU
load_tensors: offloaded 37/37 layers to GPU
llama_kv_cache: layer 0..35: dev = GPUOpenCL
clip_ctx: CLIP using OpenCL backend
reserve_compute_meta:     OpenCL compute buffer size =   322.49 MiB
reserve_compute_meta:        CPU compute buffer size =    24.93 MiB
reserve_compute_meta: graph splits = 1, nodes = 766
ggml_opencl: compiling fa f32_f16 split
```

Four independent signals, not one self-report:

1. **All 37/37 layers** assigned to `GPUOpenCL`, per-layer, named individually.
2. **KV cache** also on `GPUOpenCL` (a CPU-resident KV cache would betray a
   fallback even with weights offloaded).
3. **`graph splits = 1` over 766 nodes** -- the decisive one. A silent CPU
   fallback shows up as the scheduler splitting the graph to hand unsupported
   ops back to CPU. One split over the whole graph means no op fell back. This
   is exactly the signal that was *absent* in the macroco.de Hexagon report
   cited in the research doc.
4. **The vision encoder is on GPU too** (`clip_ctx: CLIP using OpenCL
   backend`), and OpenCL flash-attention kernels are actually compiled at
   runtime (`compiling fa f32_f16 split`).

The only CPU residency is a 24.93 MiB compute buffer and `token_embd.weight`
(`cannot be used with preferred buffer type CPU_REPACK, using CPU instead`) --
expected, and not inference compute.

### Attempt 4 -- CPU control run, for a real comparison

Same command with `-ngl 0`. Verified to be a genuine control: `load_tensors:
layer 0..35 assigned to device CPU`.

| | prompt eval | decode | mtmd (vision) encode |
|---|---|---|---|
| **GPU** (`-ngl 99`, cold) | 5699.57 ms / 280 tok = **49.13 tok/s** | 1102.85 ms / 19 = **17.23 tok/s** | 4652 ms |
| **CPU** (`-ngl 0`) | 6462.17 ms / 280 tok = **43.33 tok/s** | 1675.23 ms / 23 = **13.73 tok/s** | 4289 ms |
| **GPU** (warm, later re-verify) | 3915.32 ms / 270 tok = **68.96 tok/s** | 5669.46 ms / 120 = **21.17 tok/s** | 2742 ms |

Read this honestly: **the GPU win is real but modest** -- roughly 1.25x decode
and 1.13x prefill cold, improving to ~1.54x decode / ~1.59x prefill once
kernels and page cache are warm. The Adreno X1-85 is not delivering an
order-of-magnitude speedup over these CPU cores, and on the *cold* run the
vision encoder was actually slightly slower than CPU (4652 vs 4289 ms). The
value of this path is not raw speed; it is that it (a) works at all for a VLM,
(b) is *verifiably* on an accelerator, and (c) frees the CPU.

Numbers vary run-to-run (kernel compilation is cached after first use); the
warm row is a later re-verification of the same binaries and files, not a
correction to the first two rows.

## Known warnings (real, benign, worth knowing)

```
load: control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load_hparams: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks
load_hparams: if you encounter problems with accuracy, try adding --image-min-tokens 1024
```

The second one is **not hypothetical -- it is the observed failure above**. At
the default image-token count the model misplaces a dead-centre circle as
"lower center". Anything depending on position (bounding boxes, "point at X",
"which side is the defect on") must pass `--image-min-tokens 1024` and be
re-verified. That raises prefill cost substantially -- measured on the 8B, the
same change took prompt tokens from 270 to 1045 and cut decode from 13.26 to
6.46 tok/s.

`llama-mtmd-cli` prints `WARN: This is an experimental CLI for testing
multimodal capability.` This refers to the **CLI wrapper**, not the OpenCL
backend.

## Status

Real, reproducible, verified-on-GPU VLM inference. What does **not** yet exist:
no `Brain`-seam implementation in `src/`, no tests, and this path is not wired
into the router. This file records the hardware/toolchain verification only.
