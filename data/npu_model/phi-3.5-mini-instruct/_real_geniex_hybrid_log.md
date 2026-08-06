# GenieX `hybrid` / GGUF-on-Hexagon experiment -- real log

`_mock: false` throughout. Ran on branch `(hollowbyte)-test/hybrid-npu-gguf`.

**Question:** does GenieX's llama.cpp runtime execute a GGUF on the Hexagon NPU,
routing around gap 3/3b (`convert_model` broken) and freeing the fast brain from
"whatever Qualcomm pre-compiled"?

**Answer: no. The Hexagon path is broken on this machine** -- the model loads
fully onto HTP0 and then dies at generate time with a FastRPC DSP-queue error,
deterministically. **But the experiment produced a more consequential result
anyway:** the same GGUF on CPU and on the Adreno GPU is **2.4x-3x faster than
the existing Genie/NPU fast brain**.

## Setup

- `geniex` **0.3.18** installed into a **separate** `.venv-geniex`
  (Python 3.12 ARM64). Deliberately *not* `.venv-npu` -- that venv is the
  verified working environment for `NpuFastBrain` and must not be disturbed.
- `geniex-py version`: SDK `v0.3.18`, bundled **QAIRT `v2.45.0.260326`**
  (between this machine's system QAIRT 2.38.0.250901 and `onnxruntime-qnn`'s
  2.48.40).
- Model: `bartowski/Phi-3.5-mini-instruct-GGUF` -> `Phi-3.5-mini-instruct-Q4_0.gguf`,
  `http=200 size=2182468896`. Q4_0 per GenieX's own note that Q4_K_M is
  suboptimal for HTP. Same model as the existing NPU brain, so the comparison
  is like-for-like on model identity (not on quantization -- see caveats).

`geniex-py devices` on this machine, real output:

```
llama_cpp:
  GPUOpenCL        Qualcomm(R) Adreno(TM) X1-85 GPU
  HTP0             Hexagon
  CPU              Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Oryon(TM) CPU
qairt:
  NPU              Qualcomm NPU (QAIRT)
```

Note `GPUOpenCL` is the **same device id** the VLM work already verifies -- more
confirmation that GenieX's GPU path and this project's GPU path are one backend.

## Two documentation errors found, both material

1. **`--device npu` is NOT "pinned HTP".** GenieX's `notes/run.md` describes it
   as "Pinned single-session HTP". In practice it routes to the **qairt**
   plugin and rejects GGUF outright:

   ```
   ValueError: .gguf models are not supported by device_map='npu' (QAIRT/NPU).
   Use device_map='auto' or 'hybrid' instead.
   ```

   To actually target llama.cpp's Hexagon backend you must use the explicit
   `plugin:device` form: `--device llama_cpp:HTP0`.

2. **`hybrid` is NOT "HTP+CPU".** The docs call it a "llama_cpp per-tensor
   HTP+CPU scheduler". The real layer assignment is **GPU+HTP**:

   ```
   load_tensors: layer  0..15 assigned to device GPUOpenCL
   load_tensors: layer 16..32 assigned to device HTP0
   ```

   CPU was not in the split at all. Anyone reasoning about `hybrid` from the
   docs alone will get its behaviour wrong.

## The Hexagon failure -- real, and it is not a silent fallback

Both `--device hybrid` and `--device llama_cpp:HTP0` fail identically.
**The NPU genuinely initializes and genuinely receives the weights first:**

```
ggml-hex: Loading driver C:\Windows\System32\DriverStore\FileRepository\qcnspmcdm8380.inf_arm64_e663a92a933cab52\libcdsprpc.dll
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v73
ggml-hex: HTP0 allocating new session
ggml-hex: HTP0 hwinfo: threads 4, hvx 4, hmx 1, vtcm 8 MB
ggml-hex: HTP0 new session : session-id 0 domain-id 3 uri file:///libggml-htp-v73.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0
ggml-hex: HTP0 op batching: n-bufs 16 n-tensors 7168 n-ops 1024 vmem 3145728000
load_tensors: layer 0..32 assigned to device HTP0        (all 33 layers)
load_tensors:        HTP0 model buffer size =     0.77 MiB
load_tensors: HTP0-REPACK model buffer size =  1950.01 MiB
```

Then, at the first generate call:

```
C:/a/GenieX/GenieX/third-party/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1583:
ggml-hex: dspqueue_read failed: 0x00000072
```

Process exits **127**. Reproduced **3/3 times** (once via `hybrid`, twice via
`llama_cpp:HTP0`) -- deterministic, not flaky.

Worth stating plainly, because it is the opposite of what
`docs/vlm-npu-research.md` feared: **this is not the silent-CPU-fallback bug.**
1950 MiB of weights really were repacked and pushed to the Hexagon, a real v73
session on domain-id 3 really was opened, and the failure is a loud crash in
the FastRPC transport (`dspqueue_read`), not a quiet reroute to CPU. The
honesty of this backend is fine; its reliability on this machine is not.

Also note `llama_prepare_model_devices: using device HTP0 (Hexagon) - 0 MiB
free` -- the backend reports zero free memory for HTP0, which is likely why
`hybrid`'s scheduler split at layer 16 rather than by any capability measure.

## The result that actually matters

Same GGUF, same machine, same prompt/system prompt as the original NPU smoke
test ("What is gravity?", "You are a helpful assistant. Answer in a sentence.").

| Path | Runtime / artifact | Short (~35 tok) | Sustained (512 tok) |
|---|---|---|---|
| **CPU** | GenieX llama_cpp, GGUF Q4_0 | **39.8 tok/s** | **32.1 tok/s** |
| **GPU (Adreno)** | GenieX llama_cpp, GGUF Q4_0 | 28.7 / 28.6 tok/s | 25.8 tok/s |
| **NPU (existing `NpuFastBrain`)** | Genie SDK, w4a16 context binary | -- | **10.6 tok/s** (n=2180) |
| NPU via llama.cpp | GGUF Q4_0 on HTP0 | **crashes** | **crashes** |

Time-to-first-token: **0.1 s (CPU) / 0.2-0.3 s (GPU)** versus **~142 ms** for
the Genie path -- comparable, so the gap is decode throughput, not latency to
first byte. Cold load was **3.0 s** for the GPU GGUF path versus the Genie
path's measured **13.1 s**.

Both GGUF paths also stopped cleanly on `stop: eos`, with no explicit
stop-sequence configuration -- the EOS fragility that forced
`GenieDialog_setStopSequence` + `ABORT` bounding in `NpuFastBrain` simply does
not appear here.

### Caveats -- do not over-read this table

- **Different quantization.** w4a16 (NPU-optimized, weight-shared across 40
  graphs) versus generic Q4_0. Not a controlled comparison of silicon.
- **Power was not measured, and that is the whole point of the NPU.** This
  project is explicitly power-aware. A 3-4x throughput deficit at a large
  fraction of the wattage may still be the right trade for a background fast
  brain. **Nothing here justifies dropping the NPU path until perf-per-watt is
  measured.** That measurement is the obvious next experiment.
- Sustained numbers are one 512-token run each, not averaged.
- CPU at 32-40 tok/s will contend with everything else on the machine in a way
  the NPU does not -- throughput is not the only axis.

## Verdict

- **`hybrid` does not solve gap 3/3b.** The Hexagon backend is
  self-described "experimental" and, on this machine, crashes deterministically
  at inference. Do not build on it. Re-test if a newer GenieX or Hexagon driver
  ships; the specific signature to watch is `dspqueue_read failed: 0x00000072`
  at `ggml-hexagon.cpp:1583`.
- **But the GGUF/llama.cpp runtime itself is very much viable** -- just on CPU
  and GPU rather than NPU. It is faster than the current fast brain, loads 4x
  quicker, handles EOS correctly, needs no `convert_model`, and accepts any
  GGUF from HuggingFace. Every strategic advantage originally hoped for from
  `hybrid` is available on the **GPU** path, which is already independently
  verified by the VLM work.
- The open question is no longer "can we run GGUF on the NPU" but **"is the NPU
  worth it on perf-per-watt"**. Measure that before changing the router.
