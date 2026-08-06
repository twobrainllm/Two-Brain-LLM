# How the local fast brain runs on the NPU

Quick-reference for reviewing the architecture later. For the full build
history, real bugs found and fixed, and every receipt, see
[`superpowers/deploy-local-brain-npu.md`](../superpowers/deploy-local-brain-npu.md)
and `data/npu_model/phi-3.5-mini-instruct/`.

## Short answer: fully in-process, not a local endpoint

`NpuFastBrain` (`src/two_brain_router/routing/brains.py`) is **not** a
vLLM-style local server that the router talks to over HTTP. There is no
server process, no port, no client/endpoint split at all. The model runs
*inside* the router's own Python process, called directly as a foreign
function.

## How it actually works

1. **No server process.** The router's own process loads `Genie.dll`
   (Qualcomm's Genie SDK, shipped inside the `onnxruntime-qnn` pip wheel)
   directly via `ctypes.CDLL(...)`. No subprocess, no socket, no port to
   manage or health-check.
2. **The model session lives inside the router's process.**
   `NpuFastBrain.__init__` calls Genie's C API --
   `GenieDialogConfig_createFromJson` then `GenieDialog_create` -- which
   loads the real quantized weight-sharing context binaries
   (`weight_sharing_model_*.serialized.bin`, ~816 MB) straight onto the
   Hexagon NPU and keeps the dialog handle resident for the brain's
   lifetime (created once in `__init__`, reused across `answer()` calls,
   freed in `close()`).
3. **Inference is a direct C call, not a request.** `answer()` calls
   `GenieDialog_query(dialog_handle, prompt, callback, ...)` -- a
   synchronous C function call across the `ctypes` boundary. Genie invokes
   a Python callback (`_on_response`) directly as tokens stream out. No
   JSON-over-HTTP, no request/response serialization for the inference
   call itself.

## Why this was the deliberate choice, not an accident

From the "Decision" section of `superpowers/deploy-local-brain-npu.md`:

- **The router is latency-budget-aware by design**
  (`policy.local_latency_budget_ms`). An HTTP hop -- even to `localhost` --
  adds nondeterministic latency to the exact number the router uses to
  decide local-vs-cloud. A hosted endpoint would pollute the measurement
  the whole project is built around.
- **The privacy invariant stays simplest in one process.** The masked-vault
  guarantee (`router.py`'s mask-first ordering) is easiest to audit when
  there's no second process that could see raw or intermediate state.
- **`routing/brains.py` already defines the seam for exactly this** --
  `Brain.answer(query, context) -> BrainResponse`. An HTTP client would add
  a process boundary and health-check surface the seam doesn't need.

A hosted local endpoint (Foundry Local, Microsoft's turnkey NPU-optimized
OpenAI-compatible server) was evaluated as an alternative and explicitly
rejected for this task -- narrower/buggier NPU model catalog, and it hands
you a service instead of an owned, inspectable artifact. It's documented as
a fallback (Phase 2c) only if the in-process path had been blocked; it
wasn't needed. See that section of the workflow doc for the full
comparison.

## Where a process boundary *would* make sense instead

Per the same Decision section: if something *other than this router* needed
to share the model (a second app on the same machine), or if the model
runtime only shipped as a server, a hosted endpoint would be the right
call -- and `Brain`'s seam is shaped so that swap wouldn't require changing
anything above `brains.py`. Neither condition applies today.

## Current status

- Real, working, on-NPU inference through this exact path -- confirmed via
  `QnnGraph_execute` execution logs against the `QnnHtp` backend and a real
  QNN profiler capture (`data/npu_model/phi-3.5-mini-instruct/receipts/`).
- Wired into `TwoBrainRouter` for the `pc_3b` tier, gated behind the
  `TWO_BRAIN_NPU_BRAIN=1` env var so the base package stays stdlib-only by
  default (`router.py::_build_fast_brain`).
- Verified end-to-end: `tests/test_npu_brain.py` (skips without the real
  runtime/artifact; passes for real under `.venv-npu`).
