# Workflow: Deploy the Local Fast Brain on This Machine's NPU

Goal: replace `LocalFastBrain` (`src/two_brain_router/routing/brains.py`) —
currently a labeled stub — with a real 3B-class model actually executing on
this machine's Hexagon NPU (Snapdragon X Elite X1E80100, v73, 45 TOPS, QAIRT
2.38.0.250901; see `data/hardware_detect/ai_pc.json`), wired into
`TwoBrainRouter` without changing anything above the `Brain` seam.

This is [next step #3](../docs/WALKTHROUGH.md#next-steps) from the
walkthrough, but that item assumed an artifact from QUAD's `convert_model`.
It doesn't have one and won't get one — [`docs/GAPS.md`](../docs/GAPS.md)
#3/#3b and `data/convert_model/_real_attempts_log.md` already prove **two**
independent, unfixable-from-here defects in that toolchain (hosted server
missing `libpython3.10.so.1.0`; local QAIRT hitting a non-deterministic
uninitialized-memory read in `ReshapeOp::calculateShape`). Per
[`CLAUDE.md`](../CLAUDE.md), do not retry `convert_model` variations.

**The strategy here is to route around QUAD's compiler entirely** and use a
different, currently-working toolchain: Qualcomm AI Hub's cloud compile
service (`qai_hub_models`) + ONNX Runtime GenAI's QNN execution provider.
This is not a workaround-of-a-workaround — it's Qualcomm's and Microsoft's
own maintained, documented path for exactly this hardware, independent of
QUAD-Client's `convert_model` MCP tool and its broken `qairt-converter`
binary.

Follow the receipts rule from `CLAUDE.md` throughout: every new file under
`data/` states real vs. mocked and why. No invented benchmark numbers.

---

## Decision — in-process inference, not a hosted endpoint

**Recommendation: call the model in-process from a new `Brain`
implementation, the same shape as the existing stubs. Do not stand up a
local HTTP server as the primary path.**

Why:
- `routing/brains.py` already defines the seam (`Brain.answer(query, context)
  -> BrainResponse`) exactly for this swap. An HTTP hop adds a process
  boundary, port management, and a health-check surface the seam doesn't
  need — [Next steps #3](../docs/WALKTHROUGH.md#next-steps) says explicitly
  "nothing above `brains.py` should need to change."
- The project's whole premise is a **latency-budget-aware** router
  (`policy.local_latency_budget_ms`). An HTTP round-trip (even to
  `localhost`) adds nondeterministic latency to the exact number the router
  makes decisions on — it pollutes the measurement the project cares about.
- The privacy invariant (masked-vault, step 4 of `router.py::route`) stays
  simplest with everything in one process. A second process is one more
  place to audit for what crosses a boundary.

**When a hosted endpoint would be the right call instead:** if something
*other than this router* needs to share the model (e.g. a second app on this
machine), or if the model runtime only ships as a server (true of some NPU
runtimes). Microsoft's **Foundry Local** is the concrete option here — it is
a turnkey local OpenAI-compatible endpoint with NPU-optimized Phi-3/Llama-3
models built for this exact chip, and it's worth evaluating as a *fallback*
if Phase 2's `qai_hub_models` path proves too brittle. Treated as Phase 2b
below — not required for "done," but a documented fallback, not an
afterthought.

### Why AI Hub over Foundry Local specifically (not just "in-process over hosted")

This is a separate question from the endpoint-vs-in-process decision above —
even if we wanted a hosted endpoint, Foundry Local wouldn't be the pick for
*this* task, for reasons independent of the architecture debate:

- **Foundry Local *is* a server by construction.** It stands up its own
  local OpenAI-compatible service and manages the model lifecycle itself —
  there is no supported "just hand me the raw session" mode. Picking it
  means accepting the hosted-endpoint architecture the Decision above argues
  against, not a variant of the in-process path.
- **Its QNN/NPU catalog is currently narrow and has an open sync bug.**
  Confirmed via [microsoft/Foundry-Local#783](https://github.com/microsoft/foundry-local/issues/783):
  the catalog advertises more NPU-tagged models than are actually
  resolvable locally, and specific pulls (e.g. `qwen2.5-coder-7b-instruct-qnn-npu`)
  fail with "Model was not found in the catalog or local cache." That's the
  same *class* of gap this whole project already catalogs in `docs/GAPS.md`
  — a real, reproducible tool defect, not FUD — and it's a bad foundation to
  build a "definition of done" test suite on.
- **AI Hub hands you the raw artifact; Foundry Local hands you a service.**
  Phase 3's mandatory NPU-verification step needs to inspect actual EP node
  assignment and hold a real context binary as a receipt. AI Hub's export
  produces exactly that (`.bin` + `.onnx` wrapper you own). Foundry Local's
  packaged service doesn't expose this — you'd be trusting its internals
  instead of verifying them, which conflicts with this repo's receipts rule.
- **AI Hub isn't actually required for compute anymore — see the revised
  Phase 2 below.** Qualcomm publishes some models (Phi-3.5-mini-instruct
  included) as *already-compiled* QNN context binaries directly on
  Hugging Face. Where that exists, neither AI Hub's cloud compile nor
  Foundry Local's service is needed — you download a real artifact and run
  it in-process. AI Hub only becomes necessary as a fallback if no
  pre-built artifact exists for the chosen model.

**Bottom line:** Foundry Local is the right call if the goal were "get any
chat endpoint running on this NPU fast," but the goal here is "own a real,
inspectable NPU artifact wired directly into `Brain`." AI Hub (or, better,
a pre-published binary — see Phase 2) serves that goal; Foundry Local's
current catalog reliability and server-only shape work against it. Foundry
Local stays as Phase 2b, useful only if both the pre-built-download and
AI-Hub-export paths turn out to be blocked.

---

## Phase 0 — Scope and environment audit

### What size model this machine can actually run on-NPU

Hardware, from `data/hardware_detect/ai_pc.json`: 31.6 GB RAM, Hexagon NPU
v73 @ 45 TOPS. Qualcomm's own marketing claims "13B+ on-device," and that's
plausible for *loading* at 4-bit (a 13B model at INT4 is roughly 6.5–7 GB of
weights, well inside 31.6 GB RAM) — but this project's `Brain` role isn't
"can it load," it's "fast brain," bound by
`policy.local_latency_budget_ms = 3000`. That budget is the real ceiling on
model size, not RAM or TOPS.

The one *real, published* data point for this exact chip
(`qualcomm/Phi-3.5-mini-instruct`'s HF model card, QNN context binary,
w4a16, QAIRT 2.43, Snapdragon X Elite) reports **10.2 tokens/sec** — i.e.
~98 ms/token — for a 3.8B model. That's slower than this project's current
*mocked* `pc_3b.json` estimate (44.8 ms/token), which is exactly the kind of
gap Phase 4/5 exists to catch and retune against, not evidence the mock was
wrong to be a placeholder. Extrapolating roughly (throughput scales
sub-linearly with params but this is a coarse ballpark, not a receipt):

| Class | Rough interactive fit under a 3s local budget | Verdict |
|---|---|---|
| 1–4B (Phi-3.5-mini, Qwen2.5-1.5B/3B, Llama-3.2-1B/3B) | Yes — real 3.8B number above lands ~20-30 output tokens in budget | **Target range for the fast-brain role** |
| 7–8B (Qwen2.5-7B, Llama-3.1-8B, DeepSeek-R1-7B) | Marginal — likely usable but eats most of the budget on TTFT+first few tokens | Only if 1–4B proves insufficient quality |
| 13B+ | Loads fine, but likely blows the 3s local budget for anything but a one-word answer | Out of scope for "fast brain"; a future "medium brain" tier, not this task |

**Decision: stay in the 1–4B class**, consistent with the project's existing
`pc_3b` tier design intent — the real number above corroborates rather than
contradicts that choice.

### Least-friction model pick within that range

Targeted comparison of the 1–4B candidates that are actually documented
against this chip:

| Model | Gated? | Pre-built QNN binary available? | Runtime match | Friction |
|---|---|---|---|---|
| **Phi-3.5-mini-instruct (3.8B)** | No | **Yes** — `qualcomm/Phi-3.5-mini-instruct` on HF ships a ready QNN context binary (w4a16) for Snapdragon X Elite, with published perf numbers; `onnx-community/Phi-3-mini-instruct-hexagon-npu-assets` has the matching genai wrapper/tokenizer bundle | Same `onnxruntime-genai` runtime Phase 1 already installs | **Lowest** — no compile step needed at all |
| Qwen2.5-1.5B / Qwen3-4B | No | Present in AI Hub's model list; some ship via NexaSDK-flavored repos (`NexaAI/...`) rather than plain onnxruntime-genai | NexaSDK path is a *different* runtime than what Phase 1 targets | Higher — either self-compile via AI Hub, or adopt a second runtime |
| Llama-3.2-1B/3B | **Yes (HF license)** | Available via AI Hub self-compile; no confirmed ready-made binary as convenient as Phi's | onnxruntime-genai, same as Phi path | Higher — licensing step + self-compile |

**Pick: Phi-3.5-mini-instruct**, via the pre-built binary — it's the only
option with zero compile step, no gating, and the exact runtime already
planned. This also matches Phase 0's model choice below; the two were
previously justified only by "not gated," now corroborated by an actual
friction comparison.

- [x] Confirm target model: **Phi-3.5-mini-instruct**, sourced as the
      pre-built QNN context binary from `qualcomm/Phi-3.5-mini-instruct` on
      Hugging Face (see comparison above) — not a self-export via AI Hub,
      unless Phase 2a's download step turns out to be unusable on this
      machine.
- [x] **QAIRT version check — resolved, not a blocker.** This machine has
      **two** QAIRT SDKs installed side by side:
      `C:\Qualcomm\AIStack\QAIRT\2.32.6.250402` and
      `C:\Qualcomm\AIStack\QAIRT\2.38.0.250901` (the one `ai_pc.json`
      recorded) — neither is 2.43, the version the pre-built binary was
      compiled against. **But this doesn't matter**: confirmed via
      `onnxruntime-qnn`'s own docs/PyPI page that the pip package bundles
      its own QNN runtime libraries independent of any system QAIRT SDK
      install — no separate SDK download is required to use it. The
      current PyPI release (`onnxruntime-qnn` 2.4.0, Jul 2026) is built
      against **QAIRT SDK v2.48.40**, which is *newer* than the binary's
      2.43 — QNN's compatibility direction is a newer runtime loading an
      older context binary, which is the normal/supported case (the
      documented failure mode is the reverse: an *older* runtime rejecting
      a *newer* binary, e.g. "Flexible Context Binary requires QAIRT 2.48+
      at runtime"). **Action:** install `onnxruntime-qnn` via pip (not the
      system QAIRT SDK) for the runtime environment; the two SDKs already on
      disk are irrelevant to this path and can be left alone. This is a
      documented-compatible expectation, not a guarantee — Phase 2a/3 still
      needs to load the binary for real and confirm, per this repo's
      receipts rule.
- [ ] Confirm this machine still matches `data/hardware_detect/ai_pc.json`
      (`quad-client detect --json` — same command used originally). If it
      drifted, log a fresh capture, don't hand-edit the file. *(Not run this
      pass — CPU/RAM spot-checked below via PowerShell and match `ai_pc.json`
      exactly; a full `quad-client detect --json` re-run is still worth
      doing before Phase 4, not required to unblock Phase 0.)*
- [x] **Disk headroom — real capture, `Get-PSDrive C`:** 228.58 GB free /
      247.02 GB used (475.6 GB total, matching `ai_pc.json`'s
      `storage_gb`). Comfortably enough for model weights + quantized
      artifacts (a few GB) with no compile toolchain needed on the primary
      path.
- [x] **RAM headroom — real capture:** 33,099,296 KB total (≈31.6 GB,
      matches `ai_pc.json`) / 12,882,272 KB free (≈12.3 GB) at time of
      check, with normal desktop load already running. Enough for a 3.8B
      w4a16 model (~2 GB weights) plus KV cache and runtime overhead: worth
      re-checking free RAM immediately before Phase 3's real inference run
      rather than assuming this snapshot holds.
- [x] **WSL — already installed, not a gap.** `wsl --list --verbose` shows
      `Ubuntu-22.04` present (stopped) alongside Docker Desktop's WSL
      distros. Confirms **Phase 2b's fallback (if ever needed) has no WSL
      install step to do** — only `libc++-dev` inside the existing Ubuntu
      distro, if Phase 2b is actually triggered. Not relevant to Phase 2a.
- [x] Decide and record: Phase 2a's pre-built-binary download as primary,
      Phase 2b (AI Hub self-compile) as its own fallback, Foundry Local as
      the last resort (Phase 2c) — do not start more than one in parallel.
      No findings from this pass change that order — if anything, the WSL
      and QAIRT-version checks both *remove* friction from 2a further ahead
      of 2b/2c.

## Phase 1 — Prerequisites — DONE

**Always needed, regardless of which of Phase 2a/2b/2c is used:**

- [x] Created a dedicated native-ARM64 venv, **`.venv-npu`** (separate from
      the router's own stdlib-only `.venv` per `run.ps1` — this repo's base
      package intentionally has zero deps, so the NPU runtime stack lives in
      its own environment, installed via a new `npu` extra added to
      `pyproject.toml`). Built with `py -3.12-arm64 -m venv .venv-npu`, then
      `pip install -e ".[npu]"`. Both `onnxruntime-genai` (0.15.1) and
      `onnxruntime-qnn` (2.4.0) have real `win_arm64`/`cp312` wheels and
      installed cleanly — confirmed via PyPI's JSON API before installing,
      not assumed. Only `numpy`, `onnxruntime` (plain, 1.28.0, pulled in as
      `onnxruntime-genai`'s own dependency), `protobuf`, `flatbuffers`,
      `coloredlogs`, `sympy` came along as transitive deps — nothing
      unexpected.
- [x] Confirmed HuggingFace access needs nothing extra: both
      `qualcomm/Phi-3.5-mini-instruct` and
      `onnx-community/Phi-3-mini-instruct-hexagon-npu-assets` are plain,
      ungated public repos — no token, no license click-through (verified by
      actually downloading from both in Phase 2a below, not just reading the
      model card).
- [x] **Addendum from Phase 6**: `.venv-npu` didn't have test tooling
      (only the `npu` extra was installed here — this workflow's own real
      inference runs used `genie-t2t-run.exe`/raw Python, not `pytest`).
      Running Phase 6's real test suite needed `pip install -e
      ".[npu,dev]"` (the existing `dev` extra already had `pytest>=7.0` —
      no new dependency, just hadn't been installed into this venv yet).

### Real finding: how `onnxruntime-qnn` actually attaches to `onnxruntime`

This matters enough to log as its own receipt, since it contradicts a
plausible-sounding assumption from the original plan ("swap in `onnxruntime`
for a QNN-enabled build").

**`onnxruntime-qnn` is not a replacement build of `onnxruntime`.** Installing
both leaves the plain `onnxruntime` package (1.28.0, CPU/Azure EPs only)
intact and untouched — `onnxruntime_qnn` installs into its own separate
top-level package (`onnxruntime_qnn/`, confirmed via its wheel's `RECORD`
— no shared filenames, no clobbering). It contains **no Python bindings of
its own** — just the native QNN backend DLLs (`onnxruntime_providers_qnn.dll`,
`QnnHtp.dll` for the Hexagon HTP/NPU backend, `QnnCpu.dll`, `QnnGpu.dll`,
`libQnnHtpV73Skel.so` — v73, matching this machine's exact NPU — plus
`Genie.dll`) and a thin `__init__.py` exposing `get_library_path()` /
`get_qnn_htp_path()`.

Real, confirmed activation sequence (this is what `NpuFastBrain` in Phase 4
must do at session-creation time):

```python
import onnxruntime as ort
import onnxruntime_qnn as oq

ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
# ort.get_available_providers() now includes "QNNExecutionProvider"
```

Verified live: before registration, `ort.get_available_providers()` returned
only `['AzureExecutionProvider', 'CPUExecutionProvider']`; after calling
`register_execution_provider_library`, it returned
`['AzureExecutionProvider', 'CPUExecutionProvider', 'QNNExecutionProvider']`.
This is ONNX Runtime's newer **plugin execution provider** mechanism
(`register_execution_provider_library` / `unregister_execution_provider_library`
on the `onnxruntime` module) — the EP is loaded from an external DLL at
runtime rather than requiring a special onnxruntime build, which is why the
plain `onnxruntime>=1.20.1` dependency from `onnxruntime-genai` and
`onnxruntime-qnn`'s native bundle coexist without conflict.

**Still to confirm in Phase 3** (this only proves the EP *registers* — not
that it successfully creates a session or executes ops on HTP): actually
creating an `InferenceSession` with `providers=["QNNExecutionProvider"]` and
a `backend_path` provider option pointing at `oq.get_qnn_htp_path()`, once a
real model is in hand.

**Superseded by a Phase 2a finding — this EP-registration mechanism is real
and correct, but `NpuFastBrain` does not end up using it.** The pre-built
artifact Phase 2a actually downloads (Qualcomm's "genie" release flavor —
see its `_real_download_log.md`) is a Genie SDK bundle, not an
onnxruntime-genai one; the `onnx-community` wrapper repo that would have
supplied the missing `onnxruntime-genai`/QNN-EP-compatible `*_qnn_ctx.onnx`
files never actually shipped them. `NpuFastBrain` instead calls Genie's own
C API (`Genie.dll`, confirmed present in this same `onnxruntime-qnn` wheel)
via `ctypes` — still in-process, still no separate `onnxruntime` session,
just a different native entry point than originally planned. This
`register_execution_provider_library` mechanism stays documented here
because it's real and may matter for a future model choice that *does*
ship onnxruntime-genai-compatible weights, but `NpuFastBrain`'s Phase 4
implementation doesn't call it.

**Only needed if Phase 2a's pre-built download fails and Phase 2b
(self-compile) is triggered:**

- [ ] Create a Qualcomm AI Hub account and API token
      (`app.aihub.qualcomm.com` — getting-started guide linked from
      [onnxruntime.ai's build-for-Snapdragon doc](https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html)).
      Run `qai-hub configure --api_token <token>`.
- [ ] **x64 Python, separate from the ARM64 interpreter.** Per
      `onnxruntime-qnn`'s own docs, ONNX quantization tooling is only
      supported on x86_64 today (ARM64 has `onnx` package install issues).
      This mirrors the exact `platform.processor()` arch-detection bug
      already logged in `docs/GAPS.md` #3b/attempt 5 — expect it to resurface
      and reuse the shim (`qairt_converter_shim.py` pattern) if it does.
- [ ] Install WSL (Ubuntu) if Phase 0 confirmed it's missing; `sudo apt
      install libc++-dev` inside it (required only for Phase 2b's QNN
      context-binary extraction sub-step).
- [ ] `pip install -U qai_hub_models[phi-3-5-mini-instruct]` (adjust extra
      to final model choice) in the x64 environment.

## Phase 2 — Acquire a real, NPU-targeted model artifact (primary path)

Bypasses QUAD's `convert_model` completely. Two tiers, try in order — both
avoid QUAD's toolchain, but the first needs no compile step at all.

### Windows-native scope, confirmed

**Everything in Phase 2a runs natively on Windows.** No WSL. The only place
WSL enters this workflow at all is one sub-step of Phase 2b's fallback
(`qnn-context-binary-utility`'s JSON extraction, which Qualcomm ships
Linux-only). If Phase 2a succeeds — which is the expected/primary outcome —
this entire workflow never touches WSL. The one genuinely non-Windows-native
requirement anywhere in this plan is Phase 1's separate **x64 Python**
process for quantization tooling (still Windows, just a different
interpreter architecture than the native ARM64 one) — and that's also only
needed if Phase 2b's self-compile is triggered, since Phase 2a downloads an
already-quantized artifact.

### Phase 2a — Download the pre-built binary (try first) — DONE

- [x] Downloaded `qualcomm/Phi-3.5-mini-instruct`'s QNN context binary
      (w4a16, Snapdragon X Elite target, "genie" release flavor) from
      Hugging Face / Qualcomm's public S3. Real, logged in
      `data/npu_model/phi-3.5-mini-instruct/_real_download_log.md`.
- [x] Downloaded the `onnx-community/Phi-3-mini-instruct-hexagon-npu-assets`
      bundle — same log. **Real finding**: this bundle only ships the
      auxiliary graphs + tokenizer/config, not the weight-bearing
      `*_qnn_ctx.onnx` files its own `genai_config.json` references — it's
      not usable as an onnxruntime-genai wrapper for the "genie" binary
      Qualcomm currently publishes. See the download log's "Real finding"
      section; this changed the runtime `NpuFastBrain` calls (Genie's own
      C API, not onnxruntime-genai/QNN-EP) without changing the
      in-process/no-server architecture decision above.
- [x] Landed under `data/npu_model/phi-3.5-mini-instruct/`, receipted:
      exact URLs, SHA256, real file sizes, and the QAIRT version mismatch
      (binary built against 2.43.1.260218; this machine's system SDK is
      2.38.0.250901) — all in `_real_download_log.md`.
- [x] Loading/running: **failed** against the system QAIRT 2.38.0 SDK for
      exactly the version-mismatch reason flagged above (real error:
      blob version 3.3.4 vs. max-supported 3.2.3) — logged precisely in
      `_real_inference_smoke_log.md` rather than silently worked around.
      **Did not move to Phase 2b**, because a second real attempt using
      `onnxruntime-qnn`'s pip-bundled runtime (newer HTP skel, QAIRT
      2.48.40) succeeded — same log. Phase 2a is the artifact source that
      shipped; Phase 2b/2c were not needed.

### Phase 2b — Self-compile via AI Hub (fallback if 2a is blocked)

- [ ] Export + generate QNN context binaries:
      `python -m qai_hub_models.models.<model>.export --device "Snapdragon X
      Elite CRD" --skip-inferencing --skip-profiling --output-dir .`
      This uploads to Qualcomm's cloud compiler and can take a while —
      **this is the one step in this whole workflow that isn't purely
      local**; document that honestly, it doesn't compromise the "runs
      on-device" claim since only the *compile* is cloud-assisted, not
      inference.
- [ ] Extract QNN graph JSON per `.bin` (**the one WSL-only step in this
      entire plan**):
      `qnn-context-binary-utility --context_binary=X.bin --json_file=X.json`
- [ ] Generate the ONNX Runtime GenAI wrapper models
      (`gen_qnn_ctx_onnx_model.py -b X.bin -q X.json --quantized_IO
      --disable_embed_mode`) for each binary — this step itself has a
      documented Windows PowerShell form, so it's back to native Windows
      immediately after the WSL extraction sub-step.
- [ ] Pull the matching tokenizer/config assets, same list as Phase 2a.
- [ ] Land and log the artifact set exactly as in Phase 2a, noting this was
      the self-compiled path and why 2a didn't work.

**Roadblock if Phase 2b also fails:** fall back to Phase 2c (Foundry Local)
rather than spending more than one session retrying AI Hub — if AI Hub's
cloud compiler itself turns out broken for this model/device pair, that's a
new, separately-loggable gap, not a reason to loop.

## Phase 2c — Last resort: Foundry Local (only if both 2a and 2b stall)

- [ ] Install Foundry Local, its dedicated Hexagon NPU runtime driver
      (separate from the standard Windows NPU driver — Qualcomm Software
      Center, free developer account), pull an NPU-optimized model
      (Phi-3-mini is the documented option).
- [ ] `foundry service set --port 5273` (or similar) for a stable local
      endpoint; note this makes the hosted-endpoint path real if chosen —
      in that case, Phase 4's `Brain` implementation becomes a thin HTTP
      client instead of an in-process caller, and Phase 5's "reachable
      cleanly" tests become literal HTTP health checks (see Phase 6).
- [ ] If this path is taken, update the Decision section above to explain
      why Phase 2 was abandoned — don't leave both half-done.

## Phase 3 — Prove NPU execution before writing any router code — DONE

Do this *before* Phase 4. A model that "runs" but silently fell back to CPU
is exactly the trap `convert_model` attempt 3 caught for us once already
(ConvInteger loads fine, runs on CPU, would have invalidated every per-tier
number in this project) — don't reintroduce it here by skipping verification.

Full trace, real numbers, and the one still-open item in
`data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md`.

- [x] Ran the model standalone — not via `onnxruntime-genai` (see Phase 1's
      addendum on why not) but via `genie-t2t-run.exe` with `--log info`,
      Genie's own equivalent standalone entry point for this artifact.
- [x] Confirmed real HTP execution: `QnnGraph_execute started/done` logged
      against the `QnnHtp`/HTP backend for every generation step, 40 graphs
      loaded, with per-call durations consistent with real decode compute
      (10-40ms/split) — not a no-op or CPU path (Genie's backend is fixed
      to `QnnHtp` here, so there is no silent-CPU-fallback failure mode to
      separately rule out, unlike an ORT-EP scenario).
- [x] Cross-check via Task Manager/perf counter: **checked, not available**
      via `Get-Counter` (no NPU-labeled `GPU Engine` instance on this
      system) — **closed instead via `genie-t2t-run.exe --profile`**, a
      real bounded (`--action ABORT --sleep 6000`) capture with structured
      QNN/Genie profiler output (init time, TTFT, prompt/token rates) plus
      a full `--log info` trace confirming 704 real `QnnGraph_execute`
      pairs against `QnnHtp` and zero CPU-fallback markers. See
      `data/npu_model/phi-3.5-mini-instruct/receipts/` and the smoke log's
      point 2 under "On-device confirmation."
- [x] Recorded real per-token latency and TTFT: **~142ms TTFT**
      (post-prefill), **~94.5ms/token** (~10.6 tok/s) — corroborates
      Qualcomm's published 98ms/token figure for this exact
      model/precision/chip almost exactly. First real numbers for the
      `pc_3b` tier, feeding Phase 4's data update.

## Phase 4 — Wire it into the router — DONE

- [x] Added `NpuFastBrain` to `routing/brains.py` implementing the existing
      `Brain` protocol (`answer(masked_query, context) -> BrainResponse`),
      calling Genie's C API directly via `ctypes` (see Phase 1's addendum
      for why Genie, not onnxruntime-genai/QNN-EP). `router.py`,
      `policy.py`, and `signals/` are unchanged — the seam held. Two real
      bugs were found and fixed while getting this to actually run (not
      just import cleanly), logged in
      `data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md`'s
      "Attempt 3": (1) `add_dll_directory()` alone wasn't enough for
      Genie's *internal* `QnnSystem.dll` lookup at dialog-creation time —
      needed a `PATH` prepend too; (2) `GenieDialog_setStopSequence` needs
      a JSON *object* (`{"stop-sequence": [...]}`), not a bare array —
      confirmed against this SDK's own `Dialog.cpp`.
- [x] Registered it in `TwoBrainRouter.__init__` in place of the stubbed
      `LocalFastBrain`, for the `pc_3b` tier only, gated behind the
      `TWO_BRAIN_NPU_BRAIN=1` env var (see `router.py::_build_fast_brain`)
      so the base package's test suite/CLI stay stdlib-only and fast by
      default; mobile/cloud stubs are untouched. Verified for real: the CLI
      demo run with `TWO_BRAIN_NPU_BRAIN=1` under `.venv-npu` returns a
      genuinely generated answer for the easy/local query
      ("Tokyo is in the Japan Standard Time (JST) time zone, which is
      GMT+9.") while the hard/cloud and PII/cloud queries are unchanged.
- [x] Updated `data/profile_workload/pc_3b.json` and
      `data/orchestrate_workload/pc_3b.json` with the real numbers from
      Phase 3, `_mock: false`, diffs against the old mocked values recorded
      in each directory's `_real_call_log.md` (per-token latency: real
      hardware is ~2.1x slower than the mock guessed — 94.5ms vs. 44.8ms).
      Thresholds (`escalate_threshold=0.55`, `local_latency_budget_ms=3000`)
      were **not** retuned in this pass — see Phase 5, both still hold for
      the three demo queries with the real numbers in place.

## Phase 5 — Regression and threshold retune — DONE

- [x] Ran the full existing suite (`.venv\Scripts\python.exe -m pytest
      tests\ -q`) — green, unmodified, 9/9 passing including
      `test_escalated_pii_never_reaches_cloud_unmasked`. The real
      per-token numbers did **not** flip any of the three demo queries'
      routing outcome (easy still stays local, both hard/PII still
      escalate) — no threshold retune was actually needed this pass; noted
      here rather than retuned preemptively, per `CLAUDE.md`'s framing that
      a routing-outcome shift (had one happened) would be review-worthy,
      not something to paper over with a threshold change made without a
      shifted outcome to justify it.
- [x] Re-ran the CLI demo (`two_brain_router` module, base `.venv`, no NPU
      env var — this is what the README's transcript documents). Output
      byte-identical to the existing README transcript for query 3 (the
      one shown): `est_latency_ms=1159`, `est_cost_usd=0.11520`, same
      `local_latency_est=6190ms`. Unaffected by the `pc_3b` data change
      because that query escalates to cloud, which reads from
      `cloud_large.json`, not `pc_3b.json`. **No README update needed** —
      confirmed, not assumed, by diffing real output against the
      checked-in transcript.

## Phase 6 — Definition of done: mandatory verification tests — DONE

All of the following must pass and be reproducible from a clean shell
before this is considered "deployed." These are in addition to, not instead
of, the existing 9 tests in `tests/`. Implemented in
`tests/test_npu_brain.py` — skips cleanly (module-level `skipif`) when the
NPU runtime stack/artifact isn't present (e.g. the base `.venv`), runs for
real under `.venv-npu`.

- [x] **`test_npu_ep_assignment`** — Genie's backend has no ORT-style
      dynamic EP selection to inspect (fixed to `QnnHtp` per
      `genie_config.json`, no CPU-fallback path exists in this runtime at
      all — see `brains.py`'s module docstring), so this shells out to the
      real `genie-t2t-run.exe --log info` CLI (the same proven recipe from
      Phase 3's log) and asserts on its real output: `QnnGraph_execute
      started`/`done` present, `QnnHtp` present, zero `ExecutionProvider`
      or CPU-fallback mentions. A ctypes `GenieLog_Callback_t` capture was
      considered and rejected — it's a `va_list`-based printf-style
      callback ctypes can't marshal reliably, and a subtly-wrong capture is
      worse than the CLI-subprocess approach actually used, per this repo's
      receipts rule.
- [x] **`test_real_inference_smoke`** — fixed prompt ("What time zone is
      Tokyo in?") against the real model via `NpuFastBrain.answer()`,
      asserts non-empty text, not the `[local:`/`[cloud:` stub prefixes,
      under a 60s ceiling. Passes for real.
- [x] **`test_npu_utilization_observed`** — closed in Phase 3 above via a
      real `genie-t2t-run.exe --profile` capture
      (`data/npu_model/phi-3.5-mini-instruct/receipts/`), not folded into
      the pytest suite itself since it's a one-time hardware receipt, not
      a per-run assertion.
- [x] **`test_router_end_to_end_with_real_brain`** — all three demo queries
      through `TwoBrainRouter` with `NpuFastBrain` wired in
      (`TWO_BRAIN_NPU_BRAIN=1`); confirms the easy query returns a real
      generated answer and masking/escalation is unchanged for the other
      two. Passes for real. Surfaced one more real finding while landing
      this: two concurrent `NpuFastBrain`/Genie sessions collide on this
      hardware (`Could not create context from binary ... err 1002`) — the
      test fixture is function-scoped, not module-scoped, to avoid it; see
      the smoke log's "Attempt 4."
- [x] **`test_brain_reachable_cleanly`** — in-process (2a was used, no
      hosted endpoint): a clean subprocess imports `NpuFastBrain`, loads
      the session, answers one query, asserts non-empty output and cold
      load under a 60s ceiling (real measured ~13.1s). Passes for real.
- [x] **Full regression** — `pytest tests/ -q` green under both venvs: base
      `.venv` shows 9 passed + 4 skipped (NPU tests skip cleanly, no
      runtime stack there); `.venv-npu` shows all 13 passed for real
      (`pytest` added to that venv via the existing `dev` extra —
      `pip install -e ".[npu,dev]"` — since Phase 1 only installed the
      `npu` extra and hadn't needed test tooling before now).
- [x] **Receipts audit** — `data/profile_workload/pc_3b.json` and
      `data/orchestrate_workload/pc_3b.json` are `_mock: false` with
      diffs logged in their `_real_call_log.md`s;
      `data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md`
      documents all four real attempts (CLI load failure/success, the two
      `NpuFastBrain` bugs, the profiling capture, the concurrent-session
      finding) plus the `receipts/` profiler output. `mobile_1b.json` and
      `cloud_large.json` remain honestly mocked, out of scope per
      Non-goals.

---

## Roadblocks — known going in

| # | Roadblock | Why it matters | Mitigation in this plan |
|---|---|---|---|
| R1 | QUAD's `convert_model` is provably broken on both execution paths (gaps 3/3b) | Can't produce an artifact through the "intended" tool | Bypass entirely via `qai_hub_models` (Phase 2) — different toolchain, not a retry |
| R2 | ONNX quantization tooling only supported on x86_64 today | This machine is ARM64; Phase 2's compile step needs a separate x64 Python | Reuse the arch-detection lesson from `_real_attempts_log.md` attempt 5; isolate quantization to an x64 venv |
| R3 | QNN context-binary JSON extraction (Phase 2b only) is Linux-only | Windows-only assumption breaks | **Resolved/de-risked**: WSL (Ubuntu-22.04) is already installed on this machine (`wsl --list --verbose` confirmed) — only `libc++-dev` would need adding inside it, and only if Phase 2b actually triggers. Phase 2a (primary path) never touches this at all. |
| R4 | AI Hub compile step is cloud-assisted, can take hours, needs an account/token | Not fully "local"; external dependency and turnaround time | Document honestly (only compile is cloud, inference is on-device); budget wall-clock; Foundry Local as fallback if it's unworkable |
| R5 | Model artifacts from AI Hub target a specific model (Phi-3.5-mini / Llama-3.2-3B), neither is the exact model this repo's mocked `pc_3b` data was shaped around | Real profile numbers will differ from the mocked ones | Phase 4 explicitly diffs before overwriting; Phase 5 explicitly expects threshold retuning, not a fixed target |
| R6 | Silent CPU fallback is a proven failure mode on this exact hardware/toolchain family (see `convert_model` attempt 3) | A "working" demo could quietly not be using the NPU at all, invalidating the whole point | Phase 3 verifies NPU execution *before* any router code is written, with two independent signals (EP log + observed utilization) |
| R7 | Llama-3.2-3B (if chosen over Phi-3.5-mini) is HF-gated | Adds a license-acceptance step with an external dependency | Default to Phi-3.5-mini in Phase 0 specifically to avoid this |
| R8 | This machine's actual free disk/RAM headroom for SDK + WSL + weights + artifacts is unverified | Could stall mid-workflow | **Resolved**: real capture — 228.58 GB disk free, 31.6 GB RAM total / ~12.3 GB free at check time. Comfortably sufficient; re-check free RAM immediately before Phase 3's real run since 12.3 GB was a live-desktop snapshot, not a clean baseline. |
| R9 | The pre-built `qualcomm/Phi-3.5-mini-instruct` binary is compiled against QAIRT 2.43; this machine has two *system* QAIRT SDKs installed (2.32.6.250402, 2.38.0.250901), neither matching | Context binaries are version-sensitive — a version gap could mean a load failure or, worse, a silent bad result | **De-risked, not fully closed**: `onnxruntime-qnn`'s pip package bundles its own QNN runtime independent of the system SDK, and the current release (2.4.0) bundles QAIRT 2.48.40 — newer than 2.43, which is the supported/normal compatibility direction (older binary on newer runtime). The system SDKs on disk are irrelevant to this path. Still needs a real load-and-run confirmation in Phase 2a/3 before calling it closed — documented compatibility isn't a tested guarantee. |
| R10 | Foundry Local's own NPU/QNN catalog has an open, documented sync bug (advertised models not resolvable locally) | Picking Foundry Local as primary would import someone else's currently-open gap into this project | AI Hub / pre-built-binary path preferred; Foundry Local demoted to last-resort Phase 2c with this caveat on record |

---

## Non-goals for this pass

- Mobile tier NPU deployment (Snapdragon 8 Elite phone) — separate, blocked
  by the same gap-3 family plus gap 5b; out of scope here.
- Cloud AI 100 real plumbing — unrelated tier, tracked separately (next
  steps #7).
- Replacing the difficulty heuristic (next steps #4) — reasonable follow-up
  once `NpuFastBrain` exists and can emit real logprobs, but it's a distinct
  piece of work and shouldn't be bundled into this deployment task.
