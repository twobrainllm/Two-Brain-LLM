# phi-3.5-mini-instruct on-NPU inference -- real verification log

Per `superpowers/deploy-local-brain-npu.md` Phase 3: prove real NPU execution
*before* writing any router code. `_mock: false` throughout -- every number
below came from an actual run on this machine, logs kept as raw evidence.

## Attempt 1 -- system QAIRT 2.38.0.250901, FAILED with a real, logged error

Ran `genie-t2t-run.exe -c genie_config.json --prompt_file sample_prompt.txt
--log info` from
`data/npu_model/phi-3.5-mini-instruct/raw/phi_3_5_mini_instruct-genie-w4a16-qualcomm_snapdragon_x_elite/`,
with `PATH` pointed at this machine's system QAIRT SDK
(`C:\Qualcomm\AIStack\QAIRT\2.38.0.250901\lib\aarch64-windows-msvc`, the
same SDK `ai_pc.json` recorded).

Real error, verbatim from the log:

```
Genie:   7266.6ms [ ERROR ]  <E> Can't read future blob. Newest blob version supported: 3.2.3. Current blob version: 3.3.4.
Genie:   7266.8ms [ ERROR ]  <E> Unable to determine Context Blob format
Genie:   7267.1ms [ ERROR ]  <E> Skel failed to process context binary.
Genie:   8094.3ms [ ERROR ] qnn-api initialization failed!
Failure to initialize model.
Failed to create the dialog.
```

This is the exact failure mode `superpowers/deploy-local-brain-npu.md`'s
R9 named as a *possible* risk but assessed as unlikely ("older binary on
newer runtime" was expected to be safe) -- it happened, but from the
opposite runtime than R9 checked. R9 verified `onnxruntime-qnn`'s pip
package (bundles QAIRT 2.48.40). This failure is from the **system** QAIRT
SDK's own bundled HTP skel (`libQnnHtpV73Skel.so` under
`C:\Qualcomm\AIStack\QAIRT\2.38.0.250901\lib\aarch64-windows-msvc\`), which
R9 explicitly said was irrelevant to this path -- it turned out to still be
on `PATH` and to be what actually got loaded, since `genie-t2t-run.exe`
itself only ships in the system QAIRT SDK, not in the `onnxruntime-qnn`
pip wheel.

**Root cause:** the binary's context-blob format version (3.3.4, matching
QAIRT 2.43.1.260218 it was compiled against) is newer than what QAIRT
2.38.0.250901's HTP skel can parse (max 3.2.3). Confirms the mismatch
`superpowers/deploy-local-brain-npu.md` Phase 0 flagged as needing a real
load-and-run check, not a documented-compatible assumption.

## Attempt 2 -- `onnxruntime-qnn` pip wheel's bundled DLLs, SUCCEEDED

Same command, same artifact, `PATH` changed to put
`.venv-npu\Lib\site-packages\onnxruntime_qnn\` (its `Genie.dll`,
`libQnnHtpV73Skel.so`, `QnnHtp.dll`, etc. -- QAIRT 2.48.40 per the PyPI
metadata Phase 1 already checked) ahead of the system QAIRT SDK's `bin`
dir (kept on `PATH` only for `genie-t2t-run.exe` itself, which
`onnxruntime-qnn` does not ship).

Real log confirms this resolved the blob-version error and proceeded to
load and execute on-device:

```
Genie:   3836.3ms [  INFO ] qnn-api initialized with 40 graph(s)
...
Genie:   3892.7ms [  INFO ]  <I> QnnBackend_create done successfully. backend = 0xce956438
Genie:   4312.7ms [  INFO ]  <I> QnnDevice_create done. device = 0x1. status 0x0
```

40 QNN graphs loaded (10 LUT, 10 DECODER split x2 arities, 10 LMHEAD --
matches the 4-way weight-sharing split x 10 context-length/autoregressive
buckets `genie_config.json`'s `ctx-bins` list implies). No blob-version
error this run.

### Real generation trace -- prompt from `sample_prompt.txt`

Prompt (verbatim, Qualcomm's own sample):
```
<|system|>
You are a helpful assistant. Answer in a sentence.<|end|>
<|user|>
What is gravity?<|end|>
<|assistant|>
```

Real, on-device generated text (first sentence, before the model ran on
past its first answer -- see "Known issue" below):

> Gravity is the natural force by which all things with mass attract each
> other, commonly experienced as the pull of Earth's mass on objects,
> giving them weight.

Correct and coherent -- not a repeated token, not garbage, not the mocked
stub string. This is real evidence of a working forward pass, not just a
loaded session.

### Real timing (from the log's own timestamps, relative to process start)

| Stage | Real timestamp | Derived metric |
|---|---|---|
| `GenieDialog_query` entered | 13,035.0 ms | -- |
| First response token emitted | 13,177.2 ms | **TTFT (post-prompt-processing) ~142 ms** |
| `GenieDialog_query` returned | 219,008.7 ms | total generation wall time 205,973.7 ms |

Prompt processing itself (the `ar128_cl512_*_of_4` graphs, i.e. the
128-token-batch prefill path) ran in ~140 ms for this short prompt.

Token count: the response text file (log with all `Genie:`-prefixed debug
lines stripped) contains 2,180 emitted token fragments between `[BEGIN]`
and `[END]`.

**Per-token rate: 205,973.7 ms / 2,180 tokens = ~94.5 ms/token (~10.6
tok/s).** This lands almost exactly on the one real, published data point
`superpowers/deploy-local-brain-npu.md` Phase 0 cited for this exact
chip/model/precision (Qualcomm's own model card: 10.2 tok/s, ~98 ms/token)
-- strong corroboration that this run is real hardware execution, not an
artifact of measurement error.

### On-device confirmation (two independent signals, per Phase 3 / R6)

1. **EP/graph-assignment log**: every generation step logs
   `QnnGraph_execute started/done` against the HTP backend
   (`QnnHtp.dll`/`QnnHtpV73Stub.dll`/`libQnnHtpV73Skel.so`) with per-call
   durations (10-40 ms per split, matching decode-time compute, not a
   memcpy or no-op). No CPU-execution-provider fallback path exists in this
   runtime at all (Genie's backend is fixed to `QnnHtp` per
   `genie_config.json`'s `engine.backend.type`) -- so unlike an ORT-EP
   scenario there is no silent-fallback failure mode to separately rule
   out here; the graph-execute timings are the direct evidence NPU compute
   happened.
2. **NPU utilization counter -- checked, not available on this system**:
   tried `Get-Counter -Counter "\GPU Engine(*)\Utilization Percentage"`
   during/after a run; this machine's exposed `GPU Engine` perf-counter
   instances only cover `engtype_3d`/`videoprocessing`/`compute`/etc. for
   GPU adapters, no NPU-labeled instance -- Windows does not expose the
   Hexagon NPU through the classic `GPU Engine` perfmon counter set on
   this build.

   **Closed via the other documented option**: a real
   `genie-t2t-run.exe --profile <file>` capture (bounded to ~6s of
   generation with `--action ABORT --sleep 6000` so the receipt doesn't
   need to wait out a multi-minute run). Real, structured QNN/Genie
   profiler output, not a screenshot substitute:
   `receipts/profile_output_abort6s.json` records real
   `GenieDialog_create` init time (13,773,909 us -- matches the ~13.1s
   cold load measured independently through `NpuFastBrain` below),
   `time-to-first-token` (111,044 us), `prompt-processing-rate`
   (189.1 toks/sec), and `token-generation-rate` (14.7 toks/sec over 88
   tokens in the bounded window -- higher than the fuller n=2180 sample's
   10.6 toks/sec above; real run-to-run variance from a short, ABORT-cut
   sample, not a correction to that measurement). Full stdout
   (`receipts/profile_run_stdout_abort6s.log`, `--log info`) independently
   confirms 704 `QnnGraph_execute started/done` pairs against the `QnnHtp`
   backend and zero CPU-fallback/ExecutionProvider mentions anywhere in
   the log -- second, independent confirmation of on-HTP execution beyond
   the graph-log evidence in point 1. This closes the item Phase 6's
   `test_npu_utilization_observed` was carried forward to.

### Known issue found, relevant to `NpuFastBrain`'s design (Phase 4)

The response did not stop after the first sentence despite
`sample_prompt.txt`'s system instruction ("Answer in a sentence.") --
generation continued past `<|end|>` into fabricated follow-up turns until
Genie's own internal length limit was reached (visible as the context-length
bucket climbing 512 -> 1024 -> 2048 -> 3072 across the run). `genie_config.json`
sets `eos-token: [32007, 32001, 32000]` but the CLI's default invocation
does not appear to stop on it reliably. **`NpuFastBrain` must call
`GenieDialog_setStopSequence` explicitly (e.g. `<|end|>`, `<|user|>`,
`<|system|>`) and additionally bound output length via
`GenieDialog_signal(ABORT)` once a token budget is hit** -- relying on the
model's own EOS behavior alone is not safe, per this real run.

## Environment for both attempts

- `PROCESSOR_ARCHITECTURE`: real ARM64 (`systeminfo` reports
  "System Type: ARM64-based PC") -- `genie-t2t-run.exe`'s
  `aarch64-windows-msvc` build runs natively, no emulation.
- Both attempts loaded DLL dependencies via `PATH` (Bash/POSIX shell); the
  Python `NpuFastBrain` implementation instead uses
  `os.add_dll_directory()` before `ctypes.CDLL(...)`, so it does not
  depend on process-wide `PATH` mutation -- see `routing/brains.py`.

## Attempt 3 -- `NpuFastBrain` (Phase 4's Python/ctypes path), two real bugs found and fixed

First real run of `NpuFastBrain` itself (not the raw `genie-t2t-run.exe`
CLI) surfaced two bugs the CLI attempts above couldn't have caught, since
they don't go through this code path. Both are now fixed in
`routing/brains.py`; logged here per the receipts rule since they're real,
observed failures, not hypothetical ones.

**Bug 1 -- `add_dll_directory()` alone was not enough.** First run failed
inside `GenieDialog_create` with a real error:

```
[ERROR] "Unable to find a valid system interface."
Resource manager not able to create QnnSystemInterface
GenieDialog_create failed with Genie_Status_t=-1
```

Unlike `Genie.dll` itself (which loaded fine via `add_dll_directory`),
Genie's *internal* `QnnSystem.dll` lookup at dialog-creation time did not
resolve through the process's `AddDllDirectory`-registered path alone.
Prepending the same directory onto `PATH` (matching Attempt 2's proven
recipe above, which used `PATH`, not `add_dll_directory`) fixed it. `_load_genie`
now does both: `os.add_dll_directory()` for `ctypes.CDLL(...)` itself, and
a `PATH` prepend for Genie's own internal dependent-DLL resolution.

**Bug 2 -- `GenieDialog_setStopSequence` needs a JSON object, not a bare
array.** Second run got past dialog creation and model load (816,554,496
bytes allocated across 7 buffers -- real weight load) but failed at
`GenieDialog_setStopSequence` with `Genie_Status_t=-8` and the real error
`"Top level config is not an object"`. Confirmed against this SDK's own
`examples/Genie/Genie/src/Dialog.cpp` (`item.key() == "stop-sequence"`,
`qualla/dialog.cpp`'s `Config::optional<std::vector<std::string>>(...,
"stop-sequence", {})`): the expected payload is `{"stop-sequence": [...]}`,
not a bare `[...]`. Fixed in `_create_dialog`. This failure also left a
`GenieDialog_create`d-but-not-yet-configured native session torn down by
Python's GC/interpreter-exit path, which crashed the process (`exit code
139`, i.e. SIGSEGV) -- another real signal that this code path needs to
reach a clean `close()` to avoid tearing down a live QNN session
mid-teardown, not just a cosmetic JSON fix.

**Attempt 3b -- clean run, both fixes applied:**

```
[INFO]  "Using create From Binary"
[INFO]  "Allocated total size = 816554496 across 7 buffers"
cold load: 13134ms
latency_ms: 1289.86
text: 'Tokyo is in the Japan Standard Time (JST) time zone, which is GMT+9.'
```

Real, correct, on-topic answer (query: "What time zone is Tokyo in?") via
`NpuFastBrain.answer()` end-to-end -- not the raw CLI, the actual `Brain`
seam this project routes through. Process exited cleanly (`brain.close()`
called explicitly, no crash). First real per-token number through this
exact code path: `1289.86ms` total for this short answer, consistent with
the ~94.5ms/token + ~142ms TTFT envelope already captured in
`data/profile_workload/pc_3b.json` from the raw CLI run.

## Attempt 4 -- Phase 6 test suite, one more real finding: no concurrent Genie sessions

Running `tests/test_npu_brain.py` under `.venv-npu` (real `pytest`, added to
that venv since it only had the runtime stack, not test tooling) surfaced a
third real bug, this time a hardware/runtime limitation rather than a code
bug: a second `NpuFastBrain` created while a first one's `GenieDialog`
session was still open (module-scoped pytest fixture kept one alive across
the whole file, while a second test built its own via `TwoBrainRouter`)
failed with a real error:

```
[ERROR] "Could not create context from binary for context index = 2 : err 1002"
[ERROR] "Create From Binary FAILED!"
[ERROR] "Failed to free device: 14003"
[ERROR] "Device Free failure"
Failure to initialize model.
```

This HTP backend/context-binary setup does not support two live Genie
dialog sessions concurrently on this hardware -- not a memory-capacity
issue (816MB x 2 is well inside 31.6GB RAM), a real constraint on
concurrent context-binary slots. Fixed by making the test fixture
function-scoped so only one `NpuFastBrain` session is ever alive at a
time, matching how the actual router uses it (one `fast_brain` instance
held for the router's lifetime, not multiple concurrent instances) --
this is a real test-isolation fix, not a workaround for a code defect in
`NpuFastBrain` itself. Worth remembering if a future change ever tries to
run two tiers' NPU brains concurrently in-process: this hardware won't
support it without closing the first session first.

After the fix, all 4 of Phase 6's real-hardware tests pass under
`.venv-npu`: `test_real_inference_smoke`, `test_npu_ep_assignment`,
`test_router_end_to_end_with_real_brain`, `test_brain_reachable_cleanly`.
