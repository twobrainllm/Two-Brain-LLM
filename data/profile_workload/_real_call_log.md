# profile_workload -- real call against the hosted QUAD MCP server

`convert_model` never produced a real per-tier LLM artifact (see
`../convert_model/_real_attempts_log.md`), so `profile_workload` was instead exercised for real
against the only complete artifact already sitting in `quad_artifacts/`
(`mobilenetv2-12-static-qdq/model.data`, from an earlier unrelated session) -- to prove the tool
itself, not to get an LLM number.

```json
{"model_path": "quad_artifacts/mobilenetv2-12-static-qdq/model.data",
 "platform": "windows", "runtime": "auto", "duration_s": 5, "profiling_level": "basic"}
```

Real result (trimmed): the call succeeded (no error), but every timing field came back zero
because `model.data` alone isn't a loadable graph (`"Unrecognised model extension"` -- it's the
external-data sidecar with no matching `.onnx`). The tool was honest about this rather than
fabricating numbers -- worth keeping as a reference for how QUAD self-reports low-confidence
data:

```json
"measurement_notes": {
  "latency": "not_measured:parser_no_match",
  "layers": "synthetic_composite:no_diagview_csv",
  "memory": "not_measured:process exited before first sample",
  "utilization": "measured:psutil_cpu",
  "power": "estimated:host_thermal_model"
}
```

`device.chipset` in the response was `"AMD EPYC 7B12"` -- the hosted server's own container, not
this client's Snapdragon X Elite (same self-detect gap as `hardware_detect`, see
`../../docs/GAPS.md` #1).

The three per-tier files in this directory are mocked, using this real envelope shape but with
plausible measured-style values for an actually-loadable Genie/QNN LLM bundle.

## Update: `pc_3b.json` is now a real capture, not a mock

Per `../../superpowers/deploy-local-brain-npu.md` Phase 2-4, a real Genie/QNN
LLM bundle (Phi-3.5-mini-instruct) actually ran on this machine's Hexagon
NPU -- see `../npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md`
for the full trace. `pc_3b.json` was replaced with the real numbers from
that run. Diff against the old mock:

| Field | Mocked (old) | Real (new) | Note |
|---|---|---|---|
| `model_id` | `qwen2.5-3b-instruct` | `phi-3.5-mini-instruct` | mock guessed the wrong model entirely |
| `runtime` | `npu` (generic) | `genie_qnn_htp` | real runtime is Qualcomm's Genie SDK, not ONNX Runtime GenAI |
| `latency_ms.ttft_mean` | 140 | 142 | coincidentally close |
| `latency_ms.per_token_mean` | 44.8 | 94.5 | **real hardware is ~2.1x slower than the mock guessed** -- this is exactly the kind of gap this project's mocks exist to be caught and corrected against, not evidence the mock-writer was careless |
| `throughput_tokens_per_s` | 22.3 | 10.6 | follows from per-token change above |
| `power_mw`, `energy_per_1k_tokens_mj`, `memory_peak_mb`, `utilization` | plausible-looking mocked numbers | `0` / not fabricated | no power/memory/utilization instrumentation was attached to the real run -- rather than replace one guess with another, these are honestly marked `not_measured` in the new file's `measurement_notes`, matching this real log's own convention above (fields QUAD itself couldn't measure came back `0` with a reason, not a plausible substitute) |

`mobile_1b.json` and `cloud_large.json` remain mocked -- mobile-tier NPU
deployment and Cloud AI 100 plumbing are out of scope for this pass (see
`deploy-local-brain-npu.md`'s Non-goals).
