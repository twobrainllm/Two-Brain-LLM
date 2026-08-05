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
