# orchestrate_workload -- real call against the hosted QUAD MCP server

Same placeholder artifact as `profile_workload` (see its `_real_call_log.md`):

```json
{"model_path": "quad_artifacts/mobilenetv2-12-static-qdq/model.data", "power_mode": "balanced"}
```

Real result (trimmed): succeeded, and correctly forced the whole graph to CPU because it
couldn't recognize any NPU-placeable op in an unloadable input --

```json
"allocation": {"model": "cpu"},
"cpu_utilization_pct": 100.0,
"decisions": [{"category": "op_allocation", "choice": "1 op(s) forced to CPU",
  "reason": "The NPU/HTP does not support op types: composite.", "severity": "warn"}]
```

This confirms the tool's fallback/decision-reporting logic is real and works correctly even on
a degenerate input. The per-tier files in this directory are mocked with the same
`allocation`/`decisions` shape, using the QDQ-compatible op mixes implied by
`../convert_model/*.json`'s `unsupported_ops` lists.

## Update: `pc_3b.json` is now a real capture, not a mock

Real Phi-3.5-mini-instruct Genie/QNN run (see
`../npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md`) showed
every one of 40 loaded graphs assigned to the `QnnHtp` backend with zero
forced-to-CPU decisions -- unlike this real degenerate-input call above,
which did see one. `pc_3b.json`'s `power_mode` also changed from the
mocked `"balanced"` to the real `"burst"`, matching
`htp_backend_ext_config.json`'s actual shipped `perf_profile` value for
this artifact. `npu_utilization_pct`/`cpu_utilization_pct` moved from the
mocked 91/9 split to a real 100/0, since Genie's backend selection here is
all-or-nothing per bundle, not a per-op split the way QUAD's schema models
it -- see the new file's `_reason` field for why that's a real finding, not
an invented "cleaner" number.

`mobile_1b.json` remains mocked (mobile-tier NPU deployment is out of
scope for this pass, per `../../superpowers/deploy-local-brain-npu.md`'s
Non-goals).
