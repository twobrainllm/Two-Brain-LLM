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
