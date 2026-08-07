# Deployment Guide — Building O Against This Repo

**Audience:** whoever's building the orchestrator (O). You don't need the
Galaxy S25, a Qualcomm AI Hub account, or anything about Genie/QNN to use
this repo — just this guide and `L_INTERFACE_CONTRACT.md`.

## What's in here for you

- `mock_phone_brain_server.py` — a stand-in for the real on-device L,
  speaking the exact same API contract. Develop and test all of O's
  routing logic against this, today, no phone needed.
- `L_INTERFACE_CONTRACT.md` — the actual contract. Read this first, it's
  short. It also documents why we're on **self-reported confidence** (not
  self-consistency) and what that means for how L behaves.
- `confidence_estimator.py` — reference implementation of the
  confidence-parsing logic (`self_reported()`, plus `self_consistency()`
  and `hybrid()` as fallbacks if self-reported confidence turns out
  poorly calibrated once real data comes in). Import it directly rather
  than reimplementing the parsing from scratch.

## 1. Get the code

```bash
git clone <repo-url>
cd phone_brain
```

No dependencies to install for anything in this list — everything here is
Python 3 standard library only. (The export pipeline that produces the
real on-device model has its own separate setup, covered in `README.md`
— that's not your side of the fence.)

## 2. Run the mock L server

```bash
python3 mock_phone_brain_server.py --port 8000
```

Leave this running in a terminal. It behaves like the real on-device model
for API purposes: confident answers on easy-looking prompts, hedging
answers on prompts that look hard (keyword heuristic), and always includes
a `CONFIDENCE: NN` line in its response — matching the self-reported
approach the contract settled on.

## 3. Talk to it

Directly:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"..."}]}'
```

Or through the reference implementation:

```bash
python3 confidence_estimator.py --prompt "..." --strategy self_reported
```

Or import it into your own O code:

```python
from confidence_estimator import estimate_confidence

result = estimate_confidence(base_url, model, prompt, strategy="self_reported")
if result.should_escalate:
    ...  # hand off to C
```

`estimate_confidence` returns a `ConfidenceResult`: `confidence` (0.0–1.0),
`should_escalate` (bool), `calls_made`, `total_latency_sec`, and the raw
response text(s). Build O's routing decision on top of this shape rather
than re-parsing `CONFIDENCE:` lines yourself.

## 4. Where this fits in what you're building

Per the project doc, O owns three decisions per query:

- **Route** — this repo covers this piece. `confidence_estimator.py` is
  the building block; the actual threshold/policy logic is yours.
- **Mask** — pseudonymizing PII before anything leaves the device. Not in
  this repo. Entirely your territory.
- **Compress** — pruning redundant context before sending to C. Also not
  in this repo, also yours.

This repo is deliberately scoped to just the L side of Route — it's not
trying to hand you an orchestrator, just the piece it depends on.

## 5. Switching from mock to the real device later

Nothing in your code should change — only the `--base-url` / base URL you
point at. When the real S25 is ready, it'll be reachable the same way
(`http://localhost:8000/...` via `adb reverse tcp:8000 tcp:8000`, or the
phone's IP directly). If making that switch requires changing your code,
the contract has drifted — flag it so `L_INTERFACE_CONTRACT.md` gets fixed
rather than working around it in O.

## Known limitations of the mock — don't over-read the numbers

- Confidence values are canned by a keyword heuristic, not real model
  reasoning. Good for confirming your *parsing and routing logic* works
  end-to-end, not for predicting real accuracy or escalation rates.
- Latency is randomly simulated (0.4–1.2s per call), not calibrated to the
  real S25's throughput.
- Once real numbers exist from `bench_phone_brain.py` on the actual device,
  expect to retune whatever temperature/threshold values you land on now.
