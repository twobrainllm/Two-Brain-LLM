# L (Phone Brain) → O (Orchestrator) Interface Contract

This is the only thing O needs to know about L. Whoever builds O does not
need to know anything about Genie, QNN, GenieX, or the S25 — just this
contract. It's identical whether O is talking to the real on-device model
or the mock server in this folder.

## Endpoint

```
POST http://<host>:8000/v1/chat/completions
Content-Type: application/json
```

- Real device: `<host>` is the S25's IP (or `localhost` via `adb forward tcp:8000 tcp:8000`).
- Local dev: `<host>` is `localhost`, backed by `mock_phone_brain_server.py`.

> **`forward`, not `reverse` — and this document said `reverse` until it was
> run against real hardware.** The two are opposites, and only one of them
> works here:
>
> - `adb forward tcp:8000 tcp:8000` opens port 8000 **on the host** and tunnels
>   it to the device. The server is on the phone and the caller (`O`, the
>   router) is on the dev machine, so this is the one you want.
> - `adb reverse tcp:8000 tcp:8000` opens port 8000 **on the device** and
>   tunnels it to the host. That is for an app on the phone calling a server on
>   your machine — the opposite of this contract.
>
> The error survived a long time because it is invisible against
> `mock_phone_brain_server.py`: the mock listens on the dev machine's own
> loopback, so `http://localhost:8000` works whether or not any adb tunnel
> exists, and works even if the wrong one was set up. It only fails with a
> phone actually attached. `tools/phone/live_test.py` runs the correct command
> and is the executable statement of this.

## Request

```json
{
  "model": "llama-3.2-3b-instruct",
  "messages": [{"role": "user", "content": "<prompt text>"}],
  "max_tokens": 256,
  "temperature": 0.2
}
```

## Response

```json
{
  "id": "...",
  "object": "chat.completion",
  "model": "...",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "<answer text>"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 48,
    "total_tokens": 60
  }
}
```

`usage.completion_tokens` combined with wall-clock latency is what you
need for tokens/sec (used in the energy/latency side of the demo).

## Confidence / difficulty signal (for O's Route decision)

**DECIDED: self-reported confidence.** L answers and self-rates in one
call; O applies the threshold and makes the final routing decision. L
never decides to escalate itself — it only ever produces a number, O owns
the decision every time, same as the privacy mask (L doesn't decide what's
sensitive either).

**This is mostly O's job, not L's — but L's design constrains which options
are actually cheap enough to use, so it's worth agreeing on before O's
routing logic is built.**

The doc's plan is for O to estimate confidence "from log probabilities or
self-consistency." Do not build routing around logprobs being present —
whether the real device server exposes them isn't confirmed (llama.cpp's
own server supports OpenAI-style logprobs, and GenieX can run on a
llama.cpp backend, so it's plausible, but untested by us). Two real options,
different cost profiles:

**Option 1 — multi-sample self-consistency (more robust, more expensive).**
All the computation is O's: it calls L 2–3 times on the same prompt
(temperature > 0, e.g. 0.5–0.7), compares the responses (embedding
similarity, or a cheap heuristic like keyword overlap for a first pass),
and treats agreement as a confidence signal. L does nothing special beyond
honoring the `temperature` parameter already in this contract — but the
cost lands on L: **every single user query now means 2-3 sequential
on-device inference calls before O even knows whether to escalate.** On a
phone, that's 2-3x the latency and battery draw of a single answer, not a
rounding error. Whether this is affordable depends entirely on L's
measured tokens/sec on the real device (see `bench_phone_brain.py`) — this
is a real number to check, not an assumption.

**Option 2 — single-call self-reported confidence (chosen).**
O asks L, in one call, to answer *and* rate its own confidence (e.g. via a
structured prompt: "answer, then output CONFIDENCE: 0-100"). One inference
call instead of 2-3. Less statistically grounded than self-consistency
(the model's stated confidence isn't necessarily well-calibrated), but far
cheaper on-device, and easier to parse than free-form logprob math. **Test
this against the real device early** — if the self-reported numbers don't
track actual correctness well once you have real queries running, fall
back to `hybrid()` in `confidence_estimator.py` rather than trying to fix
calibration in the prompt.

Recommendation: benchmark L's real per-call latency on the S25 first, then
decide. If a single call is already close to the edge of "feels
interactive," Option 1 probably isn't affordable and Option 2 (or a hybrid
— cheap self-report first, escalate to a 2nd sample only when self-reported
confidence is borderline) is the more realistic path. This is squarely a
question for whoever owns O, but the answer depends on data only L's side
can produce.

Either way, this works identically against the mock server and the real
device. If logprobs *do* turn out to be available later, treat them as a
bonus signal to blend in, not a dependency.

## Error handling O should assume

- Request timeout: treat as "L failed," fall back to escalate-to-C rather
  than retry indefinitely (mirrors the doc's cloud-unreachable fallback,
  just inverted).
- Non-200 response: same — treat as an escalate signal, don't block the user.

## Swapping mock → real

Nothing in O's code should change except the base URL. If it does, the
contract has drifted and needs to be fixed here first.
