# Wiring the phone into the router

How the Mobile tier's fast brain is reached, how to run it without the S25,
and what is deliberately still open. The contract itself is
[`L_INTERFACE_CONTRACT.md`](L_INTERFACE_CONTRACT.md); the audit of what was
built on the `local_brain` branch is [`PHONE_BRAIN.md`](PHONE_BRAIN.md).

---

## What was added

| Piece | Where |
|---|---|
| `OpenAIHttpBrain` | `routing/brains.py` — a fast brain served over `POST /v1/chat/completions` |
| `BrainUnavailableError` | `routing/brains.py` — "endpoint is down", the one error the router escalates on |
| `TWO_BRAIN_PHONE_BRAIN=1` gate | `routing/router.py::_build_fast_brain` |
| Unreachable ⇒ escalate | `routing/router.py::route` |
| Mock server | `tools/phone/mock_phone_brain_server.py` |
| Tests | `tests/test_phone_brain_wiring.py` (14, all offline) |

`OpenAIHttpBrain` is deliberately **not** phone-specific. It is one class for
any served OpenAI-shaped model, following the precedent `NpuFastBrain` set:
a flat class in the existing `brains.py`, `urllib` imported lazily inside the
request method, registered by an env-var switch in `_build_fast_brain`. No
subpackage split — see `PHONE_BRAIN.md`'s superseded-layout note.

**`policy.py` and the four privacy invariants are untouched.** That was the
test of whether the seam was drawn in the right place.

---

## Run it without a phone

```bash
python tools/phone/mock_phone_brain_server.py --port 8000

TWO_BRAIN_PHONE_BRAIN=1 PYTHONPATH=src python -m two_brain_router
```

The mock speaks the identical contract, so swapping to the real device is a
`TWO_BRAIN_PHONE_URL` change and nothing else. If anything else has to change,
the contract has drifted.

## Run it with the S25

```bash
adb devices                       # confirm the phone is attached
adb forward tcp:8000 tcp:8000     # host 127.0.0.1:8000 -> phone's 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"hi"}]}'

TWO_BRAIN_PHONE_BRAIN=1 PYTHONPATH=src python -m two_brain_router
```

**`adb forward`, not `adb reverse`.** `L_INTERFACE_CONTRACT.md` says `reverse`,
which is backwards for this direction: `reverse` opens a port *on the device*
that forwards to the host, and here the server is on the phone and the caller
is on the laptop. Worth fixing in that doc.

**Port 8000 may already be taken** by the UI's `server.py`. Use
`--port 8777` plus `TWO_BRAIN_PHONE_URL=http://127.0.0.1:8777` if so.

---

## Loopback is enforced

`OpenAIHttpBrain` refuses a non-loopback `base_url` unless constructed with
`allow_remote=True`. That closes `PHONE_BRAIN.md` reconciliation point #1:
`base_url` was an ordinary argument, so pointing it at a remote host would
send the query off-device silently.

`adb forward` makes the phone reachable *at 127.0.0.1 over USB*, so the strict
default costs nothing in the intended setup — which is the real argument for
ADB as the transport for text and images. It is not only a throughput choice;
it is the option that makes the on-device claim checkable.

Verified end-to-end: with the query `email bob@example.com and call
555-123-4567 about lunch`, what reached the endpoint was
`email [PII_EMAIL_1] and call [PII_PHONE_1] about lunch`. Masking happens
before the brain is called, per invariant #1, and rehydration only after the
answer is back.

---

## Unreachable means escalate, not crash

Per the contract's error handling, a timeout or non-200 from the served brain
is an escalate signal. `route()` catches `BrainUnavailableError` specifically
and falls through to the deep brain, appending a note to the audit trail. A
phone that is unplugged, asleep or thermally throttled no longer takes the
user's request down with it.

Two deliberate limits:

- **Only `BrainUnavailableError` falls through.** A genuine bug in a brain
  still raises, rather than being silently converted into a cloud call.
- **An image-bearing query re-raises.** Escalating one requires
  `describe_image()` on the very brain that just failed, so there is nothing
  to fall back to. Failing loudly beats silently dropping the image and
  answering about the text alone.

---

## Still open

**1. Confidence is not wired in — this is the significant one.**
`confidence_estimator.py` returns a real signal; `signals/difficulty.py` is
still surface-feature scoring. They were not connected here because
`PHONE_BRAIN.md` #2 is a genuine structural problem, not a plumbing gap:
self-reported confidence returns the answer *and* the score in the same call,
but `route()` scores difficulty *then* calls the brain. Wiring it as-is means
escalation throws away an answer already paid for — on a phone, real latency
and battery.

Two honest options, and this is a decision for whoever owns the orchestrator:

- Use `self_consistency`/`hybrid` as a separate scoring call before routing —
  keeps `route()`'s ordering, costs 1–2 extra on-device inferences.
- Restructure `route()` to accept "the answer arrived early" — cheaper at
  runtime, but it changes the shape of the privacy-critical path and deserves
  its own review.

When it does land: consume `confidence`, ignore `should_escalate`
(`PHONE_BRAIN.md` #3 — their threshold is `0.5`, `RoutePolicy`'s is `0.55`,
and their own contract says the orchestrator owns the decision).

**2. `data/profile_workload/mobile_1b.json` is still the Qwen2.5-1.5B mock**
(ttft 180 ms, 27.5 ms/token) while the phone runs Llama-3.2-3B w4a16. Real
numbers come from `bench_phone_brain.py` on the device, with a receipt per the
`data/` rule. Until then the local-latency estimate for the mobile tier is
describing a different model than the one answering.

**3. Video and audio have no path yet.** `Brain.answer` is one request, one
response; a stream has no shape there. That needs a protocol extension
alongside `Brain`/`VisionBrain` — and note that streams cannot be masked
(`PIIGuard` is regex over text, which a face or a voice defeats), so they must
follow the `describe_image` pattern: consumed on-device, with only a masked
textual description ever eligible to escalate.
