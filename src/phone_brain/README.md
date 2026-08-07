# Phone Brain (L) — Setup

Target device: Samsung Galaxy S25, Snapdragon 8 Elite (Oryon CPU / Adreno GPU / Hexagon NPU), 12GB RAM.
Model: Llama-3.2-3B-Instruct, quantized w4a16 (Qualcomm's officially benchmarked, published precision for this chipset).

## No phone yet? Start here.

You don't need the S25 in hand to make progress. Two tracks run in parallel:

- **Track A (needs the phone):** the numbered steps below — export, push, serve, benchmark.
- **Track B (needs nothing but this machine):** run `mock_phone_brain_server.py` right
  now and hand `L_INTERFACE_CONTRACT.md` to whoever's building the orchestrator. They
  can build and test all routing logic against the mock today; swapping in the real
  device later is just a URL change, per the contract.

## Precision note: w4a16 vs w4a8

Both are real, supported Qualcomm quantization options — w4a8 isn't broken or
unsupported, it's just been available since May 2024 via the lower-level
`--quantize_full_type w4a8` compile option, while the packaged one-liner export
script this repo uses (`qai_hub_models...llama_v3_2_3b_instruct.export`) ships
Qualcomm's own published, benchmarked w4a16 recipe. Reaching true w4a8 for Llama
means assembling your own `submit_quantize_job` + `submit_compile_job` pipeline
instead of the one-liner. Recommendation: ship w4a16 first (proven, one command),
treat w4a8 as a phase-2 optimization if time allows.

## Confidence estimator — try it now, no phone needed

`confidence_estimator.py` implements the two strategies from
`L_INTERFACE_CONTRACT.md` (self-consistency, self-reported, and a hybrid of
the two) as a runtime-agnostic module against L's contract:

```bash
python mock_phone_brain_server.py --port 8000 &
python confidence_estimator.py --prompt "Prove there are infinitely many primes" --strategy hybrid
```

Or run `./verify_confidence_estimator.sh` for a scripted comparison across
all three strategies on an easy and a hard prompt — this is what
demonstrated the actual cost tradeoff (self-consistency: fixed 3 calls
regardless of difficulty; hybrid: 1 call on the easy prompt, 2 on the hard
one). Once real device numbers come in from `bench_phone_brain.py`, revisit
which strategy is affordable.

## 1. One-time setup (dev machine)

```bash
pip install "qai-hub-models[llama-v3-2-3b-instruct]"

# Sign up at https://aihub.qualcomm.com, then:
qai-hub configure --api_token <YOUR_AI_HUB_TOKEN>

# Llama weights are gated on Hugging Face — request access to
# meta-llama/Llama-3.2-3B-Instruct, then:
huggingface-cli login
```

Before running the export, confirm what precision flags your installed
version actually exposes (don't assume — check):

```bash
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --help
```

## 2. Export the model

```bash
./export_phone_brain.sh genie_bundle_l_phone
```

This compiles Llama-3.2-3B-Instruct into QNN context binaries targeting
`qualcomm-snapdragon-8-elite`, matching the S25's SoC.

## 3. Push to device and serve

```bash
adb push genie_bundle_l_phone /data/local/tmp/genie_bundle_l

# Install GenieX (or the raw Genie/QAIRT runtime) on-device, then serve
# an OpenAI-compatible endpoint, e.g.:
adb forward tcp:8000 tcp:8000
adb shell geniex serve --bundle /data/local/tmp/genie_bundle_l --port 8000
```

## 4. Smoke-test before wiring in the orchestrator

Against the real device once it's up:

```bash
python bench_phone_brain.py --base-url http://localhost:8000 --model llama-3.2-3b-instruct
```

Against the mock, right now, no device needed:

```bash
python mock_phone_brain_server.py --port 8000 &
python bench_phone_brain.py --base-url http://localhost:8000 --model llama-3.2-3b-instruct
```

Check for on your S25 before moving on:
- **Latency / tokens-per-sec** feel interactive (public benchmarks put a
  similar 3B/w4a16 setup around 10-13 tok/s on this chipset class — treat
  that as a rough sanity bound, not a guarantee).
- **Memory headroom** — check `adb shell dumpsys meminfo` while a prompt is
  running; you have 12GB total, but the OS/other apps eat into that.
- **Thermal behavior** over a few back-to-back prompts, since sustained NPU
  load can throttle on a phone in ways it wouldn't on a PC.

## Open questions to settle before the next step (wiring in O)

- [x] Target precision: **w4a16**, confirmed.
- [ ] Decide shared-weights vs. split-weights for O: does the orchestrator
      reuse this same L model with a routing-only system prompt, or does it
      get its own smaller/faster model? (Recommended default: reuse L, keep
      the orchestrator's `max_new_tokens` small so the routing pass stays
      cheap.)
- [ ] Confidence-estimation strategy for O's Route decision — see
      `L_INTERFACE_CONTRACT.md` for the two options (multi-sample
      self-consistency vs. single-call self-reported confidence) and their
      cost tradeoff on-device. This determines how many times per query L
      needs to be called, which is worth deciding before benchmarking
      real-device throughput, not after.
