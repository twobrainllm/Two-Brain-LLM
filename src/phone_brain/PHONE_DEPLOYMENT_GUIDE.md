# Phone Brain — End-to-End Deployment & Test Guide (Real Device)

**This guide is for getting L actually running on the physical Galaxy S25**
— not the mock. If you're building O and just need something to develop
against today, you don't need this yet: see `DEPLOYMENT_GUIDE.md` instead.

## What happens where

| Step | Runs on | Needs the phone? |
|---|---|---|
| Export / quantize the model | Dev machine (submits to Qualcomm's cloud) | No |
| Install the runtime (GenieX/QAIRT) | Dev machine pushes it → phone | Yes |
| Push the compiled model bundle | Dev machine → phone via ADB | Yes |
| Serve the model | On the phone itself | Yes |
| Test / benchmark / call it | Dev machine, over an ADB tunnel | Yes (server must be running) |

Only the middle three steps actually touch the S25. Export is pure dev-machine + cloud.

## Part 0 — Prerequisites checklist

- [ ] Galaxy S25, USB cable, dev machine (Mac/Linux/Windows)
- [ ] Qualcomm AI Hub account + API token (https://aihub.qualcomm.com → Account → Settings → API Token)
- [ ] Hugging Face account with **gated access approved** for `meta-llama/Llama-3.2-3B-Instruct` (request it now if not — approval isn't instant)
- [ ] ADB installed on the dev machine (comes with Android SDK Platform Tools)
- [ ] Python 3.10+ on the dev machine

## Part 1 — One-time dev machine setup

```bash
pip install "qai-hub-models[llama-v3-2-3b-instruct]"
qai-hub configure --api_token <YOUR_AI_HUB_TOKEN>
huggingface-cli login
```

Confirm what your installed version actually supports before trusting anything below:

```bash
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --help
```

## Part 2 — Enable the phone for development

1. Settings → About phone → tap **Build number** 7 times → unlocks Developer Options.
2. Settings → Developer Options → enable **USB debugging**.
3. Connect the S25 via USB.
4. On the dev machine:
   ```bash
   adb devices
   ```
   You should see the S25's serial listed as `device`. If it says
   `unauthorized`, check the phone screen for an RSA fingerprint prompt and
   tap **Allow**.

**If the phone doesn't show up:** try a different cable/port, run
`adb kill-server && adb start-server`, and re-check the USB debugging
authorization prompt on the phone (it resets sometimes).

## Part 3 — Export & compile the model (dev machine only)

```bash
./export_phone_brain.sh genie_bundle_l_phone
```

On Windows, use `export_phone_brain.ps1` instead — same export, plus
preflight checks (Python/venv, `qai_hub_models` importable, AI Hub config,
Hugging Face token) that fail fast with a specific fix rather than partway
through the real compile job:

```powershell
.\export_phone_brain.ps1 -OutputDir genie_bundle_l_phone
```

If PowerShell refuses to run it, either unblock it for the session
(`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`) or invoke it
directly (`powershell -ExecutionPolicy Bypass -File .\export_phone_brain.ps1`).

This submits a real compile job to Qualcomm's cloud — expect real
wall-clock minutes, not instant. Output: a `genie_bundle_l_phone/`
directory containing the QNN context binaries. This step does **not**
touch the phone at all yet.

## Part 4 — Get the runtime onto the phone

Install GenieX (or the raw QAIRT/Genie SDK) on the S25 itself — this is
what will actually load and execute the bundle. Being upfront: the exact
install command depends on how GenieX packages itself (APK vs. a pushed
native binary), which I haven't personally confirmed — check GenieX's own
install instructions once you've got the SDK downloaded. It'll be one of:

```bash
adb install <geniex.apk>
# or
adb push <geniex-binary> /data/local/tmp/
adb shell chmod +x /data/local/tmp/<geniex-binary>
```

## Part 5 — Push the compiled model bundle

```bash
adb push genie_bundle_l_phone /data/local/tmp/genie_bundle_l
```

Verify it landed:

```bash
adb shell ls -la /data/local/tmp/genie_bundle_l
```

## Part 6 — Start serving, on-device

```bash
adb reverse tcp:8000 tcp:8000
adb shell geniex serve --bundle /data/local/tmp/genie_bundle_l --port 8000
```

Leave this running. `adb reverse` is what lets your dev machine reach
`localhost:8000` and transparently have it land on the phone — nothing on
the calling side needs to know ADB is involved at all.

## Part 7 — Smoke test from the dev machine

```bash
python bench_phone_brain.py --base-url http://localhost:8000 --model llama-3.2-3b-instruct
```

What "working" looks like: coherent text back, no errors, and a
tokens/sec number in a plausible range (public benchmarks put a similar
3B/w4a16 setup around 10 tok/s on this chipset class — treat that as a
rough sanity bound, not a guarantee for your exact setup).

While a prompt is running, worth checking once:

```bash
adb shell dumpsys meminfo   # memory headroom — 12GB total, OS/apps eat into it
```

## Part 8 — Validate the confidence-estimator pipeline on the real device

```bash
python confidence_estimator.py --base-url http://localhost:8000 --model llama-3.2-3b-instruct --prompt "<something hard>" --strategy self_reported
```

**Note:** `verify_confidence_estimator.sh` is mock-only by design — it
launches its own mock server on port 8321 to regression-test the
estimator code itself. Don't point it at the real device; call
`confidence_estimator.py` directly against `localhost:8000` instead, as above.

This step matters more than it looks: it's the **first real test of
whether the actual model reliably follows the self-report instruction
format** ("output CONFIDENCE: 0-100") — something the mock can't tell you,
since the mock always fakes that line regardless of what's asked. If the
real model's `CONFIDENCE:` line doesn't parse cleanly or its confidence
doesn't track actual answer quality, that's the calibration risk flagged
in `L_INTERFACE_CONTRACT.md` — fall back to `hybrid()` rather than trying
to fix it purely by rewording the prompt.

## Part 9 — "Done" checklist

- [ ] `adb devices` shows the S25
- [ ] Export completed, `genie_bundle_l_phone/` exists
- [ ] Runtime installed on the phone
- [ ] Bundle pushed to `/data/local/tmp/genie_bundle_l`
- [ ] Server running, reachable at `localhost:8000` via the dev machine
- [ ] `bench_phone_brain.py` runs clean with a real, recorded tokens/sec number
- [ ] `confidence_estimator.py --strategy self_reported` runs against the
      real device and the `CONFIDENCE:` line actually parses

## Troubleshooting

- **Device not found by ADB** — different cable/port, re-check USB
  debugging authorization, `adb kill-server && adb start-server`.
- **Port already in use** — `adb reverse --remove tcp:8000` then redo Part 6.
- **Hugging Face access still pending** — check email for the gated-repo approval; it isn't instant.
- **Compile job stuck or slow** — check the AI Hub Workbench dashboard for job status rather than assuming it's hung.
- **Server crashes or OOMs on the 3B model** — check `adb shell dumpsys meminfo`; consider lowering the context length and re-exporting: the `CONTEXT_LEN` env var on `export_phone_brain.sh`, or `-ContextLength` (falls back to the same `CONTEXT_LEN` env var if unset) on `export_phone_brain.ps1`.

## Known unknowns, stated plainly

- Exact GenieX on-device install packaging (Part 4) — check its own docs once downloaded; I haven't personally confirmed the exact command.
- Whether the real model's self-reported confidence is actually well-calibrated — Part 8 is where you find out, not before.
