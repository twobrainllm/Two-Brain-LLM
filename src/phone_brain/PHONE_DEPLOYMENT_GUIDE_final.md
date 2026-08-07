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
python test_phone_brain.py --base-url http://localhost:8000 --model llama-3.2-3b-instruct
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
- [ ] `test_phone_brain.py` runs clean with a real, recorded tokens/sec number
- [ ] `confidence_estimator.py --strategy self_reported` runs against the
      real device and the `CONFIDENCE:` line actually parses

## Troubleshooting

- **Device not found by ADB** — different cable/port, re-check USB
  debugging authorization, `adb kill-server && adb start-server`.
- **Port already in use** — `adb reverse --remove tcp:8000` then redo Part 6.
- **Hugging Face access still pending** — check email for the gated-repo approval; it isn't instant.
- **Compile job stuck or slow** — check the AI Hub Workbench dashboard for job status rather than assuming it's hung.
- **Server crashes or OOMs on the 3B model** — check `adb shell dumpsys meminfo`; consider lowering `CONTEXT_LEN` in `export_phone_brain.sh` and re-exporting.

## Known unknowns, stated plainly

- Exact GenieX on-device install packaging (Part 4) — check its own docs once downloaded; I haven't personally confirmed the exact command.
- Whether the real model's self-reported confidence is actually well-calibrated — Part 8 is where you find out, not before.

---

## Alternative path — GPU (Adreno/OpenCL), no QNN/AI-Hub account needed

Built in parallel with the NPU path above (2026-08-07) so there's a working
demo either way before the hackathon deadline, following the same pattern
hollowbyte proved for the AI-PC tier: the NPU needs a vendor-precompiled
binary (Hexagon's toolchain is closed), but the Adreno GPU needs no vendor
binary at all — just a stock quantized GGUF run through llama.cpp's OpenCL
backend. **This path never touches Qualcomm AI Hub, torch, or the gated HF
weights download for compilation** — only the GGUF (a public quant) is
needed at deploy time.

### What's already built — use these directly, don't rebuild

Already cross-compiled for Android arm64-v8a with `GGML_OPENCL=ON`,
targeting the S25 Ultra's Adreno 830, and sitting on disk on this same
machine **right now** — if you're tunneled into this box, skip straight to
"Deploying to the phone" below with these exact paths (gitignored, so they
won't show up via `git pull` — that's expected, read them straight off
disk):

| File | Full path | What it is |
|---|---|---|
| `llama-server` | `C:\Users\qc_de\Two-Brain-LLM\.llama-cpp-opencl-android\llama-server` | Android arm64 OpenAI-compatible server binary — `ELF ... interpreter /system/bin/linker64` |
| `llama-cli` | `C:\Users\qc_de\Two-Brain-LLM\.llama-cpp-opencl-android\llama-cli` | Android arm64 CLI binary — same interpreter |
| `libOpenCL.so` | `C:\Users\qc_de\Two-Brain-LLM\.llama-cpp-opencl-android\libOpenCL.so` | ICD loader stub the phone's real Adreno driver resolves against at runtime |

From WSL (e.g. to `adb push` them — `adb` itself runs from the Windows
side, see below), the same files are at
`/mnt/c/Users/qc_de/Two-Brain-LLM/.llama-cpp-opencl-android/`.

Only rebuild from the recipe below if these files are gone or you need a
different target (different API level, different quant kernel set, etc).

### The cross-compile problem, and how it was solved

This was the hard part — worth recording so it isn't re-derived. The
Android NDK's own bundled clang is **x86_64-Linux only**; this dev
environment cross-compiles from WSL2 on an ARM64 Windows host, so the NDK's
clang can't execute at all (`Exec format error`, no x86 emulation
registered in this WSL kernel, no sudo to add it). Two workarounds were
tried and abandoned before landing on the one that works:

- **qemu-x86_64 user-mode emulation** of the NDK's clang: got the emulator
  running (Ubuntu's `qemu-user-static` `.deb`, extracted without root via
  `apt-get download` + `dpkg-deb -x`), but the NDK's clang is dynamically
  linked against glibc, which this filesystem doesn't have at all — would
  need a full foreign-arch glibc rootfs too. Abandoned.
- **Zig** (`zig cc -target aarch64-linux-android`): aarch64-native,
  self-contained, looked ideal — but Zig 0.16.0 does not bundle
  Android/Bionic libc support at all (confirmed: its `lib/libc/` has
  darwin/freebsd/glibc/mingw/musl/netbsd/openbsd/wasi, no android). Trivial
  compiles succeed and mask this; anything that includes a real libc header
  fails. Zig's own external-libc mechanism (a `--libc <file>`
  description file) isn't accepted by the `cc` subcommand. Abandoned.
- **What actually works:** a real, **native aarch64-Linux** build of
  upstream LLVM/clang (`LLVM-<ver>-Linux-ARM64.tar.xz` from
  `github.com/llvm/llvm-project` releases — note the unusual naming, it's
  `LLVM-*` not `clang+llvm-*` for this platform, but clang is bundled in
  `bin/`), pointed at the **NDK's sysroot** via plain `--sysroot=` (headers
  and `.so` stubs are architecture-independent data, not executables — the
  NDK's x86_64-only clang *binary* was never actually needed, only its
  sysroot contents). Two extra fixes were needed beyond that:
  1. Don't set `-resource-dir` to the NDK's own clang-18 resource
     directory — it shadows the native clang's own `arm_neon.h` with an
     older, incompatible NEON-intrinsics ABI and breaks ggml's SIMD code
     across dozens of files.
  2. `libclang_rt.builtins.a` and `libunwind.a` (the two Android-target
     runtime archives the NDK provides that upstream LLVM's target doesn't
     ship) need copying into place by hand — `libunwind.a` resolves via a
     normal `-B`/`-L` search, but `libclang_rt.builtins.a` is looked up by
     clang at a **hardcoded resource-dir-relative path**
     (`<llvm>/lib/clang/<ver>/lib/aarch64-unknown-linux-android28/libclang_rt.builtins.a`)
     that ignores `-B`/`-L` entirely — the fix is literally copying the
     file there.

  Full recipe, wrapper scripts, and every dead end: see the
  `two-brain-phone-deploy-progress` memory entry from this session, or ask
  to have this section expanded into a standalone build script if
  rebuilding from scratch.

### Deploying to the phone

```bash
adb devices                         # confirm the S25 Ultra shows as "device"

# adb.exe isn't on PATH on this machine -- full path:
# C:\Users\qc_de\AppData\Local\Android\Sdk\platform-tools\adb.exe

adb push "C:\Users\qc_de\Two-Brain-LLM\.llama-cpp-opencl-android\llama-server" /data/local/tmp/
adb push "C:\Users\qc_de\Two-Brain-LLM\.llama-cpp-opencl-android\libOpenCL.so"  /data/local/tmp/
adb shell chmod +x /data/local/tmp/llama-server

# Pull a Q4_0 GGUF of Llama-3.2-3B-Instruct onto the dev machine first --
# Q4_0 specifically: Qualcomm's OpenCL backend (GGML_OPENCL_USE_ADRENO_KERNELS)
# is tuned for Q4_0, and stock GGUF repos (e.g. Qwen's own) often only ship
# Q4_K_M/Q8_0 -- an unsloth-style Q4_0 reupload may be needed.
adb push Llama-3.2-3B-Instruct-Q4_0.gguf /data/local/tmp/

adb reverse tcp:8080 tcp:8080
adb shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp \
  OCL_ICD_FILENAMES=/data/local/tmp/libOpenCL.so \
  ./llama-server -m Llama-3.2-3B-Instruct-Q4_0.gguf -ngl 99 -c 4096 --port 8080"
```

`-c`/`--n-ctx` is mandatory per hollowbyte's AI-PC notes — without it,
llama.cpp's auto-fit context sizing can blow past device memory. `-ngl 99`
offloads all layers to the Adreno GPU via OpenCL. Smoke-test from the dev
machine against `http://localhost:8000` the same way as the NPU path's
Part 7, once `adb reverse` is set up.

**Not yet done:** actually running this on the physical S25 Ultra (no
device was connected when the binaries were built), and getting a Q4_0
GGUF onto the phone. Both are the immediate next steps once hardware is
available.
