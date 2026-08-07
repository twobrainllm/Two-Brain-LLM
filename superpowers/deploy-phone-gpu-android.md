# Deploying the Mobile-tier fast brain on the Adreno GPU (Android, S25 Ultra)

Session date: 2026-08-07. Full build history and every dead end, so this
doesn't get re-derived. Written as a handoff doc — if you're picking this up
without the live chat, this plus `docs/PHONE_BRAIN.md` and
`PHONE_DEPLOYMENT_GUIDE_final.md` should be enough context to continue.

## Why this track exists

`src/phone_brain/` (see `docs/PHONE_BRAIN.md`) already targets the NPU via
Qualcomm AI Hub → Genie/QNN, following the exact pattern hollowbyte proved
for the AI-PC tier (`docs/npu-deployment.md`): the NPU has no choice but a
vendor-precompiled binary, because Hexagon's toolchain is closed.

Hollowbyte's AI-PC work also found the *opposite* is true for the Adreno
GPU: it needs **no vendor binary at all**, just a stock quantized GGUF run
through llama.cpp's OpenCL backend (`docs/local-inference-status.md` on his
branch). This doc is that same GPU path, retargeted from his
Windows-Snapdragon-X-Elite build to Android/S25-Ultra — started in parallel
with a stalled NPU attempt so there'd be a working demo either way before
the hackathon deadline, and both ended up working.

## Status as of this session

| Track | Status |
|---|---|
| **GPU** | **Build complete.** `llama-cli` / `llama-server` / `libOpenCL.so` cross-compiled for Android arm64-v8a with `GGML_OPENCL=ON`, verified as real Android PIE executables. **Not yet deployed** — no phone was connected via `adb` when built. Artifacts in `.llama-cpp-opencl-android/` at repo root (gitignored — see below). |
| **NPU** | Export environment fixed and verified (`export.py --help` runs clean, defaults to our exact device). A real compile job was submitted to Qualcomm AI Hub and was still running locally (ONNX/quantization step, ~24GB RAM) when this doc was written — no AI Hub job ID/URL had appeared yet. Check `src/phone_brain/_real_export_attempts_log.md` Attempt 4 and the running job's own output for current state. |

## GPU track — what to do next

1. Connect the S25 Ultra via USB, enable USB debugging (Settings → About
   phone → tap Build number ×7 → Developer Options → USB debugging).
2. `adb devices` should show it.
3. Follow the "Alternative path — GPU (Adreno/OpenCL)" section that was
   added to `src/phone_brain/PHONE_DEPLOYMENT_GUIDE_final.md` — it has the
   exact `adb push` / `llama-server` invocation.
4. You'll need a **Q4_0** GGUF of Llama-3.2-3B-Instruct on the dev machine
   first (not yet downloaded this session) — Q4_0 specifically, because
   Qualcomm's OpenCL backend (`GGML_OPENCL_USE_ADRENO_KERNELS`) is tuned for
   it, and several popular GGUF repos (e.g. Qwen's own) only ship
   Q4_K_M/Q8_0.
5. If the artifacts in `.llama-cpp-opencl-android/` are missing (they're
   gitignored, not vendored — matching the repo's existing
   `.llama-cpp-opencl/` convention for the AI-PC tier), rebuild using the
   recipe below. It's not a quick rebuild — expect to hit the same
   toolchain issues in order unless you follow this exactly.

## The cross-compile problem — full saga

This was the actual hard part of the GPU track, and cost more time than
everything else combined. The dev machine is **Windows on ARM64**, doing
the build inside **WSL2 Ubuntu-22.04** (also aarch64 — WSL2 doesn't cross
host architectures, the guest kernel matches the host). That single fact
broke three different toolchains before one worked.

### Dead end 1 — the Android NDK's own clang

The NDK (tested: r27c) only ships an **x86_64-Linux** prebuilt clang. This
WSL kernel is aarch64 with no x86 emulation registered
(`/proc/sys/fs/binfmt_misc/qemu-x86_64` absent) and no passwordless sudo to
add it. Every invocation: `Exec format error`.

### Dead end 2 — qemu-x86_64 user-mode emulation

Got a real aarch64-native `qemu-x86_64-static` binary running — not from
`multiarch/qemu-user-static` on GitHub (that project only publishes
x86_64-*host* builds, for the opposite direction: x86 hosts emulating arm
targets, the Docker-buildx use case). Instead: Ubuntu's own
`qemu-user-static` package, fetched **without root** via
`apt-get download qemu-user-static` (download-only, unlike `install`, needs
no sudo) then `dpkg-deb -x pkg.deb dir` (extraction, also no sudo needed).

That got a working emulator, but the NDK's clang is **dynamically linked
against glibc**, and qemu-user only emulates CPU instructions — it doesn't
provide a foreign C library. First run: `Could not open
'/lib64/ld-linux-x86-64.so.2'`. Fixing this needs a full x86_64 glibc
rootfs (fetchable the same `apt-get download` + `dpkg-deb -x` way, but
`amd64` `.deb`s from `archive.ubuntu.com` rather than `arm64` ones from
`ports.ubuntu.com`) — untried, abandoned once native LLVM turned out
simpler.

### Dead end 3 — Zig

`zig cc -target aarch64-linux-android` looked ideal: self-contained,
aarch64-native (`zig-aarch64-linux-0.16.0`), no emulation needed. A trivial
`int main(){return 0;}` compile even succeeded — but that's misleading,
because Zig 0.16.0 **does not bundle Android/Bionic libc support at all**.
Confirmed by listing `lib/libc/`: darwin, freebsd, glibc, mingw, musl,
netbsd, openbsd, wasi. No android. Anything that includes a real libc
header (tested: `pthread.h`) fails with "file not found".

Tried pointing Zig at the real NDK sysroot manually
(`-isystem`/`-B`/`-L`) — Zig's own sysroot-joining logic mangled the paths
(visible path duplication in the linker error:
`.../sysroot/home/.../sysroot/usr/lib/...`), and even fixed, Zig's `cc`
frontend refuses to link against a libc flavor it doesn't recognize
("unable to provide libc for target ... android"). Zig does have a
documented escape hatch for exactly this — a `--libc <file>`
description file (`include_dir=`/`sys_include_dir=`/`crt_dir=` pointing at
an external toolchain) — but **it isn't accepted by the `cc` subcommand**
in 0.16.0 ("Unknown Clang option: --libc=..."); that flag apparently only
applies to other zig subcommands (`build-exe`/`run`), not the
clang-compatible shim. Abandoned.

### What actually works — native LLVM 22 + the NDK's sysroot only

The insight: clang has always natively understood
`-target aarch64-linux-android` + `--sysroot=<path>` — that was never the
problem. The *only* problem was "get a clang binary that can execute on
this aarch64 host at all." The NDK's own clang and Zig both failed at that
one job for different reasons; the fix is to bring a different compiler
binary and keep using the NDK purely as a source of *data* (headers, `.so`
stubs), never as something we execute.

1. Download a **native aarch64-Linux** build of real upstream LLVM/clang:
   `https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.8/LLVM-22.1.8-Linux-ARM64.tar.xz`.
   Note the naming quirk — Linux ARM64 releases are `LLVM-*`, not
   `clang+llvm-*` like other platforms, but `bin/clang` is in there.
2. Still grab the Android NDK — not for its clang binary, but for its
   **sysroot** (`toolchains/llvm/prebuilt/linux-x86_64/sysroot/`: headers
   and `.so` stubs, architecture-independent data, not executables) and two
   runtime archives under
   `toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/lib/linux/`.
3. Compiler wrapper (`ndkclang.sh` / `ndkclang++.sh`) — the exact working
   form, after two more rounds of fixing:
   ```bash
   #!/usr/bin/env bash
   NDK_SYSROOT=<ndk>/toolchains/llvm/prebuilt/linux-x86_64/sysroot
   RTSHIM=<shim dir, see below>
   exec <native-llvm>/bin/clang \
     --target=aarch64-linux-android28 \
     --sysroot="$NDK_SYSROOT" \
     -B"$RTSHIM" \
     -L"$RTSHIM" \
     "$@"
   ```
   Two things that look reasonable but **actively break the build**, both
   discovered the hard way:
   - **Do not** add `-resource-dir=<ndk's clang-18 dir>`. It seems like the
     obvious way to hand clang the NDK's builtin support, but it also
     shadows the *native* clang 22's own `arm_neon.h` with the NDK's
     clang-18 version — which encodes NEON intrinsics against clang 18's
     older builtin calling convention. Clang 22 then rejects dozens of
     ggml's SIMD source files: `invalid conversion between vector type
     'float16x8_t'... incompatible constant for this __builtin_neon
     function`. Looks like an unrelated compiler bug; it's a header/builtin
     version mismatch from the `-resource-dir` override.
   - `libclang_rt.builtins.a` and `libunwind.a` — the two Android-target
     runtime archives upstream LLVM's own target doesn't ship, that only
     the NDK provides — need two *different* fixes, because clang resolves
     them two different ways:
     - `libunwind.a` is found via a normal `-l:libunwind.a` search, so
       copying it (renamed from the NDK's arch-suffixed filename) into a
       shim directory and passing `-B<shimdir> -L<shimdir>` works.
     - `libclang_rt.builtins.a` is instead resolved by clang building a
       **hardcoded resource-dir-relative path**
       (`getCompilerRTPath()`) and handing that path straight to the
       linker — `-B`/`-L` never even get consulted for it. The only fix is
       copying the file to the *exact* path clang 22 will look for:
       `<native-llvm>/lib/clang/22/lib/aarch64-unknown-linux-android28/libclang_rt.builtins.a`
       (create the directory; it doesn't exist by default).
4. CMake invocation (both for OpenCL-ICD-Loader and llama.cpp):
   ```
   -DCMAKE_SYSTEM_NAME=Linux          # deliberately NOT "Android" --
   -DCMAKE_SYSTEM_PROCESSOR=aarch64   # that triggers CMake's built-in NDK
   -DCMAKE_C_COMPILER=<wrapper>       # auto-detection, which expects
   -DCMAKE_CXX_COMPILER=<wrapper>     # CMAKE_ANDROID_NDK and fights a
   -DCMAKE_C_COMPILER_WORKS=1         # hand-rolled compiler wrapper.
   -DCMAKE_CXX_COMPILER_WORKS=1
   ```
   llama.cpp's *only* `CMAKE_SYSTEM_NAME STREQUAL "Android"` check just
   flips one default (`LLAMA_SUBPROCESS`) — pass `-DLLAMA_SUBPROCESS=OFF`
   explicitly instead of trying to make CMake think it's really Android.
5. Build order matters: OpenCL-ICD-Loader first
   (`-DOPENCL_ICD_LOADER_HEADERS_DIR=<cloned KhronosGroup/OpenCL-Headers>`)
   to produce a real `libOpenCL.so` stub, *then* llama.cpp with
   `-DGGML_OPENCL=ON -DOpenCL_INCLUDE_DIR=<headers>
   -DOpenCL_LIBRARY=<that libOpenCL.so>`.

Verified end to end: a real pthread compile+link with this wrapper produces
a genuine `ELF 64-bit LSB pie executable, ARM aarch64 ... interpreter
/system/bin/linker64` — and the full llama.cpp build completed with real
`llama-cli`/`llama-server` binaries in the same shape.

## Tooling gotchas hit along the way (unrelated to the toolchain itself)

- **WSL2 needs explicit memory/swap sizing.** No `.wslconfig` defaults to
  roughly half the host's RAM; a pip/torch install OOM-killed at 15.6GB RSS
  against a ~15GB default cap. `qai_hub_models`'s own export step later
  recommended 80GB RAM+swap for the 3B model's local quantization step —
  ended up at `memory=28GB, swap=64GB` in `.wslconfig` (swap is disk-backed,
  cheap to size generously). **Changing `.wslconfig` requires `wsl
  --shutdown` to apply, which kills every running WSL background job** —
  budget for that if something is mid-build when you need to bump memory.
- **Inline `bash -c "...with $VARS..."` through `wsl.exe`, invoked from a
  git-bash-based tool, is unreliable.** Confirmed twice: a `\$PATH`
  meant to stay literal for WSL instead got expanded to the *Windows* PATH
  before `wsl.exe` ever saw it (breaking on the stray `(` in `Program Files
  (x86)`), and plain cross-line variable assignment (`X=hello` then
  `echo $X` on the next line) intermittently returned empty. Fix that works
  reliably: write the script to a real file first (WSL2's filesystem is
  reachable from Windows at `\\wsl.localhost\<distro>\...`), then invoke it
  with a trivial-argv command: `bash /home/user/script.sh` — no embedded
  `$`, quotes, or semicolons left for anything to corrupt.
- **git-bash's MSYS path conversion** separately rewrites any argument that
  looks like a POSIX path (`/home/user/foo.sh`) into a Windows path
  (`C:/Program Files/Git/home/user/foo.sh`) before native executables like
  `wsl.exe` ever see it. Fix: prefix the command with
  `MSYS_NO_PATHCONV=1`.
- **`qai_hub_models[llama-v3-2-3b-instruct]`'s pip resolution** — see
  `src/phone_brain/_real_export_attempts_log.md` Attempt 4 for the full
  writeup; short version, don't let pip resolve the extra freely, extract
  the exact pin list from the wheel's `METADATA` and install everything
  except `aimet-onnx` explicitly.

## Model choice

Sticking with Llama-3.2-3B-Instruct on both tracks. AI Hub already has an
official w4a16 recipe for our exact chipset (defaults to "Samsung Galaxy
S25 (Family)" with zero flags), and all the router-side scaffolding
(`L_INTERFACE_CONTRACT.md`, mock server, confidence estimator) is already
built around it. Phi-3.5-mini-instruct remains the fallback if the NPU
compile job ultimately fails — hollowbyte's AI-PC tier proved it works via
the same AI-Hub-bypass pattern — but that hasn't been needed.

## A note on combining NPU + GPU for one query

Asked and answered in-session: true single-model execution split across
both accelerators isn't practical here — Genie/QNN and llama.cpp/OpenCL are
separate runtimes with incompatible model formats (QNN context binary vs.
GGUF), so there's no shared compute graph to divide. What *is* practical,
using the router's existing `Brain` protocol
(`routing/brains.py`): treat NPU and GPU as two independent brains and
either race the same prompt on both (lower latency, ~2x power cost) or
round-robin concurrent requests across them (higher throughput under
multi-user load, same per-query latency). Not yet implemented — flagged as
a next step if there's time after both tracks are actually deployed.
