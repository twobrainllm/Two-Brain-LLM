# Gaps hit while building this project

Five real, reproducible gaps surfaced while driving the four QUAD MCP tools
against the hosted server (`https://quad.infra.foundries.io/mcp`) and, for
gap 3, a local QAIRT SDK install as a workaround. None of these are
guesses -- each has a captured error string or a diffable JSON response
backing it (see `data/*/_real_*_log.md`).

## 1. `hardware_detect` self-reports the server's container, not the client device

Calling `hardware_detect(platform="windows")` directly over MCP returns the
**hosted server's own** hardware (`AMD EPYC 7B12`, Ubuntu 24.04, no real NPU)
-- not this machine's real Snapdragon X Elite. `quad-client detect` avoids
this by falling back to a `client-local-static` probe
(`discovery_source` field), which is what `data/hardware_detect/ai_pc.json`
actually used. Anything that calls the raw MCP tool directly for
hardware-dependent decisions (as `profile_workload`/`orchestrate_workload`
do internally -- see their `device.chipset` fields) inherits the same
mis-detection.

## 2. G8 -- privacy/PII guardrail (`quad.privacy`) not present in this checkout

Tracked as `[G8 - DELIVERED]` in QUAD's private core-platform repo: a
detect/mask/rehydrate PII guardrail with a masked-token invariant, already
wired into a chat product elsewhere. A full-repo search of
`QUAD-Client-main` (this checkout) found zero references to a
`quad.privacy` module -- it lives in the private `QUAD` core repo, which
per this workspace's `CLAUDE.md` is not checked out on this machine.
**Workaround applied (per the documented gap note): mocked the same
detect/mask/rehydrate contract in `privacy_mask.py`**, real Python with a
regex-based detector, so the routing logic around it is genuinely testable.
Swap `privacy_mask.PIIGuard` for the real `quad.privacy` client once this
environment has access to that gap's delivered component, and re-run
`tests/test_router.py::test_escalated_pii_never_reaches_cloud_unmasked`
against it unchanged -- the contract (`mask`/`rehydrate`/invariant) is
designed to match.

## 3. `convert_model`'s compiler is broken on the hosted deployment

Four real attempts (`data/convert_model/_real_attempts_log.md`) walked
through every `model_source` kind before hitting the actual blocker:

1. `huggingface` source_source without `filename` -- rejected (needs an
   exact file, not a checkout).
2. `huggingface` source with `filename` -- **the server has no
   `huggingface_hub` package installed**, so this source kind is dead
   regardless of args.
3. `url` source, a pre-quantized (dynamic-quant) ONNX file -- correctly
   **rejected by real op-format validation**: Hexagon HTP needs static QDQ
   quantization, not ONNX Runtime's default `ConvInteger`/`MatMulInteger`
   dynamic quant. Good signal, not a bug.
4. `url` source, a self-contained FP32 file, asking QUAD to quantize itself
   -- **`qairt-converter` fails to even import**:
   `ImportError: libpython3.10.so.1.0: cannot open shared object file`.
   The hosted server's own QAIRT SDK install
   (`/home/quad/work/QUAD/QUAD/sdks/v2.41.0.251128`) is missing a shared
   library dependency. This blocks the compiler for *any* model.

This is a server-environment defect (missing `.so`), not fixable from the
client side. It requires either shell access to `quad.infra.foundries.io`
(not available in this session -- see below) or the QUAD platform team
installing the missing shared library (e.g. `libpython3.10` /
`python3.10-dev` on whatever base image the server container uses).

**3b. Bypassing the server entirely: real, but a second independent
defect.** This machine has its own QAIRT SDK install with Windows-native
`qairt-converter` builds, a completely different toolchain from the
broken Linux one on the server. Running it directly (attempt 5 in
`data/convert_model/_real_attempts_log.md`) required fixing three real
environment bugs -- QAIRT's Windows arch-detection is broken on this CPU
(worked around with a `platform.processor()` shim), the pinned `onnx`
package doesn't match a modern `onnx` install (`onnx.mapping` was
removed -- fixed by pinning `onnx==1.14.1`), and the model's 51 dynamic
KV-cache inputs needed explicit static shapes. With all three fixed, the
converter got well into real op transformation before hitting a **second,
independent native-code defect**: `modeltools::ops::ReshapeOp::calculateShape`
reads uninitialized memory for a specific `Unsqueeze` pattern -- proven by
running the identical command three times and getting three different
non-deterministic garbage totals (`2016977728`, `-1863713320`,
`1598663576`) for a tensor that should total exactly `2`. This is inside
QAIRT 2.38.0.250901's closed-source compiled `.pyd`, not fixable from
the Python/CLI layer -- worth filing against the SDK release itself,
separately from the missing-library defect on the hosted server.

`profile_workload` and `orchestrate_workload` were still exercised for real
against a placeholder artifact already present in `quad_artifacts/` to
confirm those two tools work correctly and self-report honestly on bad
input (see their own `_real_call_log.md` files) -- so the per-tier numbers
in this project's `data/` are mocked only for the compile-dependent path,
not because the tools themselves are unreachable.

## 5. Mobile `hardware_detect` -- resolved, but surfaced two more gaps

**5a. adb authorization needed a physical tap (resolved).** No `adb` binary
existed on the build host at all; installed Android SDK platform-tools
directly from `dl.google.com` (winget's `Google.PlatformTools` package
failed its own installer hash check). Windows then confirmed a phone was
physically attached (`Get-PnpDevice`: `Galaxy S25 Ultra`, WPD class;
`ADB Interface`, USBDevice class, `Status=Unknown`) but `adb devices`
listed nothing until the user enabled Developer Options > USB debugging
and accepted the on-device "Allow USB debugging?" prompt directly on the
phone -- an intentional Android security control with no remote/software
bypass (UI Automator can't help either: it runs *over* adb, so it can't
bootstrap the very authorization it depends on). Once accepted,
`adb devices` showed the device (serial `R3CXC0804XN`, `model:pa3quew`).

**5b. `quad-client detect --platform android` itself returns placeholder
data.** With the device authorized and reachable, running the client's own
detect command (`discovery_source: client-local-static`, confirming it did
reach the device) still returned `chipset: "unknown"`, `cpu_cores: 1`,
`ram_gb: 0.1`, and `storage_gb: 475.6` -- that last number is this Windows
PC's own disk size, not the phone's, meaning the android code path is
falling through to stale/default values rather than a real per-device
probe. Worked around by querying the device directly:
`adb shell getprop ro.soc.model` (`SM8750`), `ro.product.model`
(`SM-S938U1`), `ro.build.version.release` (`16`), and
`adb shell cat /proc/meminfo` (`MemTotal: 11381324 kB`). This is a real bug
in `quad_mcp_client`'s android hardware-detection path, separate from and
in addition to gap #1 (the MCP server's own self-detect issue) -- worth
filing upstream.

`data/hardware_detect/mobile.json` is now real (`_mock: false`), sourced
from direct `adb shell` queries against the attached Galaxy S25 Ultra
(Snapdragon 8 Elite for Galaxy / SM8750), with `npu_tops: null` because
TOPS isn't exposed via `getprop` and isn't in the client's SoC hint table
for this platform (unlike the AI PC tier, where the hint table does
resolve 45 TOPS for X Elite).

## 4. Cloud AI 100 is not a modeled target anywhere in QUAD-Client-main

Separate from gaps 1-3: `hardware_detect`'s platform enum
(`windows|linux|android|gateway|robotics`) and `convert_model`'s
`target_sdk` enum (`qnn|snpe|nwaios|executorch`) have no cloud/AI100 value
at all. "Cloud AI 100" appears exactly once in this repo, as prose in an
unrelated deliverable doc. The Cloud tier in this project is therefore
mocked end-to-end (hardware, convert, profile) from Qualcomm's public
datasheet -- there was no tool call to even attempt.
