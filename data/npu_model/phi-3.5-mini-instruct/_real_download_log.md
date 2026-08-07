# phi-3.5-mini-instruct NPU artifact -- real download log

Per `superpowers/deploy-local-brain-npu.md` Phase 2a: bypasses QUAD's
`convert_model` entirely. This is Qualcomm's own pre-compiled QNN context
binary for Phi-3.5-mini-instruct, targeting this machine's exact chip
(Snapdragon X Elite), fetched directly -- no AI Hub account, no cloud
compile job, no WSL.

`_mock: false` for everything under this directory. Nothing here is
generated or guessed; every file is byte-for-byte what the two URLs below
served.

## Source of the download URLs

`qualcomm/Phi-3.5-mini-instruct` on Hugging Face does **not** host the
binaries as repo files (its API `siblings` list is just
`.gitattributes`, `LICENSE`, `README.md`, `release_assets.json`). The
model card's download buttons resolve through `release_assets.json`,
fetched real:

```
GET https://huggingface.co/qualcomm/Phi-3.5-mini-instruct/raw/main/release_assets.json
```

Relevant contents (verbatim):

```json
{
  "version": "0.59.0",
  "precisions": {
    "w4a16": {
      "chipset_assets": {
        "qualcomm-snapdragon-x-elite": {
          "genie": {
            "tool_versions": { "qairt": "2.43.1.260218" },
            "download_url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/phi_3_5_mini_instruct/releases/v0.59.0/phi_3_5_mini_instruct-genie-w4a16-qualcomm_snapdragon_x_elite.zip"
          }
        }
      }
    }
  }
}
```

Confirms the exact QAIRT version the binary was compiled against:
**2.43.1.260218** -- more precise than the model card's rounded "QAIRT
2.43", and the number `superpowers/deploy-local-brain-npu.md` R9 checks
against this machine's `onnxruntime-qnn` pip package (2.4.0, bundling QAIRT
2.48.40 -- newer, the supported compatibility direction).

## Files fetched, real

### `raw/` -- Qualcomm's own genie bundle (S3, public, no auth)

```
curl -L -o raw/phi_3_5_mini_instruct-genie-w4a16-qualcomm_snapdragon_x_elite.zip \
  "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/phi_3_5_mini_instruct/releases/v0.59.0/phi_3_5_mini_instruct-genie-w4a16-qualcomm_snapdragon_x_elite.zip"
```

Server-reported size (`HEAD`, real): `Content-Length: 2078158835` (≈1.94 GiB),
`Content-Type: binary/octet-stream`, `Last-Modified: Tue, 28 Jul 2026`.

**No license gate, no account, no token required** -- a plain public S3
GET. This directly contradicts the earlier plan draft's assumption that an
AI Hub account/API token would be needed for this step; that's only true of
Phase 2b's self-compile fallback, not this pre-built download.

**Download complete, real:** final size on disk `2078158835` bytes (matches
the `Content-Length` above exactly). SHA256 (`sha256sum`, real):
`4e1573f75d666e0bf2a24678dcfb539edfc9e7c8a109e1c3f9b8913633718e13`.
Extraction was a plain `unzip` -- no tooling beyond that. The zip also
contained a `__MACOSX/` sidecar directory (AppleDouble metadata from
whatever tool Qualcomm zipped this with on their end); harmless, left in
place, ignored by everything downstream.

Real, extracted file listing (`ls -la`) of
`raw/phi_3_5_mini_instruct-genie-w4a16-qualcomm_snapdragon_x_elite/`:

| File | Bytes (real) |
|---|---|
| `genie_config.json` | 5,376 |
| `htp_backend_ext_config.json` | 428 |
| `sample_prompt.txt` | 116 |
| `tokenizer.json` | 1,844,304 |
| `weight_sharing_model_..._1_of_4.serialized.bin` | 98,766,848 |
| `weight_sharing_model_..._2_of_4.serialized.bin` | 1,200,652,288 |
| `weight_sharing_model_..._3_of_4.serialized.bin` | 1,200,791,552 |
| `weight_sharing_model_..._4_of_4.serialized.bin` | 100,626,432 |

### `wrapper/` -- ONNX Runtime GenAI wrapper + tokenizer bundle

Source: `onnx-community/Phi-3-mini-instruct-hexagon-npu-assets` on Hugging
Face -- confirmed via its real API `siblings` listing, then each file
fetched individually (plain public repo, no gating):

```
curl -L -o wrapper/<file> \
  "https://huggingface.co/onnx-community/Phi-3-mini-instruct-hexagon-npu-assets/resolve/main/<file>"
```

| File | Bytes (real, `ls -la`) |
|---|---|
| `dequantizer.onnx` | 181 |
| `genai_config.json` | 27,684 |
| `position-processor.onnx` | 2,246,935 |
| `position-shifter.onnx` | 136,750 |
| `quantizer.onnx` | 59,682 |
| `special_tokens_map.json` | 599 |
| `tokenizer.json` | 3,897,866 |
| `tokenizer_config.json` | 3,495 |

All 8 files landed, all HTTP 200, all non-empty and structurally plausible
for their type (`genai_config.json`/`tokenizer_config.json` parse as JSON
with expected keys on inspection).

## Real finding: `wrapper/` and `raw/` are two different, incompatible runtimes

Confirmed by inspection, not assumed: **`wrapper/genai_config.json`'s
decoder pipeline references ONNX-wrapped QNN context binaries by name**
(`ar128_cl4096_1_of_4_qnn_ctx.onnx`, `ar1_cl4096_1_of_4_qnn_ctx.onnx`, etc.)
-- files for ONNX Runtime GenAI's QNN execution provider. **Neither
`wrapper/` nor `raw/` actually contains any file with those names.**

Re-fetched `onnx-community/Phi-3-mini-instruct-hexagon-npu-assets`'s live HF
API `siblings` listing to rule out an incomplete original download -- it
genuinely only ships the 8 files above (auxiliary graphs + tokenizer/config).
It never hosted the weight-bearing `*_qnn_ctx.onnx` files this
`genai_config.json` requires; the repo (`lastModified` 2025-03-05) predates
Qualcomm's current (`v0.59.0`, Jul 2026) "genie"-only release flavor for
this model on `release_assets.json`.

**`raw/`'s `genie_config.json` is Qualcomm's own Genie SDK config format**
(dialog/context/sampler/engine/model sections, `.serialized.bin` weight
files) -- a different runtime from ONNX Runtime GenAI entirely, with no
compatibility relationship to `wrapper/`'s pipeline. The `wrapper/` bundle
is not usable with this `raw/` download; the reverse -- running `raw/`
through Genie's own runtime, which it was built for -- is real and proven
working (see `_real_inference_smoke_log.md`).

**Consequence for `superpowers/deploy-local-brain-npu.md`'s Decision
section:** `NpuFastBrain` calls Qualcomm's Genie C API (`Genie.dll`,
shipped inside the `onnxruntime-qnn` pip wheel -- see Phase 1's DLL
inventory) via `ctypes`, not `onnxruntime-genai`. This is still in-process,
still no server, still the same `Brain` seam -- the only change from the
original plan is which native library gets called. `wrapper/`'s files stay
downloaded and documented here (a real repo, really fetched) but are
unused by the shipped `NpuFastBrain`; nothing in the Decision section's
in-process/no-server reasoning changes.

## Re-placed in a second checkout (2026-08-06) -- copied, not re-fetched

The `.../Downloads/QUAD/QUAD-Client-main/samples/two_brain_privacy_router`
checkout started with no `raw/` at all (it is gitignored). Rather than a fresh
S3 GET, the zip was copied from a sibling clone on the same machine
(`C:\Users\qc_de\dev\hollowbyte\Two-Brain-LLM`) and verified against the
receipt above **before** extraction:

- size `2078158835` -- exact match
- SHA256 `4e1573f75d666e0bf2a24678dcfb539edfc9e7c8a109e1c3f9b8913633718e13`
  -- exact match

Since the hash matches, the bytes are provably identical to what S3 served, so
this is equivalent for correctness. It is **not** an independent re-verification
that the URL above is still live -- that was not attempted on this pass, and
nothing here should be read as claiming it.

### Correction to the file table above: `genie_config.json` is 5,375 bytes, not 5,376

Verifying the extracted files against the table in "Files fetched, real" turned
up one mismatch. Seven of eight match exactly; `genie_config.json` came out of
the verified zip at **5,375** bytes.

Byte-diffing against the sibling clone's copy found the cause at offset 694:
that clone's file reads `"use-mmap": false` where the pristine archive says
`"use-mmap": true`. `true` -> `false` is exactly the missing byte. So the
**table above recorded a locally-modified file**, not the archive's own
contents -- the edit is documented nowhere in that clone, and this is the first
time it has been noticed.

The pristine value is kept here, unmodified. `use-mmap: true` was then tested
and is not merely harmless: cold load is roughly twice as fast (6.5 s vs the
13.1 s Attempt 3b recorded against the edited config). See
`_real_inference_smoke_log.md` Attempt 5.

## Not vendored into git

Per `.gitignore`, `data/npu_model/**/raw/`, `*.zip`, `*.bin`, `*.dll`,
`*.so`, `*.cat`, and `*.onnx` under this tree are excluded -- these are
large binaries fetched from stable public URLs, reproducible from this log,
not something to commit. `genai_config.json`, `tokenizer*.json`, and
`special_tokens_map.json` (small, text, useful for review) remain trackable.
