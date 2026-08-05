# convert_model -- real attempts against the hosted QUAD MCP server

Four real `convert_model` calls were made against `https://quad.infra.foundries.io/mcp`
before falling back to mocked output. Kept verbatim because each one surfaced a
genuine, reproducible defect -- useful signal beyond "it didn't work".

## Attempt 1 -- HuggingFace model_source, repo_id only
```json
{"source_format":"pytorch","target_sdk":"qnn","quantization":"int4","aihub":"auto",
 "model_source":{"kind":"huggingface","repo_id":"Qwen/Qwen2.5-1.5B-Instruct"}}
```
Result: `Error calling tool 'convert_model': huggingface source requires 'repo_id' and 'filename'`
-- the tool needs a concrete file inside the repo, not a full PyTorch checkout to trace/export itself.

## Attempt 2 -- HuggingFace model_source, repo_id + filename
```json
{"source_format":"onnx","target_sdk":"qnn","quantization":"int8","aihub":"auto",
 "model_source":{"kind":"huggingface","repo_id":"onnx-community/Qwen2.5-1.5B-Instruct",
                 "filename":"onnx/model.onnx"}}
```
Result: `Error calling tool 'convert_model': huggingface_hub is not installed on the server; cannot fetch HF models.`
-- **the hosted server is missing the `huggingface_hub` package.** The `model_source.kind="huggingface"`
path is dead on this deployment regardless of args. This is a server-side install gap, not something
a client-side retry can work around.

## Attempt 3 -- direct URL model_source (bypasses huggingface_hub), pre-quantized file
```json
{"source_format":"onnx","target_sdk":"qnn","quantization":"int8","aihub":"auto",
 "model_source":{"kind":"url",
   "url":"https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main/onnx/model_int8.onnx"}}
```
Result:
```
Error calling tool 'convert_model': Model uses ConvInteger INT8 format (DynamicQuantizeLinear,
MatMulInteger) -- not supported on Hexagon HTP. The Hexagon NPU requires QDQ
(QuantizeLinear/DequantizeLinear) quantization; a ConvInteger model loads without error but
silently runs on CPU, not the NPU. Re-quantize with QDQ (e.g. onnxruntime quantize_static with
QuantFormat.QDQ, or AIMET), or convert via the QUAD/AI Hub path.
```
-- **real signal, not a shallow failure**: the server actually downloaded and inspected the graph,
and correctly rejected ONNX Runtime's default *dynamic* quantization (ConvInteger/MatMulInteger)
as unsupported on Hexagon HTP, which needs static QDQ. This is genuinely useful validation --
it is exactly the kind of "silently runs on CPU" trap the archetype's per-tier NPU placement
depends on catching.

## Attempt 4 -- direct URL, self-contained FP32 source, ask QUAD to quantize itself
```json
{"source_format":"onnx","target_sdk":"qnn","quantization":"int8","aihub":"auto",
 "model_source":{"kind":"url",
   "url":"https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/main/onnx/model.onnx"}}
```
(Switched to the 0.5B checkpoint specifically because its `model.onnx` has no external-data
sidecar -- a single self-contained file a plain URL fetch can actually load.)

Result:
```
Error calling tool 'convert_model': qairt-converter failed (exit 1):
...
ImportError: libpython3.10.so.1.0: cannot open shared object file: No such file or directory
...
ImportError: cannot import name 'libDlModelToolsPy' from partially initialized module
'qti.aisw.dlc_utils' (circular import)
```
-- **the primary, reproducible blocker.** The hosted server's own QAIRT SDK install
(`/home/quad/work/QUAD/QUAD/sdks/v2.41.0.251128`) is missing `libpython3.10.so.1.0`, so its
`qairt-converter` binary cannot even import, for *any* model, regardless of source/quantization
choice. This is a broken server-side toolchain dependency, not fixable from the client side.

## Attempt 5 -- bypass the hosted server entirely, run qairt-converter locally

The Windows AI PC has its own QAIRT SDK installed
(`C:\Qualcomm\AIStack\QAIRT\2.38.0.250901`), with **three separate
`qairt-converter` builds** shipped in the SDK: `x86_64-linux-clang` (the one
broken on the hosted server), `x86_64-windows-msvc`, and `arm64x-windows-msvc`.
The Windows builds are a completely different binary/toolchain, so they don't
share the missing-`.so` problem. Getting one to actually run surfaced three
more real, fixable environment bugs before hitting a fourth, unfixable one:

1. **QAIRT's own Windows-arch detection is broken on this CPU.**
   `qti.aisw.dlc_utils.__init__` picks its native-extension folder
   (`windows-x86_64` vs `windows-arm64ec`) from `platform.processor()`. On
   this machine that string is the real physical CPU ID
   (`"ARMv8 ... Qualcomm..."`) regardless of which Python process
   architecture is actually running -- so it always resolves to
   `windows-arm64ec`, which fails to load
   (`ImportError: DLL load failed... %1 is not a valid Win32 application`)
   under a native ARM64 CPython, since the shipped `arm64x` `.pyd` needs an
   ARM64EC-capable process. **Fixed** with a launcher shim
   (`qairt_converter_shim.py`) that monkeypatches `platform.processor()`
   before `qti.aisw.dlc_utils` imports, forcing the correct folder for
   whichever interpreter/arch is actually running.
2. **Needed a genuinely x64 Python.** Installed Python 3.10.11 x64 (via
   winget's `Python.Python.3.10` -- the ARM64 platform-tools package had
   earlier failed winget's own hash check, but this one verified fine) to
   run under Windows' x64 emulation and load the SDK's `x86_64-windows-msvc`
   `.pyd`s cleanly, sidestepping the ARM64EC question entirely. 3.10 also
   matches what QAIRT's own `check-python-dependency` script says it
   supports (3.8 or 3.10 only -- this SDK version predates 3.11+).
3. **The pinned `onnx` package is incompatible with a modern `onnx` install.**
   `qti.aisw.converters.onnx.util` does `from onnx import ... mapping` --
   `onnx.mapping` was removed from current `onnx` (installed: 1.22.0), so the
   import silently failed inside a bare `except:` and `onnx` got set to
   `None`, surfacing later as `AttributeError: 'NoneType' object has no
   attribute 'AttributeProto'`. **Fixed** by pinning `onnx==1.14.1`.
4. **Real op-level shape declarations required.** Once it could actually
   parse the model (`onnx-community/Qwen2.5-0.5B-Instruct`'s self-contained
   fp32 `model.onnx`, downloaded locally -- the same file attempt 4 used),
   it demanded explicit static shapes for all 51 dynamic inputs (`input_ids`,
   `attention_mask`, `position_ids`, and `past_key_values.{0..23}.{key,value}`
   -- standard decoder-with-KV-cache ONNX export shape). Supplied via 51
   `-s NAME dim,dim,...` flags (batch=1, num_key_value_heads=2 and
   head_dim=64 from the model's published config.json, past_seq_len=1).

With all four fixed/worked around, the converter got past initialization,
op registration, and shape declaration, and **started actually transforming
real ops** (`INFO_STATIC_RESHAPE: Applying static reshape to
model.embed_tokens.weight`) -- genuine progress into the real compiler, not
a shallow failure. It then hit:

```
ValueError: modeltools::ops::ReshapeOp::calculateShape: Unable to calculate
ReshapeOp output shape for op /model/Unsqueeze_6. shape params dont result
in same cumulative total sum for input and output. 2 != <garbage>
```

`/model/Unsqueeze_6` unsqueezes `attention_mask` (2 elements, by construction
-- `Unsqueeze` never changes element count). Three back-to-back runs of the
literal same command produced three different `<garbage>` values:
`2016977728`, then `-1863713320`, then `1598663576` -- non-deterministic,
including a negative number for what should be a size (total element counts
can't be negative). **That's the signature of uninitialized memory being
read**, not a real shape computation -- a genuine native-code defect inside
`modeltools`'s compiled `ReshapeOp::calculateShape` for this Unsqueeze
pattern in QAIRT 2.38.0.250901's Windows build. It isn't fixable by changing
input shapes (verified: the failure recurs with identical inputs, just with
different garbage each time) and isn't fixable from the Python/CLI layer --
it's inside the closed-source compiled `.pyd`.

## Conclusion

Real `convert_model` calls work all the way up to the point of invoking the actual QNN
compiler -- hardware detection, HTTP fetch, and format/op validation are all live and correct.
The compiler step itself is broken on the hosted deployment (attempt 4). Bypassing the hosted
server and running the same compiler *locally* (attempt 5) got substantially further -- four
real environment bugs fixed/worked around, real ops actually transformed -- before hitting a
second, independent native-code defect (non-deterministic uninitialized-memory read in
`ReshapeOp::calculateShape`) in this QAIRT version's Windows build. Two different execution
paths (hosted Linux server, local Windows SDK), two different genuine compiler-level defects.
`profile_workload` and `orchestrate_workload` were still exercised for real (see their own dirs)
against an existing placeholder artifact in `quad_artifacts/`, which confirmed the same "real
tool, degenerate input" pattern -- both ran and self-reported honestly (`measurement_notes`)
rather than fabricating numbers. The three per-tier model variants below are therefore
**mocked**, shaped to match QUAD's own real response envelope and the real constraints already
discovered (QDQ-only on HTP).
