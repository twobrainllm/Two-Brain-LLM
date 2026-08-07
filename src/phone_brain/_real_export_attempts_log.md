# Real export attempts — `export_phone_brain.sh` toolchain

Per `CLAUDE.md`'s receipts rule: this logs actual tool calls, arguments, and
verbatim errors for attempts to run the Llama-3.2-3B-Instruct → Genie/QNN
export (`export_phone_brain.sh`) on this machine. No model artifact has been
produced yet — everything below is a blocker, not a success.

Machine: Windows 11, **ARM64** (Snapdragon). WSL2 default distro:
`Ubuntu-22.04` (also ARM64 — WSL2 does not cross architectures; the guest
kernel matches the host).

---

## Attempt 1 — native Windows venv (`src/phone_brain/venv-x64`)

```
venv-x64\Scripts\python.exe -m qai_hub_models.models.llama_v3_2_3b_instruct.export --help
```

Fails at import time:

```
File ".../qai_hub_models/models/_shared/llm/common.py", line 5, in <module>
    import fcntl
ModuleNotFoundError: No module named 'fcntl'
```

`fcntl` is POSIX-only. `qai_hub_models`'s LLM model code imports it
unconditionally at module load, so this export path is **not runnable on
native Windows at all**, regardless of CPU architecture. Not previously
documented — `venv-x64` predates this log and was never actually exercised
against this specific model's export module.

---

## Attempt 2 — WSL Ubuntu-22.04 (ARM64), fresh venv

Environment built fresh (`python3 -m venv` unavailable without
`apt install python3.10-venv`, which needs a sudo password not available in
this session — worked around via `pip install --user virtualenv` +
`virtualenv ~/phone_brain_venv`, no sudo needed):

```
pip install "qai-hub-models[llama-v3-2-3b-instruct]"
```

Installed clean (pulls torch 2.5.1, transformers 4.45.0, qai-hub-models
0.32.0, ~2GB). `fcntl` resolves fine here — confirms attempt 1's blocker was
Windows-specific, not this model's code.

`--help` then failed on an unrelated pinned-version conflict:

```
File ".../datasets/features/features.py", line 634, in <module>
    class _ArrayXDExtensionType(pa.PyExtensionType):
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
```

`datasets==2.14.5` (pinned by `qai_hub_models/models/llama_v3_2_3b_instruct/requirements.txt`)
needs the pre-14.0 `pyarrow` API; pip resolved `pyarrow==25.0.0`. Fixed:

```
pip install "pyarrow<14"   # landed on 13.0.0
```

---

## Attempt 3 — same env, real blocker: `aimet-onnx` has no `linux_aarch64` wheel

`--help` still fails after the pyarrow fix:

```
File ".../qai_hub_models/models/_shared/llm/model.py", line 752, in <module>
    class LLM_AIMETOnnx(AIMETOnnxQuantizableMixin, LLMConfigEditor, BaseModel, ABC):
NameError: name 'AIMETOnnxQuantizableMixin' is not defined
```

`qai_hub_models/global_requirements.txt` pins
`aimet-onnx==2.6.0; sys_platform == 'linux' and python_version == "3.10"` —
exactly this environment — but it isn't pulled in by the
`[llama-v3-2-3b-instruct]` extra, and installing it directly fails:

```
pip install "aimet-onnx==2.6.0"
ERROR: Could not find a version that satisfies the requirement aimet-onnx==2.6.0 (from versions: none)
```

Confirmed via PyPI's JSON API (`pypi.org/pypi/aimet-onnx/json`): version 2.6.0
ships exactly one wheel, `aimet_onnx-2.6.0-cp310-cp310-manylinux_2_34_x86_64.whl`.
**No `linux_aarch64` wheel exists for `aimet-onnx` at any released version**
(checked 0.0.1 through 2.36.0 latest). The model module imports the
`AIMETOnnxQuantizableMixin` class unconditionally at load time, so this isn't
a quantization-only feature gap — the whole `llama_v3_2_3b_instruct` export
module is unimportable on this architecture without it.

**This is the same gap class already logged in
`superpowers/deploy-local-brain-npu.md` (risk R2): "ONNX quantization tooling
only supported on x86_64 today."** That effort's fix was isolating the
quantization step to a separate x86_64 Python process. The same detour is
the likely fix here — an x86_64 WSL distro or emulated interpreter — but it
has not yet been attempted for this model, and Qualcomm's own package
publishing x86_64-only wheels is an external constraint, not something fixable
from this repo.

**Status at the time: blocked, unresolved.** No AI Hub compile job had been
submitted — neither an NPU (HTP) nor a GPU-targeted variant, since the
export module couldn't even be imported yet to inspect its
`--target-runtime`/backend flags. AI Hub and HuggingFace credentials were
never touched in this session (user declined to relay tokens through the
assistant, correctly).

---

## Attempt 4 (2026-08-07) — RESOLVED: the extra pulls aimet-onnx, but the
## export code itself doesn't need it for an already-quantized model

Picked back up in a fresh WSL venv, AI Hub token and HF login already
configured from a prior step. `qai_hub_models`'s own runtime message (see
`--help` output) states plainly: *"Some quantized models require the
AIMET-ONNX package, which is only supported on Linux. Quantized model can
be exported without this requirement."* — AIMET is not a hard requirement
of the export path we actually need (Qualcomm's pre-published w4a16 recipe
for an already-quantized checkpoint), it's a transitive pin of the
`[llama-v3-2-3b-instruct]` **packaging extra**, which bundles everything
needed for *every* possible workflow including on-the-fly PTQ.

Confirmed unfixable in general: `aimet-onnx` genuinely has no `linux_aarch64`
wheel at any version (same finding as Attempt 3), and no alternate package
index was found publishing one.

**The fix:** stop letting pip resolve the extra as a whole (its resolver
backtracks ~25+ qai-hub-models releases hunting for a version where
`aimet-onnx` is satisfiable, never finds one, and lands on an ancient
0.32.0-era release with an incompatible `datasets`/`pyarrow`/`huggingface_hub`
combo instead — a different-looking failure that's really the same root
cause). Instead:

```bash
pip download --no-deps qai-hub-models==0.59.1 -d /tmp/qdl
cd /tmp/qdl && unzip -q *.whl -d extracted
grep -E "^Requires-Dist" extracted/*/METADATA | grep 'extra == "llama-v3-2-3b-instruct"'
```

This lists the *exact*, mutually-consistent pin set for the extra (it's
literally the first candidate pip tries before backtracking begins — proven
consistent by that fact alone). Install `qai-hub-models==0.59.1 --no-deps`,
then install every pin from that list **except `aimet-onnx`** explicitly
alongside `torch==2.10.0` in one `pip install` command. Resolves instantly,
no backtracking, no version drift.

Verified working: `python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --help`
runs clean and defaults `--device` to **"Samsung Galaxy S25 (Family)"** — our
exact target, no `--chipset` override needed. A real export job
(`--skip-profiling`) was submitted and was still running (deep in the
ONNX/quantization step, ~24GB RSS) as of this log entry — see
`docs/PHONE_BRAIN.md` and `PHONE_DEPLOYMENT_GUIDE_final.md` for current
status and the full pin list.

The earlier WSL memory-crash investigation (separate issue, same session)
is documented in the `two-brain-phone-deploy-progress` Claude memory entry
if resuming without that context.
