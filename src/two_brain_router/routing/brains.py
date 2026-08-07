"""The two brains, behind one interface.

The cloud brain is still a stub: the Cloud AI 100 has no plumbing in
QUAD-Client at all (gap #4), so it returns a *labeled* stub answer with
latency/cost estimated from `profile_workload`'s envelope shape.

The `pc_3b` local tier is no longer a stub. `NpuFastBrain` runs a real
Phi-3.5-mini-instruct artifact on this machine's Hexagon NPU via Qualcomm's
Genie SDK (`Genie.dll`, called through `ctypes` -- not ONNX Runtime GenAI;
see `data/npu_model/phi-3.5-mini-instruct/_real_download_log.md` for why).
`LocalFastBrain` remains the stub used for the mobile tier, which is still
blocked (gap #3/#3b + #5b) -- see
`superpowers/deploy-local-brain-npu.md`'s Non-goals.

This is the seam to replace with real inference: implement `Brain.answer` and
keep the returned `BrainResponse` shape, and the router, policy, and privacy
guarantees above it are unchanged.
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from two_brain_router.signals.loader import DATA_DIR, TierSignals


@dataclass
class BrainResponse:
    """What any brain must return: the text, plus what it cost to get it."""

    text: str
    latency_ms: float
    cost_usd: float = 0.0


class Brain(Protocol):
    """Implement this to plug a real runtime in behind the router."""

    def answer(self, masked_query: str, context: str = "") -> BrainResponse: ...


def _estimate_tokens(masked_query: str) -> int:
    """Token count the latency/cost estimates are driven off."""
    return max(len(masked_query.split()) * 2, 16)


class LocalFastBrain:
    """On-device fast brain (Mobile 1B / AI PC 3B).

    Returns a labeled stub -- no compiled artifact exists to actually run (see
    data/convert_model/_real_attempts_log.md). Latency is estimated from
    profile_workload's real envelope shape (data/profile_workload/<tier>.json).
    """

    def __init__(self, tier: str, signals: TierSignals) -> None:
        self.tier = tier
        self.signals = signals

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        latency_ms = self.signals.profile["latency_ms"]
        n_tokens = _estimate_tokens(masked_query)
        est_latency = latency_ms["ttft_mean"] + n_tokens * latency_ms["per_token_mean"]
        return BrainResponse(
            text=f"[local:{self.tier} mock fast-brain response to: {masked_query!r}]",
            latency_ms=est_latency,
            cost_usd=0.0,
        )


class CloudDeepBrain:
    """Off-device deep brain (Cloud AI 100).

    Also a labeled stub: the Cloud AI 100 has no real plumbing in QUAD at all
    (data/hardware_detect/cloud_ai100.json). Latency and cost are estimated
    from data/profile_workload/cloud_large.json, network RTT included.
    """

    def __init__(self, signals: TierSignals) -> None:
        self.signals = signals

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        prof = self.signals.profile
        latency_ms = prof["latency_ms"]
        n_tokens = _estimate_tokens(masked_query)
        est_latency = (
            latency_ms["network_rtt_mean"]
            + latency_ms["ttft_mean"]
            + n_tokens * latency_ms["per_token_mean"]
        )
        cost = (n_tokens / 1000.0) * prof["token_cost_usd_per_1k"]
        return BrainResponse(
            text=(
                f"[cloud:ai100 mock deep-brain response to: {masked_query!r} "
                f"| context_used={context!r}]"
            ),
            latency_ms=est_latency,
            cost_usd=cost,
        )


class CloudBrainError(RuntimeError):
    """The Cloud AI 100 inference endpoint returned a non-OK response."""


class CirrascaleDeepBrain:
    """Off-device deep brain, running for real on hosted Cloud AI 100 silicon.

    Replaces `CloudDeepBrain`'s labeled stub. Talks to Cirrascale's AI Suite
    endpoint (`INFERENCE_CLOUD_ENDPOINT`, OpenAI-shaped `/chat/completions`)
    with `INFERENCE_CLOUD_API_KEY`. Credentials come from the environment --
    never from a tracked file; `secrets.txt` is gitignored.

    **What this does and does not close of gap #4.** It makes the cloud tier
    *real*: a real network hop, a real large model, real token counts, real
    measured latency. It does **not** make Cloud AI 100 a modeled target in
    QUAD-Client -- `hardware_detect`'s platform enum still has no cloud value
    (see docs/GAPS.md #4). And the API reports nothing about the silicon it
    runs on: the Cloud AI 100 attribution comes from Cirrascale's own service
    documentation, not from anything observable here. `/models`, `/health` and
    the completion response were checked; none expose device information.
    Recorded per the data/ receipts rule rather than asserted.

    Only masked text reaches this class -- that is invariant #3, enforced by
    the router, and this class must never be given the vault.
    """

    #: Published Cirrascale catalogue: (usd_per_1m_input, usd_per_1m_output).
    #: All four are 8K context. These are the vendor's list prices, not
    #: measured spend -- but unlike the datasheet guess they replaced (which
    #: implied ~$1800/1M and made a single query look like $1.03), they are
    #: real published rates.
    MODEL_PRICING: dict[str, tuple[float, float]] = {
        "Llama-3.1-8B": (0.02, 0.22),
        "Qwen-QwQ-32B": (0.08, 0.36),
        "Llama-3.3-70B": (0.19, 0.69),
        "DeepSeek-R1-Distill-Llama-70B": (0.19, 0.69),
    }
    #: Every catalogue model is 8K. Escalation sends masked query + compressed
    #: context, so this is a real ceiling on how much context may cross.
    CONTEXT_LIMIT_TOKENS = 8192

    #: The 70B is the better deep brain -- an 8B is only ~2x the local fast
    #: brain, and this device already runs an 8B locally, so an 8B deep brain
    #: buys throughput rather than capability. Falls back automatically when
    #: the 70B is unavailable, which is a real and observed condition.
    DEFAULT_MODEL = "Llama-3.3-70B"
    FALLBACK_MODEL = "Llama-3.1-8B"
    _MAX_NEW_TOKENS = 512
    _TIMEOUT_S = 300

    def __init__(
        self,
        signals: TierSignals,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.signals = signals
        self.model = model or os.environ.get("TWO_BRAIN_CLOUD_MODEL") or self.DEFAULT_MODEL
        self._endpoint = (endpoint or os.environ.get("INFERENCE_CLOUD_ENDPOINT") or "").rstrip("/")
        self._api_key = api_key or os.environ.get("INFERENCE_CLOUD_API_KEY") or ""
        if not self._endpoint or not self._api_key:
            raise CloudBrainError(
                "INFERENCE_CLOUD_ENDPOINT and INFERENCE_CLOUD_API_KEY must be set "
                "(export them from secrets.txt -- it is gitignored and must stay so)."
            )

    def _cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Real published rates, billed separately for input and output."""
        rate_in, rate_out = self.MODEL_PRICING.get(model, (0.0, 0.0))
        return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1_000_000

    def _post(self, model: str, messages: list[dict]) -> tuple[dict, float]:
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {"model": model, "messages": messages, "max_tokens": self._MAX_NEW_TOKENS}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._endpoint}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._TIMEOUT_S) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Surface the service's own message -- it distinguishes "busy",
            # "invalid model" and auth failures, which matter differently.
            # Never echo the request itself: it carries the Authorization header.
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("message", "")
            except Exception:  # noqa: BLE001
                pass
            raise CloudBrainError(f"cloud endpoint HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise CloudBrainError(f"cloud endpoint unreachable: {exc.reason}") from exc
        return body, (time.perf_counter() - start) * 1000

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": masked_query})

        # The catalogue's larger models are frequently "Models Busy/Unavailable"
        # (HTTP 500) -- observed for the 32B and both 70Bs, and for the 8B too
        # during a service-wide dip, so this is load state rather than
        # provisioning. Degrade to the smaller model rather than failing the
        # whole escalation; the caller can see which one actually answered.
        candidates = [self.model]
        if self.FALLBACK_MODEL not in candidates:
            candidates.append(self.FALLBACK_MODEL)

        last_error: CloudBrainError | None = None
        for model in candidates:
            try:
                body, latency_ms = self._post(model, messages)
            except CloudBrainError as exc:
                last_error = exc
                continue
            if "choices" not in body:
                last_error = CloudBrainError(f"unexpected response: {str(body)[:200]}")
                continue
            usage = body.get("usage", {})
            self.model_used = model
            return BrainResponse(
                text=body["choices"][0]["message"]["content"].strip(),
                latency_ms=latency_ms,
                cost_usd=self._cost_usd(
                    model,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                ),
            )
        raise last_error or CloudBrainError("no cloud model answered")


class GenieError(RuntimeError):
    """A Genie C API call returned a non-success `Genie_Status_t`."""


#: Where Phase 2a's real download landed -- see
#: data/npu_model/phi-3.5-mini-instruct/_real_download_log.md. Override with
#: TWO_BRAIN_DATA_DIR (same convention as signals/loader.DATA_DIR) to point
#: at a different capture.
_NPU_ARTIFACT_DIR = (
    DATA_DIR
    / "npu_model"
    / "phi-3.5-mini-instruct"
    / "raw"
    / "phi_3_5_mini_instruct-genie-w4a16-qualcomm_snapdragon_x_elite"
)

#: GenieDialog_SentenceCode_t values this module needs (GenieDialog.h).
_GENIE_SENTENCE_COMPLETE = 0
_GENIE_SENTENCE_END = 3
_GENIE_SENTENCE_ABORT = 4

#: GenieDialog_Action_t (GenieDialog.h).
_GENIE_DIALOG_ACTION_ABORT = 0x01

#: GenieDialog_QueryCallback_t (GenieDialog.h): void(const char* response,
#: GenieDialog_SentenceCode_t sentenceCode, const void* userData).
_GenieQueryCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)


def _genie_lib_dir() -> Path:
    """The directory holding Genie.dll and its dependent QNN backend DLLs.

    This is `onnxruntime_qnn`'s package directory (confirmed real in
    superpowers/deploy-local-brain-npu.md Phase 1 -- the pip wheel bundles
    Genie.dll alongside QnnHtp.dll etc, all built against QAIRT 2.48.40).
    Imported lazily: this whole module must stay importable from the base
    stdlib-only `.venv` (see pyproject.toml's `npu` extra) even when
    `onnxruntime_qnn` isn't installed -- only calling `NpuFastBrain(...)`
    requires it.
    """
    import onnxruntime_qnn

    return Path(onnxruntime_qnn.LIB_DIR_FULL_PATH)


class NpuFastBrain:
    """On-device fast brain (AI PC 3B), running for real.

    Loads Phi-3.5-mini-instruct's Genie SDK bundle
    (data/npu_model/phi-3.5-mini-instruct/) and calls it through Genie's C
    API via `ctypes` -- no subprocess, no server, no separate `onnxruntime`
    session (see the Decision section of
    superpowers/deploy-local-brain-npu.md). The dialog session is created
    once and reused across `answer()` calls, matching how a real deployed
    fast brain keeps its model resident rather than reloading per query --
    cold load takes several seconds (see
    data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md).

    A real run generated well past its first sentence before Genie's own
    length limit kicked in (see that same log's "Known issue" section) --
    this class does not rely on the model's own EOS behavior alone; it sets
    an explicit stop sequence and additionally bounds output length itself.
    """

    _MAX_NEW_TOKENS = 48
    _STOP_SEQUENCES = ["<|end|>", "<|user|>", "<|system|>"]

    def __init__(self, tier: str, signals: TierSignals, artifact_dir: Path | None = None) -> None:
        self.tier = tier
        self.signals = signals
        self._artifact_dir = artifact_dir or _NPU_ARTIFACT_DIR
        self._lib = self._load_genie()
        self._config_handle = self._create_config()
        self._dialog_handle = self._create_dialog()

    def _load_genie(self) -> ctypes.CDLL:
        lib_dir = _genie_lib_dir()
        if hasattr(os, "add_dll_directory"):  # Windows only; this brain is Windows-only anyway.
            os.add_dll_directory(str(lib_dir))
        # add_dll_directory alone is not enough here: a real run without this
        # failed inside GenieDialog_create with "Unable to find a valid
        # system interface / Resource manager not able to create
        # QnnSystemInterface" -- Genie's own internal QnnSystem.dll lookup at
        # dialog-creation time needs the dir on PATH too, matching the
        # recipe proven in
        # data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md's
        # attempt 2 (which used PATH, not add_dll_directory).
        lib_dir_str = str(lib_dir)
        path = os.environ.get("PATH", "")
        if lib_dir_str not in path.split(os.pathsep):
            os.environ["PATH"] = lib_dir_str + os.pathsep + path
        lib = ctypes.CDLL(str(lib_dir / "Genie.dll"))

        lib.GenieDialogConfig_createFromJson.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.GenieDialogConfig_createFromJson.restype = ctypes.c_int
        lib.GenieDialogConfig_free.argtypes = [ctypes.c_void_p]
        lib.GenieDialogConfig_free.restype = ctypes.c_int
        lib.GenieDialog_create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.GenieDialog_create.restype = ctypes.c_int
        lib.GenieDialog_query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            _GenieQueryCallback,
            ctypes.c_void_p,
        ]
        lib.GenieDialog_query.restype = ctypes.c_int
        lib.GenieDialog_setStopSequence.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.GenieDialog_setStopSequence.restype = ctypes.c_int
        lib.GenieDialog_reset.argtypes = [ctypes.c_void_p]
        lib.GenieDialog_reset.restype = ctypes.c_int
        lib.GenieDialog_signal.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.GenieDialog_signal.restype = ctypes.c_int
        lib.GenieDialog_free.argtypes = [ctypes.c_void_p]
        lib.GenieDialog_free.restype = ctypes.c_int
        return lib

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:  # GENIE_STATUS_SUCCESS
            raise GenieError(f"{what} failed with Genie_Status_t={status}")

    def _create_config(self) -> ctypes.c_void_p:
        """Load genie_config.json, rewriting its relative asset paths to
        absolute ones -- Genie resolves them itself and this brain's cwd
        can't be assumed to be the artifact directory."""
        raw = json.loads((self._artifact_dir / "genie_config.json").read_text())
        dialog = raw["dialog"]
        dialog["tokenizer"]["path"] = str(self._artifact_dir / dialog["tokenizer"]["path"])
        backend = dialog["engine"]["backend"]
        backend["extensions"] = str(self._artifact_dir / backend["extensions"])
        binary = dialog["engine"]["model"]["binary"]
        binary["ctx-bins"] = [str(self._artifact_dir / name) for name in binary["ctx-bins"]]

        handle = ctypes.c_void_p()
        self._check(
            self._lib.GenieDialogConfig_createFromJson(
                json.dumps(raw).encode("utf-8"), ctypes.byref(handle)
            ),
            "GenieDialogConfig_createFromJson",
        )
        return handle

    def _create_dialog(self) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(
            self._lib.GenieDialog_create(self._config_handle, ctypes.byref(handle)),
            "GenieDialog_create",
        )
        # GenieDialog_setStopSequence requires a JSON *object* with a
        # "stop-sequence" key -- confirmed against this SDK's own
        # Genie/src/Dialog.cpp (`item.key() == "stop-sequence"`), not a bare
        # array. Passing a bare array fails with Genie_Status_t=-8
        # ("Top level config is not an object") -- a real error hit and
        # logged here, not a documented assumption.
        stop_sequences = json.dumps({"stop-sequence": self._STOP_SEQUENCES}).encode("utf-8")
        self._check(
            self._lib.GenieDialog_setStopSequence(handle, stop_sequences),
            "GenieDialog_setStopSequence",
        )
        return handle

    @staticmethod
    def _render_prompt(query: str, context: str) -> str:
        system = "You are a helpful, concise assistant. Answer in one or two sentences."
        if context:
            system += f" Context: {context}"
        return f"<|system|>\n{system}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        prompt = self._render_prompt(masked_query, context)
        chunks: list[str] = []
        token_count = 0

        def _on_response(response: bytes, sentence_code: int, _user_data: int) -> None:
            nonlocal token_count
            # A ctypes callback must never let a Python exception escape into
            # the calling C code -- that terminates the process.
            try:
                if response:
                    chunks.append(response.decode("utf-8", errors="replace"))
                    token_count += 1
                if (
                    token_count >= self._MAX_NEW_TOKENS
                    and sentence_code not in (_GENIE_SENTENCE_END, _GENIE_SENTENCE_ABORT)
                ):
                    self._lib.GenieDialog_signal(self._dialog_handle, _GENIE_DIALOG_ACTION_ABORT)
            except Exception:  # noqa: BLE001 -- must not propagate across the C boundary
                pass

        callback = _GenieQueryCallback(_on_response)
        self._check(self._lib.GenieDialog_reset(self._dialog_handle), "GenieDialog_reset")

        start = time.perf_counter()
        self._check(
            self._lib.GenieDialog_query(
                self._dialog_handle,
                prompt.encode("utf-8"),
                _GENIE_SENTENCE_COMPLETE,
                callback,
                None,
            ),
            "GenieDialog_query",
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return BrainResponse(text="".join(chunks).strip(), latency_ms=latency_ms, cost_usd=0.0)

    def close(self) -> None:
        self._lib.GenieDialog_free(self._dialog_handle)
        self._lib.GenieDialogConfig_free(self._config_handle)

    def __del__(self) -> None:
        # Best-effort: interpreter shutdown may have already torn down ctypes
        # state, so swallow errors here rather than raise from a destructor.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class GpuBrainError(RuntimeError):
    """`llama-server` failed to start, or returned a non-OK HTTP response."""


#: llama.cpp OpenCL/Adreno build. Not vendored (see .gitignore) -- override with
#: TWO_BRAIN_LLAMA_BIN. Verified build 10291 (803b7fcae); see
#: data/vlm_gpu_model/qwen3-vl-4b-instruct/_real_inference_smoke_log.md.
_LLAMA_BIN_DIR = DATA_DIR.parent / ".llama-cpp-opencl" / "extracted"

#: Default weights: the same model NpuFastBrain runs, as a GGUF, so the two
#: are a like-for-like comparison of device rather than of model.
_GPU_MODEL_PATH = DATA_DIR / "npu_model" / "phi-3.5-mini-instruct" / "gguf" / "Phi-3.5-mini-instruct-Q4_0.gguf"

#: Recommended vision defaults, chosen by measurement rather than by size.
#: `data/vlm_gpu_model/_eval/RESULTS.md` scored four variants on 7
#: objectively-checkable items with deterministic decoding:
#:
#:     8B Q4_0  7/7   <- this default
#:     4B Q8_0  6/7
#:     4B BF16  6/7   (full precision buys nothing over Q8_0, at 1.9x the size)
#:     4B Q4_0  5/7
#:
#: The 8B at 4-bit beats the 4B at *full precision*, so parameters matter more
#: than bit-width here. It costs ~1.6x the latency and cannot keep its vision
#: encoder on the GPU -- its vision tower is head_dim 72 and llama.cpp's OpenCL
#: flash-attention kernels cover only 64/128, so `mmproj_offload=False` is
#: mandatory or the process segfaults.
_VLM_DIR = DATA_DIR / "vlm_gpu_model"
_VLM_MODEL_PATH = _VLM_DIR / "qwen3-vl-8b-instruct" / "raw" / "Qwen3-VL-8B-Instruct-Q4_0.gguf"
_VLM_MMPROJ_PATH = _VLM_DIR / "qwen3-vl-8b-instruct" / "raw" / "mmproj-F16.gguf"
#: The speed-oriented alternative: one point behind on the eval, but keeps the
#: vision encoder on the GPU (head_dim 64) and is ~39% faster.
_VLM_FAST_MODEL_PATH = _VLM_DIR / "qwen3-vl-4b-instruct" / "raw" / "Qwen3-VL-4B-Instruct-Q8_0.gguf"
_VLM_FAST_MMPROJ_PATH = _VLM_DIR / "qwen3-vl-4b-instruct" / "raw" / "mmproj-Qwen3VL-4B-Instruct-f16.gguf"


class GpuLocalBrain:
    """On-device fast brain running on the Adreno GPU, running for real.

    One class covers both LLM and VLM weights on purpose. The LLM/VLM split is
    a *capability* difference, not a backend one: both are llama.cpp GGUFs on
    the same `ggml-opencl` backend, same device, same context sizing. The only
    difference is whether a multimodal projector is loaded, which is a
    constructor argument (`mmproj_path`) rather than a subclass. A VLM also
    answers text-only queries perfectly well, so a single vision-capable
    instance can serve both roles.

    **`answer()` deliberately takes text-only *input*, even when a projector is
    loaded.** To be clear about which side is constrained: these models are
    image-text-to-text (Qwen3-VL's own GGUF metadata tags it exactly that), so
    text-only *output* is inherent, not a limitation -- `BrainResponse.text`
    stays the right shape no matter what happens with images later. The
    projector is an input-side encoder (`mmproj loaded: vision=true`).

    It is the *input* that is withheld. `Brain.answer` has no image parameter,
    and adding one would force a change in `router.py` -- the signal this
    file's own guidance names for a seam drawn in the wrong place. More
    importantly, image input is an unsolved privacy question here: `PIIGuard`
    masks *text*, so a face, a document, or EXIF GPS in an image would cross to
    the deep brain untouched while `assert_masked_token_invariant` still
    passed, because it only inspects text. Routing images needs that decision
    made first, not an API shape that quietly pre-empts it. Loading the
    projector now simply means the instance is ready when it is.

    Unlike `NpuFastBrain`, which drives Genie in-process through `ctypes`, this
    talks to a `llama-server` child process over loopback HTTP. That is a real
    departure from the AI-PC tier's no-HTTP-hop precedent, taken deliberately:
    llama.cpp's C API would need a hand-written decode loop, sampler, and chat
    templating -- the bespoke-FFI surface that already produced two logged bugs
    in `NpuFastBrain` -- whereas `llama-server` implements all of it, speaks an
    OpenAI-shaped API, and gives multi-slot serving (`-np N`) that the Genie
    C API has no equivalent for. The hop is loopback on the same device, not a
    network call. Only stdlib is used to talk to it, so this module stays
    importable from the base venv.

    The server is started once and reused across `answer()` calls -- cold load
    is ~3 s (see data/npu_model/phi-3.5-mini-instruct/_real_geniex_hybrid_log.md).
    """

    _MAX_NEW_TOKENS = 48
    _N_CTX = 4096
    _STARTUP_TIMEOUT_S = 120.0

    def __init__(
        self,
        tier: str,
        signals: TierSignals,
        model_path: Path | None = None,
        mmproj_path: Path | None = None,
        bin_dir: Path | None = None,
        mmproj_offload: bool = True,
        log_path: Path | None = None,
    ) -> None:
        self.tier = tier
        self.signals = signals
        self._model_path = Path(os.environ.get("TWO_BRAIN_GPU_MODEL") or model_path or _GPU_MODEL_PATH)
        env_mmproj = os.environ.get("TWO_BRAIN_GPU_MMPROJ")
        self._mmproj_path = Path(env_mmproj) if env_mmproj else mmproj_path
        self._bin_dir = Path(os.environ.get("TWO_BRAIN_LLAMA_BIN") or bin_dir or _LLAMA_BIN_DIR)
        # The 8B VLM segfaults with its vision encoder on GPU (OpenCL has no
        # flash-attention kernel at its head_dim 72) -- pass False for that one.
        # See data/vlm_gpu_model/qwen3-vl-8b-instruct/_real_inference_smoke_log.md.
        self._mmproj_offload = mmproj_offload
        # llama-server hides device/offload lines at its default verbosity, so
        # with logs discarded a silent CPU fallback would be undetectable --
        # exactly the failure mode this project rejects elsewhere. Setting
        # log_path (or TWO_BRAIN_GPU_LOG) captures the server's own log at -v,
        # where `using device GPUOpenCL` / `offloaded N/N layers to GPU` /
        # `ggml_opencl: OpenCL driver:` are the lines that prove placement.
        env_log = os.environ.get("TWO_BRAIN_GPU_LOG")
        self._log_path = Path(env_log) if env_log else log_path
        self._log_file: object | None = None
        self._proc: object | None = None
        self._base_url: str | None = None
        self._start_server()

    #: Substrings that, in a captured server log, prove GPU placement.
    GPU_PLACEMENT_MARKERS = ("using device GPUOpenCL", "layers to GPU", "ggml_opencl:")

    def verify_gpu_placement(self) -> bool:
        """True if the captured server log shows real GPU offload.

        Requires `log_path` to have been set -- returns False otherwise, since
        absence of evidence is not evidence of placement.
        """
        if self._log_path is None or not self._log_path.exists():
            return False
        text = self._log_path.read_text(encoding="utf-8", errors="replace")
        return any(marker in text for marker in self.GPU_PLACEMENT_MARKERS)

    @classmethod
    def for_vision(cls, tier: str, signals: TierSignals, prefer: str = "quality", **kw) -> "GpuLocalBrain":
        """Construct with the measured-best vision weights.

        `prefer="quality"` -> Qwen3-VL-8B-Instruct-Q4_0 (7/7 on the eval), with
        `mmproj_offload=False` because its vision encoder cannot run on the GPU.
        `prefer="speed"` -> Qwen3-VL-4B-Instruct-Q8_0 (6/7), vision encoder on
        the GPU, ~39% faster.

        Both are still text-in/text-out today: `answer()` does not accept an
        image, for the protocol and privacy reasons in this class's docstring.
        Loading the projector means the instance is ready when that lands, and
        it makes the recommended weights a default rather than folklore.
        See data/vlm_gpu_model/_eval/RESULTS.md.
        """
        if prefer not in ("quality", "speed"):
            raise ValueError(f"prefer must be 'quality' or 'speed', got {prefer!r}")
        if prefer == "quality":
            kw.setdefault("model_path", _VLM_MODEL_PATH)
            kw.setdefault("mmproj_path", _VLM_MMPROJ_PATH)
            kw.setdefault("mmproj_offload", False)  # mandatory for the 8B; segfaults otherwise
        else:
            kw.setdefault("model_path", _VLM_FAST_MODEL_PATH)
            kw.setdefault("mmproj_path", _VLM_FAST_MMPROJ_PATH)
        return cls(tier, signals, **kw)

    def _server_exe(self) -> Path:
        exe = self._bin_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if not exe.exists():
            raise GpuBrainError(
                f"llama-server not found at {exe}. The OpenCL build is not vendored -- "
                "set TWO_BRAIN_LLAMA_BIN to the directory holding it."
            )
        return exe

    def _server_env(self) -> dict[str, str]:
        """Environment for the child, with the Adreno OpenCL ICD wired up.

        The ICD is not registered system-wide on this machine (registering it
        needs admin rights), so the Khronos loader is pointed at the driver's
        own ICD via OCL_ICD_FILENAMES. Without this, device enumeration returns
        "Available devices: (none)" and llama.cpp silently falls back to CPU --
        which would make this class quietly *not* a GPU brain.
        """
        env = dict(os.environ)
        icd = self._bin_dir / "OpenCL_adreno.dll"
        if icd.exists() and "OCL_ICD_FILENAMES" not in env:
            env["OCL_ICD_FILENAMES"] = str(icd)
        env["PATH"] = f"{self._bin_dir}{os.pathsep}{env.get('PATH', '')}"
        return env

    def _free_port(self) -> int:
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def _start_server(self) -> None:
        import subprocess

        if not self._model_path.exists():
            raise GpuBrainError(
                f"model not found at {self._model_path} -- set TWO_BRAIN_GPU_MODEL. "
                "GGUF weights are not vendored (see .gitignore)."
            )
        port = self._free_port()
        cmd = [
            str(self._server_exe()),
            "-m", str(self._model_path),
            # -c is not optional: llama.cpp's auto-fit sizes context from the
            # model's train length (262144 here), overcommits the KV cache on
            # this device, and then fails a ~300 MB compute-buffer allocation.
            "-c", str(self._N_CTX),
            "-ngl", "99",
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        if self._mmproj_path is not None:
            cmd += ["--mmproj", str(self._mmproj_path)]
            if not self._mmproj_offload:
                cmd.append("--no-mmproj-offload")
        if self._log_path is not None:
            cmd.append("-v")  # device/offload lines only appear above default verbosity
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("wb")
            sink = self._log_file
        else:
            sink = subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd,
            env=self._server_env(),
            stdout=sink,
            stderr=sink,
        )
        self._base_url = f"http://127.0.0.1:{port}"
        self._await_ready()

    def _await_ready(self) -> None:
        import urllib.error
        import urllib.request

        deadline = time.perf_counter() + self._STARTUP_TIMEOUT_S
        while time.perf_counter() < deadline:
            if self._proc is not None and self._proc.poll() is not None:  # type: ignore[attr-defined]
                raise GpuBrainError(
                    f"llama-server exited with code {self._proc.returncode} during startup"  # type: ignore[attr-defined]
                )
            try:
                with urllib.request.urlopen(f"{self._base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.25)
        self.close()
        raise GpuBrainError(f"llama-server did not become ready within {self._STARTUP_TIMEOUT_S:.0f}s")

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        import urllib.error
        import urllib.request

        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": masked_query})
        payload = json.dumps(
            {
                "messages": messages,
                "max_tokens": self._MAX_NEW_TOKENS,
                "temperature": 0.7,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - server-side failure
            raise GpuBrainError(f"llama-server returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        text = body["choices"][0]["message"]["content"]
        return BrainResponse(text=text.strip(), latency_ms=latency_ms, cost_usd=0.0)

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.poll() is None:  # type: ignore[attr-defined]
            proc.terminate()  # type: ignore[attr-defined]
            try:
                proc.wait(timeout=10)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 -- fall through to kill
                proc.kill()  # type: ignore[attr-defined]
        log_file, self._log_file = self._log_file, None
        if log_file is not None:
            try:
                log_file.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    def __del__(self) -> None:
        # Best-effort, same rationale as NpuFastBrain: never raise from a
        # destructor during interpreter shutdown. Leaking the child process
        # would be worse than a swallowed error here.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
