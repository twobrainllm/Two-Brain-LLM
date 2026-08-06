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
