"""The two brains, behind one interface.

The cloud brain is still a stub: the Cloud AI 100 has no plumbing in
QUAD-Client at all (gap #4), so it returns a *labeled* stub answer with
latency/cost estimated from `profile_workload`'s envelope shape.

Neither local tier is a stub any more:

- **AI PC (`pc_3b`)** -- `NpuFastBrain` runs a real Phi-3.5-mini-instruct
  artifact on this machine's Hexagon NPU via Qualcomm's Genie SDK
  (`Genie.dll`, called through `ctypes` -- not ONNX Runtime GenAI; see
  `data/npu_model/phi-3.5-mini-instruct/_real_download_log.md` for why).
  In-process by design -- `docs/npu-deployment.md`.
- **Mobile (`mobile_1b`)** -- `PhoneFastBrain` talks to the on-device Genie
  server built in `src/phone_brain/` over the OpenAI-shaped contract in
  `src/phone_brain/L_INTERFACE_CONTRACT.md`. Off-process by necessity: the
  model runs on a physically separate device.

`LocalFastBrain` remains as the labeled-stub fallback both tiers use when
their real runtime isn't wired up, so the package still runs anywhere with
no SDK and no hardware.

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

from two_brain_router.signals.confidence import SELF_REPORT_SUFFIX, parse_self_reported
from two_brain_router.signals.loader import DATA_DIR, TierSignals


@dataclass
class BrainResponse:
    """What any brain must return: the text, plus what it cost to get it."""

    text: str
    latency_ms: float
    cost_usd: float = 0.0
    #: The brain's own confidence in this answer, `[0.0, 1.0]`, or None when
    #: this brain emits no such signal (every stub, and NpuFastBrain). See
    #: signals/confidence.py for why None and 0.0 mean different things.
    confidence: float | None = None
    #: Set when the brain could not be reached or misbehaved. The router
    #: surfaces this in RouteDecision.notes rather than raising -- per
    #: L_INTERFACE_CONTRACT.md, a failed fast brain is an escalate signal, not
    #: a user-visible error.
    error: str | None = None


class Brain(Protocol):
    """Implement this to plug a real runtime in behind the router."""

    #: Whether `answer()` populates `BrainResponse.confidence`.
    #:
    #: This flips the *order* of the router's decision, so it is part of the
    #: contract rather than an implementation detail. When False the router
    #: scores difficulty first and only calls this brain if it decides to stay
    #: local. When True the brain's own confidence *is* the difficulty signal
    #: and arrives with the answer, so the router must call it before deciding
    #: -- see router.py's `route()` and docs/ORCHESTRATOR.md.
    reports_confidence: bool

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

    reports_confidence = False

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

    reports_confidence = False

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

    #: This model self-rates: it answers and reports its own confidence in one
    #: call, so the router uses that number as the difficulty signal instead of
    #: `signals/difficulty.py`'s surface-feature heuristic (Shape B --
    #: docs/ORCHESTRATOR.md). Same mechanism as `PhoneFastBrain`, so both real
    #: brains report the same way and `routing/policy.py` stays untouched.
    #:
    #: Honest caveat: this is a *prompted self-report*, not a logprob. Genie
    #: still exposes no token probabilities through the C API used here, so the
    #: number is the model's own claim about itself. Measured discrimination on
    #: this artifact is weak -- see the calibration note in
    #: data/npu_model/phi-3.5-mini-instruct/_real_inference_smoke_log.md
    #: (Attempt 5); it separates "can't answer" from "can", not easy from hard.
    reports_confidence = True

    #: Raised from 48 when the self-report was added: the answer *plus* the
    #: `CONFIDENCE:` line has to fit, and a cap that truncates the line away
    #: silently turns every query into an escalation. This is a runaway guard,
    #: not a target -- measured answers land well under it (mean ~1.8s / query,
    #: Attempt 5), because `_STOP_SEQUENCES` ends generation first.
    _MAX_NEW_TOKENS = 96

    #: `"\n\n"` is load-bearing, not cosmetic. Left to itself this model emits
    #: the answer, the `CONFIDENCE:` line, a blank line, and then paragraphs of
    #: unasked-for rationale until the token cap -- which tripled latency
    #: (~5.7s vs ~1.8s per query) and left truncated mid-word prose in the
    #: answer. `_render_prompt`'s system message forbids blank lines *inside*
    #: the reply, so a blank line can only occur after the confidence number,
    #: which makes it a safe place to stop. If the model disobeys and puts one
    #: earlier, generation stops before the number, nothing parses, and the
    #: query escalates -- the safe direction.
    _STOP_SEQUENCES = ["<|end|>", "<|user|>", "<|system|>", "\n\n"]

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
        """Raise on a Genie *error*, not on a Genie *warning*.

        GenieCommon.h splits the status space by sign: 0 is
        GENIE_STATUS_SUCCESS, negatives are errors
        (GENIE_STATUS_ERROR_GENERAL = -1 ... GENIE_STATUS_ERROR_BOUND_HANDLE
        = -14), and positives are warnings -- GENIE_STATUS_WARNING_ABORTED = 1,
        _BOUND_HANDLE = 2, _PAUSED = 3.

        Treating `status != 0` as fatal was a real bug: `answer()` signals
        GENIE_DIALOG_ACTION_ABORT itself once `_MAX_NEW_TOKENS` is hit, and
        Genie then returns WARNING_ABORTED(1) from `GenieDialog_query` -- so
        the token cap this class relies on to bound output length crashed the
        very call it was meant to truncate. It went unnoticed because every
        earlier recorded run stopped on the `<|end|>` stop sequence well before
        the cap (see data/npu_model/phi-3.5-mini-instruct/
        _real_inference_smoke_log.md, Attempt 5).
        """
        if status < 0:
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
        """Phi-3.5's chat template, with the self-report asked for in the user
        turn.

        The strict two-line instruction is what makes `_STOP_SEQUENCES`'
        `"\\n\\n"` safe (see there) and is worth keeping verbatim -- looser
        phrasings were measured and lost. Asking the model to lead with the
        confidence instead ("output CONFIDENCE first, then answer") was faster
        still but sometimes returned the number *and no answer at all*, so it
        was rejected: an empty answer is worse than a slow one.

        `SELF_REPORT_SUFFIX` goes inside the `<|user|>` block, before its
        `<|end|>`, so the stop sequence cannot fire before the model has read
        the instruction. It is imported rather than restated so this brain and
        `PhoneFastBrain` ask for the number in identical words.
        """
        system = (
            "You are a helpful, concise assistant. Reply with exactly two lines "
            "and nothing else: line 1 is your answer in one or two sentences; "
            "line 2 is 'CONFIDENCE: <number>'. Do not use blank lines. Do not "
            "explain the number."
        )
        if context:
            system += f" Context: {context}"
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{query}{SELF_REPORT_SUFFIX}<|end|>\n"
            f"<|assistant|>\n"
        )

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

        # Same tail as PhoneFastBrain.answer, deliberately: both real brains
        # self-rate, so both must report the number (and the absence of one) the
        # same way for the router's Shape B path to treat them alike.
        answer, confidence = parse_self_reported("".join(chunks).strip())
        return BrainResponse(
            text=answer,
            latency_ms=latency_ms,
            cost_usd=0.0,
            confidence=confidence,
            error=(
                None
                if confidence is not None
                else "no parseable CONFIDENCE line in the response"
            ),
        )

    def close(self) -> None:
        """Release the Genie dialog + config. Idempotent.

        The handles are cleared as they are freed because `__del__` also calls
        this: without the guard, an explicit `close()` followed by garbage
        collection frees the same native handles twice. That is not a harmless
        double-free -- a real run showed it corrupting Genie's internal state so
        that the *next* session created in the same process failed with
        GENIE_STATUS_ERROR_INVALID_HANDLE(-5) on `GenieDialog_reset`, even
        though the two sessions never overlapped. `__del__` swallowing
        exceptions did not (and could not) catch it, since freeing a stale
        handle returns a status code rather than raising.
        """
        if self._dialog_handle is not None:
            self._lib.GenieDialog_free(self._dialog_handle)
            self._dialog_handle = None
        if self._config_handle is not None:
            self._lib.GenieDialogConfig_free(self._config_handle)
            self._config_handle = None

    def __del__(self) -> None:
        # Best-effort: interpreter shutdown may have already torn down ctypes
        # state, so swallow errors here rather than raise from a destructor.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


#: Hosts that mean "this device". The phone is reached over
#: `adb reverse tcp:8000 tcp:8000`, which is precisely what makes it appear on
#: loopback -- so loopback is the honest test for "the masked query is not
#: traversing a network", not a proxy for it.
_ON_DEVICE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class RemoteBrainRefused(ValueError):
    """A fast brain was pointed at a host that is not on-device."""


class PhoneFastBrain:
    """Fast brain served over an OpenAI-shaped HTTP endpoint (Mobile tier).

    Speaks the contract in `src/phone_brain/L_INTERFACE_CONTRACT.md`
    (`POST /v1/chat/completions`), so it is identical against
    `mock_phone_brain_server.py` and against the real Genie/GenieX server on
    the Galaxy S25 -- only the base URL changes. Deliberately **not**
    phone-specific: any OpenAI-shaped endpoint works, which is why a future
    hosted AI-PC or cloud tier can reuse this class rather than copy it.

    Two things make this different from the other brains here:

    1. **It self-rates.** `reports_confidence = True`, so the router asks it
       *before* deciding local-vs-cloud and uses the returned confidence as
       the difficulty signal (one inference call, not two). See
       signals/confidence.py.
    2. **It is off-process.** Unlike `NpuFastBrain`, which is in-process by
       design (docs/npu-deployment.md), the mobile model genuinely runs on a
       separate device. That makes the base URL a privacy-relevant input, so
       it is checked -- see `_assert_on_device`.

    Latency is *measured*, not estimated from `data/profile_workload/`: this
    makes a real call, so there is a real number to report.
    """

    reports_confidence = True

    #: `adb reverse tcp:8000 tcp:8000` puts the phone here (L contract).
    #:
    #: `127.0.0.1`, not `localhost`, and this is measured rather than
    #: stylistic: on this Windows host `localhost` resolves to `::1` first,
    #: the server binds IPv4 only, and the failed IPv6 attempt costs **~2s per
    #: call** before the fallback succeeds (measured 2778-3117ms via
    #: `localhost` vs. 742-1153ms via `127.0.0.1`, same server, same prompt).
    #: For most clients that is an annoyance; here it is a correctness bug,
    #: because `RoutePolicy.local_latency_budget_ms` is 3000ms and the router
    #: decides local-vs-cloud on this exact number -- a phantom 2s would
    #: escalate queries the phone could comfortably have answered.
    DEFAULT_BASE_URL = "http://127.0.0.1:8000"
    DEFAULT_MODEL = "llama-3.2-3b-instruct"

    #: Matches confidence_estimator.py's request shape exactly -- the L
    #: contract pins these four fields.
    _MAX_TOKENS = 256
    _TEMPERATURE = 0.2
    _TIMEOUT_S = 120.0

    def __init__(
        self,
        tier: str,
        signals: TierSignals,
        base_url: str | None = None,
        model: str | None = None,
        allow_remote: bool = False,
        timeout_s: float | None = None,
    ) -> None:
        self.tier = tier
        self.signals = signals
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout_s = timeout_s if timeout_s is not None else self._TIMEOUT_S
        self._assert_on_device(self.base_url, allow_remote)

    @staticmethod
    def _assert_on_device(base_url: str, allow_remote: bool) -> None:
        """Refuse a non-loopback endpoint unless explicitly allowed.

        Defense in depth for `docs/PHONE_BRAIN.md` reconciliation point 1. The
        router only ever hands this class *masked* text, so this is not the
        thing standing between the user and a leak -- but `--base-url` is a
        plain string, and the difference between "the model runs on my phone"
        and "the model runs on someone's server" is exactly one typo. A query
        that leaves the device is a different privacy posture than the one this
        project advertises, so it takes a deliberate opt-in rather than a
        silent default.
        """
        import urllib.parse

        host = urllib.parse.urlsplit(base_url).hostname
        if allow_remote or host in _ON_DEVICE_HOSTS:
            return
        raise RemoteBrainRefused(
            f"refusing to send queries to non-on-device host {host!r}: the mobile "
            f"fast brain is reached over loopback (adb reverse). Pass "
            f"allow_remote=True (or set TWO_BRAIN_PHONE_ALLOW_REMOTE=1) if the "
            f"model really is meant to run off-device."
        )

    def _post_chat_completion(self, prompt: str) -> dict:
        """One `POST /v1/chat/completions`, stdlib only.

        `urllib` rather than `requests` on purpose: this package declares zero
        runtime dependencies (pyproject.toml), and the phone brain's own
        tooling is stdlib-only for the same reason.
        """
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._MAX_TOKENS,
            "temperature": self._TEMPERATURE,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        """Answer `masked_query`, and report how sure the model is about it.

        Never raises on a transport failure. Per L_INTERFACE_CONTRACT.md's
        error-handling section, a timeout or non-200 means "L failed" and
        should escalate rather than block the user -- so that path returns
        `confidence=0.0` (a definite escalate once inverted to difficulty) with
        `error` set for the audit trail, instead of propagating an exception
        the router would have to special-case.
        """
        prompt = f"{context}\n\n{masked_query}" if context else masked_query
        start = time.perf_counter()
        try:
            body = self._post_chat_completion(prompt + SELF_REPORT_SUFFIX)
            raw_text = body["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 -- any failure is an escalate signal
            return BrainResponse(
                text="",
                latency_ms=(time.perf_counter() - start) * 1000,
                cost_usd=0.0,
                confidence=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - start) * 1000

        answer, confidence = parse_self_reported(raw_text)
        return BrainResponse(
            text=answer,
            latency_ms=latency_ms,
            cost_usd=0.0,
            confidence=confidence,
            error=(
                None
                if confidence is not None
                else "no parseable CONFIDENCE line in the response"
            ),
        )
