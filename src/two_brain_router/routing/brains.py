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
from two_brain_router.signals.structured import (
    STRUCTURED_SUFFIX,
    STRUCTURED_SYSTEM_PROMPT,
    parse_structured,
)

#: Master switch for the structured (Shape C) reply format on the AI-PC
#: brains. On by default -- but those brains are themselves opt-in
#: (`TWO_BRAIN_NPU_BRAIN` / `TWO_BRAIN_GPU_BRAIN`), so nothing about the
#: default stdlib-only path changes. Set `TWO_BRAIN_STRUCTURED=0` to get the
#: older bare `CONFIDENCE:` self-report back (Shape B) without giving up the
#: real brain -- useful for isolating "is the model bad at JSON?" from "is the
#: model bad at this question?".
_STRUCTURED_ENV_VAR = "TWO_BRAIN_STRUCTURED"


def _structured_default() -> bool:
    return os.environ.get(_STRUCTURED_ENV_VAR, "1") != "0"


@dataclass
class BrainResponse:
    """What any brain must return: the text, plus what it cost to get it."""

    text: str
    latency_ms: float
    cost_usd: float = 0.0
    #: The brain's own confidence in this answer, `[0.0, 1.0]`, or None when
    #: this brain emits no such signal (LocalFastBrain, CloudDeepBrain -- the
    #: stubs; both real brains, NpuFastBrain included, self-rate). See
    #: signals/confidence.py for why None and 0.0 mean different things.
    confidence: float | None = None
    #: Set when the brain could not be reached or misbehaved. The router
    #: surfaces this in RouteDecision.notes rather than raising -- per
    #: L_INTERFACE_CONTRACT.md, a failed fast brain is an escalate signal, not
    #: a user-visible error.
    error: str | None = None
    #: The part of the query this brain says it *cannot* answer, in its own
    #: words -- empty when it answered fully, or when the brain doesn't report
    #: gaps at all (`Brain.reports_gaps`). This is what the deep brain's
    #: follow-up call is about; `text` stays the part that was answered. See
    #: signals/structured.py and docs/ORCHESTRATOR.md "Shape C".
    unknown: str = ""
    #: A "masked" string the model volunteered, if any. Recorded for the audit
    #: trail and never routed on -- masking is `privacy/guard.py`'s job, and it
    #: happens at the cloud boundary rather than being delegated to a model that
    #: was trusted with the raw text. See signals/structured.py.
    model_masked_output: str = ""


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

    #: Whether `answer()` also populates `BrainResponse.unknown` -- i.e. whether
    #: this brain can say *which part* it couldn't do, not just how sure it is
    #: overall. Strictly stronger than `reports_confidence`: a brain that
    #: reports gaps necessarily self-rates, so `reports_gaps` implies
    #: `reports_confidence` and never the reverse.
    #:
    #: This is what lets the router split one query across two brains instead of
    #: picking one (Shape C -- docs/ORCHESTRATOR.md): the local answer is kept
    #: and only the named gap is sent on. Read via `getattr(brain,
    #: "reports_gaps", False)` in the router, so a third-party brain predating
    #: this field is still valid.
    reports_gaps: bool

    #: Whether this brain is **inside the trust boundary** -- i.e. whether it
    #: may be handed the user's query exactly as typed, PII and all.
    #:
    #: This is the flag that decides what `answer()`'s `query` argument
    #: actually contains, so it is the most safety-critical thing in this
    #: Protocol. True only for a model executing on *this* machine, in this
    #: process or in a child process we started: `NpuFastBrain` (in-process
    #: `ctypes`), `GpuLocalBrain` (a `llama-server` child on loopback), and the
    #: in-process stub. False for anything reached across a boundary --
    #: `PhoneFastBrain` (a physically separate device) and both cloud brains --
    #: which receive masked text only.
    #:
    #: **Defaults to False everywhere it is read.** The router uses
    #: `getattr(brain, "trusted_with_raw_pii", False)`, so a brain that forgets
    #: to declare it gets masked input. Forgetting to opt *in* costs answer
    #: quality; forgetting to opt *out* would leak, so the default is the one
    #: that fails safe.
    trusted_with_raw_pii: bool

    def answer(self, query: str, context: str = "") -> BrainResponse: ...


class VisionBrain(Brain, Protocol):
    """A `Brain` that can also see.

    Kept separate from `Brain` on purpose: the cloud deep brain is a text-only
    LLM and must never be asked for either of these. The router checks for this
    capability rather than assuming it, so a text-only fast brain stays valid.

    `describe_image` is the load-bearing one. Images cannot cross the device
    boundary -- the cloud tier has no VLM at all (see
    data/cloud_ai100/_real_endpoint_log.md) -- so escalating an image-bearing
    query means converting the image to *words* locally, masking those words
    like any other text, and sending only that.
    """

    def answer(self, query: str, context: str = "", image: Path | None = None) -> BrainResponse: ...

    def describe_image(self, image: Path, question: str = "") -> BrainResponse: ...


def _estimate_tokens(query: str) -> int:
    """Token count the latency/cost estimates are driven off."""
    return max(len(query.split()) * 2, 16)


def _to_brain_response(raw_text: str, latency_ms: float, structured: bool) -> BrainResponse:
    """Turn a self-rating brain's raw completion into a `BrainResponse`.

    Shared by `NpuFastBrain` and `GpuLocalBrain` on purpose: both real AI-PC
    brains must report a number, a gap, and *the absence of either* in exactly
    the same way, or the router's Shape B/C paths would treat two backends of
    the same tier differently for reasons that have nothing to do with the
    query. `PhoneFastBrain` keeps its own tail because it is Shape B only.

    `error` is set whenever no confidence came back. That is not "the call
    failed" -- the text is still returned and still usable -- it is the audit
    trail for a formatting failure the router will read as maximally uncertain.
    """
    parsed = parse_structured(raw_text) if structured else None
    if parsed is None:
        answer, confidence = parse_self_reported(raw_text)
        unknown = ""
        model_masked = ""
        note = "no parseable CONFIDENCE line in the response"
    else:
        answer, confidence = parsed.solution, parsed.confidence
        unknown = parsed.unknown
        model_masked = parsed.model_masked_output
        note = (
            "no parseable JSON object in the response"
            if parsed.source == "raw"
            else "fell back to a bare CONFIDENCE: line -- no JSON object in the response"
        )
    return BrainResponse(
        text=answer,
        latency_ms=latency_ms,
        cost_usd=0.0,
        confidence=confidence,
        unknown=unknown,
        model_masked_output=model_masked,
        # A structured reply that degraded to `source="self_report"` still
        # produced a usable number, so it is not an error -- only the total
        # absence of a signal is.
        error=None if confidence is not None else note,
    )


class LocalFastBrain:
    """On-device fast brain (Mobile 1B / AI PC 3B).

    Returns a labeled stub -- no compiled artifact exists to actually run (see
    data/convert_model/_real_attempts_log.md). Latency is estimated from
    profile_workload's real envelope shape (data/profile_workload/<tier>.json).
    """

    reports_confidence = False
    reports_gaps = False
    #: In-process, so it is inside the boundary like the real local brains --
    #: a stub that received *different* input from the thing it stands in for
    #: would make the default path a bad rehearsal for the real one.
    trusted_with_raw_pii = True

    def __init__(self, tier: str, signals: TierSignals) -> None:
        self.tier = tier
        self.signals = signals

    def answer(self, query: str, context: str = "") -> BrainResponse:
        latency_ms = self.signals.profile["latency_ms"]
        n_tokens = _estimate_tokens(query)
        est_latency = latency_ms["ttft_mean"] + n_tokens * latency_ms["per_token_mean"]
        return BrainResponse(
            text=f"[local:{self.tier} mock fast-brain response to: {query!r}]",
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
    reports_gaps = False
    #: The boundary itself. Never raw text, even as a stub -- if this were ever
    #: True the whole sample would be pointless.
    trusted_with_raw_pii = False

    def __init__(self, signals: TierSignals) -> None:
        self.signals = signals

    def answer(self, query: str, context: str = "") -> BrainResponse:
        prof = self.signals.profile
        latency_ms = prof["latency_ms"]
        n_tokens = _estimate_tokens(query)
        est_latency = (
            latency_ms["network_rtt_mean"]
            + latency_ms["ttft_mean"]
            + n_tokens * latency_ms["per_token_mean"]
        )
        cost = (n_tokens / 1000.0) * prof["token_cost_usd_per_1k"]
        return BrainResponse(
            text=(
                f"[cloud:ai100 mock deep-brain response to: {query!r} "
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

    #: The deep brain is the *destination* of a routing decision, never an
    #: input to one. It is asked to answer, not to rate itself or to hand work
    #: back -- there is nothing above it to escalate to.
    reports_confidence = False
    reports_gaps = False
    #: This class is the reason the boundary exists.
    trusted_with_raw_pii = False

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

    def answer(self, query: str, context: str = "") -> BrainResponse:
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": query})

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

    #: Inside the boundary: this runs *in this process*, through `ctypes` into
    #: `Genie.dll`, on this machine's own NPU. Nothing it is given is
    #: transmitted anywhere. So it gets the query as the user typed it --
    #: masking it here would only degrade the answer (a model asked to draft a
    #: reply to `[PII_EMAIL_1]` writes a worse reply than one that can see the
    #: address) while protecting nothing.
    trusted_with_raw_pii = True

    #: Raised from 48 when the self-report was added: the answer *plus* the
    #: `CONFIDENCE:` line has to fit, and a cap that truncates the line away
    #: silently turns every query into an escalation. This is a runaway guard,
    #: not a target -- measured answers land well under it (mean ~1.8s / query,
    #: Attempt 5), because `_STOP_SEQUENCES` ends generation first.
    _MAX_NEW_TOKENS = 96

    #: Structured mode has strictly more to emit than `CONFIDENCE:` mode -- the
    #: JSON scaffolding, plus an `unknown` field that is *prose*, not a number
    #: -- so it gets its own, larger cap. The same "a cap that truncates the
    #: format away turns every query into an escalation" reasoning applies, only
    #: harder: a JSON object cut off mid-string doesn't degrade, it fails to
    #: parse entirely. `"\n\n"` still ends generation well before this in
    #: practice, because single-line JSON is what STRUCTURED_SUFFIX asks for.
    _STRUCTURED_MAX_NEW_TOKENS = 320

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

    def __init__(
        self,
        tier: str,
        signals: TierSignals,
        artifact_dir: Path | None = None,
        structured: bool | None = None,
    ) -> None:
        self.tier = tier
        self.signals = signals
        #: On unless TWO_BRAIN_STRUCTURED=0. `reports_gaps` tracks it exactly:
        #: without the JSON format there is no `unknown` field to report, so
        #: claiming the capability would make the router split work that never
        #: comes back split.
        self._structured = _structured_default() if structured is None else structured
        self.reports_gaps = self._structured
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

    def _render_prompt(self, query: str, context: str) -> str:
        """Phi-3.5's chat template, asking for whichever reply format is active.

        Structured mode (the default) swaps the two-line instruction below for
        `STRUCTURED_SYSTEM_PROMPT` + `STRUCTURED_SUFFIX`. The *shape* of the
        instruction is identical -- a system message that forbids blank lines,
        plus a user-turn suffix pinning the exact output format -- because that
        shape is what makes `_STOP_SEQUENCES`' `"\\n\\n"` safe, and that
        property has to survive the format change. Single-line JSON contains no
        blank line, so the stop sequence still fires only after the reply is
        complete.
        """
        if self._structured:
            system = STRUCTURED_SYSTEM_PROMPT
            if context:
                system += f" Context: {context}"
            return (
                f"<|system|>\n{system}<|end|>\n"
                f"<|user|>\n{query}{STRUCTURED_SUFFIX}<|end|>\n"
                f"<|assistant|>\n"
            )
        return self._render_self_report_prompt(query, context)

    @staticmethod
    def _render_self_report_prompt(query: str, context: str) -> str:
        """The original Shape B prompt: answer plus a bare `CONFIDENCE:` line.

        Still reachable via `TWO_BRAIN_STRUCTURED=0`, and kept verbatim rather
        than reworded -- every measurement in
        `_real_inference_smoke_log.md` Attempt 5 was taken against this exact
        wording, so changing it would silently invalidate those receipts.

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

    def answer(self, query: str, context: str = "") -> BrainResponse:
        prompt = self._render_prompt(query, context)
        max_new_tokens = (
            self._STRUCTURED_MAX_NEW_TOKENS if self._structured else self._MAX_NEW_TOKENS
        )
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
                    token_count >= max_new_tokens
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

        return _to_brain_response(
            "".join(chunks).strip(), latency_ms, structured=self._structured
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

    **`answer()`'s `image` parameter is the one place in this file that goes
    beyond the plain `Brain` protocol** -- see `VisionBrain`. The router only
    ever passes `image` to a brain it already confirmed `can_see`; a non-vision
    `answer()` call is identical to before this capability existed.

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

    **Self-rating is now on by default for text queries.** This class used to
    be Shape A ("a real option for later, not a limitation of the backend") --
    that option has been taken. In structured mode it asks for
    `signals/structured.py`'s JSON object, so it reports both a confidence and
    a named gap, which makes it a Shape C brain. `TWO_BRAIN_STRUCTURED=0`
    returns it to Shape A.

    Two deliberate exceptions, both about images:

    - **`describe_image` and any `answer(..., image=...)` call stay plain
      prose.** A description exists to be masked and forwarded to a text-only
      deep brain; wrapping it in JSON would add a parse step between the VLM
      and the guard for no benefit, and `_MAX_DESCRIBE_TOKENS` is sized for
      prose. `_structured` is ignored whenever an image is present.
    - The router therefore keeps routing image-bearing queries the old way --
      see `router.py::route`.

    **Unverified on this machine.** The `llama-server` OpenCL build and the
    GGUF weights are not present here (`.gitignore`), so the structured path
    below has been exercised only against the parser and a fake transport, not
    against a real Adreno run. `NpuFastBrain`'s structured path *has* real
    receipts. Treat that asymmetry as real until someone runs
    `tests/test_gpu_brain.py` with the stack installed.
    """

    _MAX_NEW_TOKENS = 48
    #: Structured replies carry JSON scaffolding plus a prose `unknown` field,
    #: and a JSON object truncated mid-string fails to parse outright rather
    #: than degrading. Same reasoning as `NpuFastBrain._STRUCTURED_MAX_NEW_TOKENS`.
    _STRUCTURED_MAX_NEW_TOKENS = 512
    _MAX_DESCRIBE_TOKENS = 512
    _N_CTX = 4096
    _STARTUP_TIMEOUT_S = 120.0

    #: Inside the boundary, same as `NpuFastBrain`, and the loopback HTTP hop
    #: does not change that: `llama-server` is a child process this class
    #: started, bound to `127.0.0.1` on a port it chose, on this machine. No
    #: packet leaves the host. So this brain also sees the raw query.
    trusted_with_raw_pii = True

    #: llama.cpp compiles `response_format: {"type": "json_object"}` into a GBNF
    #: grammar, which makes malformed JSON structurally impossible rather than
    #: merely unlikely -- a much stronger guarantee than `NpuFastBrain` can get,
    #: since Genie's C API exposes no grammar hook. Sent optimistically and
    #: retried without it on a 4xx, because an older `llama-server` that rejects
    #: the field should cost one extra round trip, not make this brain unusable.
    _USE_JSON_RESPONSE_FORMAT = True

    def __init__(
        self,
        tier: str,
        signals: TierSignals,
        model_path: Path | None = None,
        mmproj_path: Path | None = None,
        bin_dir: Path | None = None,
        mmproj_offload: bool = True,
        log_path: Path | None = None,
        structured: bool | None = None,
    ) -> None:
        self.tier = tier
        self.signals = signals
        self._structured = _structured_default() if structured is None else structured
        # Both flip together and both are instance-level here (unlike the other
        # brains' class constants) because this one class covers two shapes.
        self.reports_confidence = self._structured
        self.reports_gaps = self._structured
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

    @property
    def can_see(self) -> bool:
        """True when a multimodal projector is loaded, so image input works."""
        return self._mmproj_path is not None

    def describe_image(self, image: Path, question: str = "") -> BrainResponse:
        """Turn an image into words, on-device, for a text-only tier to reason over.

        This is how an image-bearing query reaches the cloud at all. The deep
        brain is a text LLM with no vision support of any kind, so the image
        stays here and only this description -- after masking by the router --
        ever crosses the boundary.

        `question` steers what gets described. A physics problem needs the
        mechanical arrangement, angles and labelled quantities; "what is this?"
        needs identification. Describing for the question rather than in general
        is what makes the downstream text-only answer possible.
        """
        if not self.can_see:
            raise GpuBrainError("no multimodal projector loaded -- construct via GpuLocalBrain.for_vision()")
        if question:
            prompt = (
                "Describe this image in precise, complete detail for someone who cannot see it "
                "and must answer the following question from your description alone. "
                "State every quantity, label, number, angle, position and spatial relationship "
                "that could matter. Do not attempt to answer the question yourself.\n\n"
                f"Question: {question}"
            )
        else:
            prompt = (
                "Describe this image in precise, complete detail for someone who cannot see it. "
                "State every quantity, label, number, angle, position and spatial relationship."
            )
        return self.answer(prompt, image=image)

    @staticmethod
    def _image_data_uri(image: Path) -> str:
        import base64
        import mimetypes

        mime = mimetypes.guess_type(str(image))[0] or "image/png"
        return f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode("ascii")

    def _post_completion(self, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:200]
            # An older llama-server rejects `response_format` outright. Drop it
            # and retry once rather than failing the query: the prompt already
            # asks for JSON in words, and signals/structured.py parses a
            # best-effort reply. Only retried when the field was actually sent,
            # so this cannot loop.
            if exc.code < 500 and "response_format" in payload:
                retry = {k: v for k, v in payload.items() if k != "response_format"}
                return self._post_completion(retry)
            raise GpuBrainError(f"llama-server returned HTTP {exc.code}: {detail!r}") from exc

    def answer(self, query: str, context: str = "", image: Path | None = None) -> BrainResponse:
        if image is not None and not self.can_see:
            raise GpuBrainError(
                "image passed to a text-only brain -- construct via GpuLocalBrain.for_vision()"
            )

        # Structured mode is text-only: a described image has to come back as
        # prose for the guard to mask and the deep brain to read. See the class
        # docstring.
        structured = self._structured and image is None

        messages = []
        system = STRUCTURED_SYSTEM_PROMPT if structured else ""
        if context:
            system = f"{system} Context: {context}".strip() if system else context
        if system:
            messages.append({"role": "system", "content": system})
        if image is not None:
            # OpenAI-shaped content parts; llama-server routes these through mtmd.
            messages.append({"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": self._image_data_uri(image)}},
            ]})
        else:
            user_text = query + STRUCTURED_SUFFIX if structured else query
            messages.append({"role": "user", "content": user_text})

        if image is not None:
            # A description has to carry the whole image; the 48-token budget
            # that suits a fast-brain reply would truncate it.
            max_tokens = self._MAX_DESCRIBE_TOKENS
        elif structured:
            max_tokens = self._STRUCTURED_MAX_NEW_TOKENS
        else:
            max_tokens = self._MAX_NEW_TOKENS

        payload: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            # Lowered from 0.7 for structured replies: the confidence number and
            # the gap description are being *read* by the router, not by a
            # person, so sampling variance in them is noise in a routing
            # decision rather than welcome variety in prose.
            "temperature": 0.2 if structured else 0.7,
            "stream": False,
        }
        if structured and self._USE_JSON_RESPONSE_FORMAT:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        body = self._post_completion(payload)
        latency_ms = (time.perf_counter() - start) * 1000

        text = body["choices"][0]["message"]["content"].strip()
        if not structured:
            return BrainResponse(text=text, latency_ms=latency_ms, cost_usd=0.0)
        return _to_brain_response(text, latency_ms, structured=True)

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
    #: Confidence only, no gap decomposition -- this brain is deliberately
    #: untouched by the Shape C work (the phone is not wired in yet; see
    #: docs/ORCHESTRATOR.md). It keeps Shape B exactly as it was, so the mobile
    #: tier's behaviour and its tests are unchanged.
    reports_gaps = False

    #: **Outside the boundary, unlike the AI-PC brains.** The AI PC's models run
    #: on this machine and are handed the raw query; this one runs on a
    #: physically separate device reached over HTTP, so it keeps receiving
    #: masked text only. `adb reverse` makes that hop *look* like loopback,
    #: which is exactly why this is declared rather than inferred from the URL:
    #: the packets really do leave the host. `_assert_on_device` guards the
    #: same distinction from the other direction.
    trusted_with_raw_pii = False

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

    #: The self-rating instruction, as a system message, asking for the rating
    #: **first**.
    #:
    #: Two things were measured on a real S25 running Llama-3.2-3B, and both
    #: point the same way.
    #:
    #: 1. SELF_REPORT_SUFFIX on the user turn is followed inconsistently: fine
    #:    on factual questions, ignored on conversational ones ("Hi" came back
    #:    as a bare greeting). Instruction-tuned models weight the system turn
    #:    far more heavily for persistent formatting rules.
    #:
    #: 2. **Trailing rating + a token cap is a broken combination.** A long
    #:    answer ("what is the history of the Roman Empire") hits _MAX_TOKENS
    #:    and is truncated mid-sentence, so a rating asked for at the end is
    #:    simply never generated. The router then reads "no parseable
    #:    confidence" as "not confident" and escalates -- meaning the *longer*
    #:    and more expensive the local answer, the more likely it is thrown
    #:    away. Exactly backwards.
    #:
    #: Asking first makes the rating survive truncation, and costs nothing:
    #: parse_self_reported() uses a regex search and strips the line wherever
    #: it appears, so position is irrelevant to everything downstream. Raising
    #: the cap instead would only move the cliff, at ~16 tok/s on-device.
    _SELF_RATE_SYSTEM = (
        "You are a helpful assistant.\n"
        "ALWAYS begin your reply with a single line of exactly this form:\n"
        "CONFIDENCE: <a number from 0 to 100>\n"
        "The number is how confident you are that you can answer the user's "
        "message correctly and completely. Then, on the following lines, give "
        "your answer.\n"
        "Include the CONFIDENCE line every single time -- for greetings, small "
        "talk, questions you are unsure about, everything."
    )
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
            "messages": [
                {"role": "system", "content": self._SELF_RATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
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

    def answer(self, query: str, context: str = "") -> BrainResponse:
        """Answer `query`, and report how sure the model is about it.

        Never raises on a transport failure. Per L_INTERFACE_CONTRACT.md's
        error-handling section, a timeout or non-200 means "L failed" and
        should escalate rather than block the user -- so that path returns
        `confidence=0.0` (a definite escalate once inverted to difficulty) with
        `error` set for the audit trail, instead of propagating an exception
        the router would have to special-case.
        """
        prompt = f"{context}\n\n{query}" if context else query
        start = time.perf_counter()
        try:
            # No SELF_REPORT_SUFFIX here: it asks for the rating *after* the
            # answer, and _SELF_RATE_SYSTEM asks for it first. Sending both
            # gives the model contradictory instructions. See the system
            # message for why first wins.
            body = self._post_chat_completion(prompt)
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
