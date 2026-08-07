"""Real-hardware verification tests for `GpuLocalBrain` (Adreno GPU).

Mirrors `test_npu_brain.py`'s shape: these need the llama.cpp OpenCL build and
a GGUF, both of which are gitignored (see .gitignore). They skip cleanly on a
clean checkout and run for real on this machine.

The important test here is `test_actually_runs_on_gpu`. Everything else could
pass while llama.cpp silently ran on CPU -- which is the exact failure mode
docs/vlm-npu-research.md rejected the Hexagon path over, so it gets a real
device-placement assertion rather than a throughput heuristic.

Run with:
    PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_gpu_brain.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from two_brain_router.routing.brains import (
    _GPU_MODEL_PATH,
    _LLAMA_BIN_DIR,
    _VLM_MMPROJ_PATH,
    _VLM_MODEL_PATH,
    BrainResponse,
    GpuLocalBrain,
)
from two_brain_router.signals.loader import TierSignals

_SERVER_EXE = _LLAMA_BIN_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server")

pytestmark = pytest.mark.skipif(
    not (_SERVER_EXE.exists() and _GPU_MODEL_PATH.exists()),
    reason=(
        "llama.cpp OpenCL build or GGUF weights not present (both gitignored) -- "
        "set TWO_BRAIN_LLAMA_BIN / TWO_BRAIN_GPU_MODEL, see docs/local-inference-status.md"
    ),
)


@pytest.fixture
def brain(tmp_path):
    """Function-scoped, matching test_npu_brain.py.

    Each instance owns a `llama-server` child process, so module scope would
    leave servers alive across tests and hold GPU memory. A log path is always
    set so device placement stays checkable.
    """
    b = GpuLocalBrain(
        "pc",
        TierSignals.load("pc_3b", "ai_pc"),
        log_path=tmp_path / "llama-server.log",
    )
    try:
        yield b
    finally:
        b.close()


def test_real_inference_smoke(brain):
    """A real answer comes back -- not a stub, not empty."""
    response = brain.answer("What is gravity?")

    assert isinstance(response, BrainResponse)
    assert response.text.strip(), "empty answer"
    assert "mock" not in response.text.lower(), "got the stub, not real inference"
    assert response.latency_ms > 0
    assert response.cost_usd == 0.0, "on-device inference must be free"
    # Loose content check: any real answer to this question mentions one of these.
    assert any(word in response.text.lower() for word in ("force", "mass", "attract"))


def test_actually_runs_on_gpu(brain):
    """The point of this class: real Adreno execution, not a silent CPU fallback.

    llama-server hides device lines at default verbosity, so this asserts
    against the captured -v log rather than trusting throughput.
    """
    brain.answer("Hi")

    assert brain.verify_gpu_placement(), (
        "no GPU placement evidence in the server log -- llama.cpp may have "
        "fallen back to CPU (check OCL_ICD_FILENAMES and the OpenCL ICD)"
    )
    log = (brain._log_path).read_text(encoding="utf-8", errors="replace")
    assert "using device GPUOpenCL" in log
    assert "layers to GPU" in log


def test_server_is_reused_across_calls(brain):
    """The model stays resident -- a fast brain must not reload per query."""
    proc_before = brain._proc
    brain.answer("First question?")
    brain.answer("Second question?")

    assert brain._proc is proc_before, "server was restarted between calls"
    assert brain._proc.poll() is None, "server died mid-session"


def test_closes_cleanly(tmp_path):
    """close() must reap the child; a leaked server would hold the GPU."""
    b = GpuLocalBrain(
        "pc",
        TierSignals.load("pc_3b", "ai_pc"),
        log_path=tmp_path / "llama-server.log",
    )
    proc = b._proc
    b.answer("Hello")
    b.close()

    assert proc.poll() is not None, "llama-server still running after close()"
    assert b._proc is None
    b.close()  # idempotent -- must not raise


def test_router_end_to_end_with_real_brain(monkeypatch):
    """The router picks this brain up behind its env gate and routes through it."""
    from two_brain_router.routing.router import TwoBrainRouter

    monkeypatch.setenv("TWO_BRAIN_GPU_BRAIN", "1")
    router = TwoBrainRouter(tier="pc")
    try:
        assert isinstance(router.fast_brain, GpuLocalBrain)
        decision = router.route("What is gravity?")
        assert decision.answer.strip()
        assert "mock" not in decision.answer.lower()
    finally:
        router.fast_brain.close()


def test_gate_is_off_by_default(monkeypatch):
    """Unset, the router must stay on the stdlib-only mock."""
    from two_brain_router.routing.brains import LocalFastBrain
    from two_brain_router.routing.router import TwoBrainRouter

    monkeypatch.delenv("TWO_BRAIN_GPU_BRAIN", raising=False)
    monkeypatch.delenv("TWO_BRAIN_NPU_BRAIN", raising=False)

    assert isinstance(TwoBrainRouter(tier="pc").fast_brain, LocalFastBrain)


_VLM_PRESENT = _VLM_MODEL_PATH.exists() and _VLM_MMPROJ_PATH.exists()


def test_vision_variants_match_the_eval():
    """Each `prefer` maps to the weights RESULTS.md measured, with the right flags.

    Cheap to run (no model is loaded), so it guards the mapping even on a
    machine without the weights. The default is `speed` because the workload is
    interactive chat -- the 4B decodes at ~21 tok/s against the 8B's ~13 -- and
    that is a deliberate trade of 5/7 against 7/7, not an oversight.

    mmproj_offload MUST be False for the 8B and only the 8B: its vision tower is
    head_dim 72, llama.cpp's OpenCL flash-attention kernels cover only 64/128,
    and leaving offload on segfaults the process.
    """
    from two_brain_router.routing.brains import _VLM_DEFAULT_PREFER, _VLM_VARIANTS

    quality, balanced, speed = (_VLM_VARIANTS[k] for k in ("quality", "balanced", "speed"))

    assert "8B" in quality[0].name and "Q4_0" in quality[0].name
    assert quality[2] is False, "the 8B's vision encoder must stay off the GPU"
    assert "4B" in balanced[0].name and "Q8_0" in balanced[0].name
    assert balanced[2] is True
    assert "4B" in speed[0].name and "Q4_0" in speed[0].name
    assert speed[2] is True

    assert _VLM_DEFAULT_PREFER == "speed", (
        "default changed -- justify it against data/vlm_gpu_model/_eval/RESULTS.md"
    )


@pytest.mark.skipif(not _VLM_PRESENT, reason="VLM weights not present (gitignored)")
def test_for_vision_loads_the_default_variant(tmp_path):
    """The default really loads and answers, with its projector attached."""
    brain = GpuLocalBrain.for_vision(
        "pc", TierSignals.load("pc_3b", "ai_pc"), log_path=tmp_path / "srv.log"
    )
    try:
        assert brain._model_path == _VLM_MODEL_PATH
        assert brain._mmproj_path is not None, "vision defaults must load a projector"

        response = brain.answer("Name three primary colours.")
        assert response.text.strip()
        assert brain.verify_gpu_placement(), "language model should be on the GPU"
    finally:
        brain.close()


@pytest.mark.skipif(not _VLM_PRESENT, reason="VLM weights not present (gitignored)")
def test_for_vision_rejects_an_unknown_preference():
    with pytest.raises(ValueError, match="quality.*speed"):
        GpuLocalBrain.for_vision("pc", TierSignals.load("pc_3b", "ai_pc"), prefer="cheapest")


class _RecordingDeepBrain:
    """Stands in for the cloud tier and records exactly what crossed.

    Deliberately not the real endpoint: these assert the *boundary*, so they
    must not need credentials, a network, or the service to be up.
    """

    def __init__(self):
        self.seen: list[tuple[str, str]] = []

    def answer(self, masked_query: str, context: str = "") -> BrainResponse:
        self.seen.append((masked_query, context))
        return BrainResponse(text="cloud answer", latency_ms=1.0, cost_usd=0.0)


@pytest.mark.skipif(not _VLM_PRESENT, reason="VLM weights not present (gitignored)")
def test_image_never_crosses_the_boundary_only_a_masked_description(tmp_path):
    """The core guarantee for image routing.

    The deep brain is a text-only LLM -- the service has no VLM at all -- so an
    escalated image query must send a locally generated *description*, never
    the image, and that description must be masked like any other text.
    """
    from two_brain_router.routing.router import TwoBrainRouter

    image = Path("data/vlm_gpu_model/_eval/images/position.png").resolve()
    router = TwoBrainRouter(tier="pc")
    router.fast_brain = GpuLocalBrain.for_vision(
        "pc", TierSignals.load("pc_3b", "ai_pc"), log_path=tmp_path / "srv.log"
    )
    recorder = _RecordingDeepBrain()
    router.deep_brain = recorder
    try:
        decision = router.route(
            "Using the arrangement shown, derive a complete step-by-step geometric "
            "proof of the relative positions and justify every step rigorously.",
            image=image,
        )
    finally:
        router.fast_brain.close()

    assert decision.tier_answered == "cloud", "this query should escalate"
    assert recorder.seen, "deep brain was never called"
    sent_query, sent_context = recorder.seen[0]

    # No image, in any encoding, may appear in what crossed.
    both = sent_query + sent_context
    assert "data:image" not in both and "base64" not in both
    assert str(image) not in both, "not even the image path may cross"
    # A real description did cross.
    assert len(sent_context) > 50, "expected a substantive local description"
    assert any(n.startswith("image described on-device") for n in decision.notes)
    assert any("cloud tier has no VLM" in n for n in decision.notes)


@pytest.mark.skipif(not _VLM_PRESENT, reason="VLM weights not present (gitignored)")
def test_image_question_is_answered_locally_when_easy(tmp_path):
    """An easy visual question should never reach the cloud at all."""
    from two_brain_router.routing.router import TwoBrainRouter

    router = TwoBrainRouter(tier="pc")
    router.fast_brain = GpuLocalBrain.for_vision(
        "pc", TierSignals.load("pc_3b", "ai_pc"), log_path=tmp_path / "srv.log"
    )
    recorder = _RecordingDeepBrain()
    router.deep_brain = recorder
    try:
        decision = router.route(
            "What colour is the square?",
            image=Path("data/vlm_gpu_model/_eval/images/position.png").resolve(),
        )
    finally:
        router.fast_brain.close()

    assert decision.tier_answered == "local"
    assert not recorder.seen, "an easy image query must not reach the cloud"
    assert "blue" in decision.answer.lower(), f"local VLM did not see the image: {decision.answer!r}"


def test_image_without_a_vision_capable_brain_is_refused():
    """Fail loudly rather than silently dropping the image.

    Sending an image to a text-only model is the exact failure mode observed on
    the cloud endpoint, which returns 200 and answers "I don't see an image".
    """
    from two_brain_router.routing.router import TwoBrainRouter

    router = TwoBrainRouter(tier="pc")  # gates unset -> LocalFastBrain, no vision
    with pytest.raises(ValueError, match="cannot see"):
        router.route("What is this?", image=Path("data/vlm_gpu_model/_eval/images/position.png"))
