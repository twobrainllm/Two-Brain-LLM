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


@pytest.mark.skipif(not _VLM_PRESENT, reason="VLM weights not present (gitignored)")
def test_for_vision_uses_the_eval_winning_defaults(tmp_path):
    """The recommended VLM config is a default, not folklore in a doc.

    Pins the outcome of data/vlm_gpu_model/_eval/RESULTS.md: 8B Q4_0 scored
    7/7, ahead of 4B Q8_0 and even 4B BF16 at 6/7. If someone changes the
    default, this should make them justify it against the eval.

    mmproj_offload MUST be False here: the 8B's vision tower is head_dim 72,
    llama.cpp's OpenCL flash-attention kernels cover only 64/128, and leaving
    offload on segfaults the process.
    """
    brain = GpuLocalBrain.for_vision(
        "pc", TierSignals.load("pc_3b", "ai_pc"), log_path=tmp_path / "srv.log"
    )
    try:
        assert "8B" in brain._model_path.name and "Q4_0" in brain._model_path.name
        assert brain._mmproj_path is not None, "vision defaults must load a projector"
        assert brain._mmproj_offload is False, "8B vision encoder must stay off the GPU"

        response = brain.answer("Name three primary colours.")
        assert response.text.strip()
        assert brain.verify_gpu_placement(), "language model should still be on the GPU"
        log = (tmp_path / "srv.log").read_text(encoding="utf-8", errors="replace")
        assert "CLIP using CPU backend" in log, "vision encoder must be on CPU for the 8B"
    finally:
        brain.close()


@pytest.mark.skipif(not _VLM_PRESENT, reason="VLM weights not present (gitignored)")
def test_for_vision_rejects_an_unknown_preference():
    with pytest.raises(ValueError, match="quality.*speed"):
        GpuLocalBrain.for_vision("pc", TierSignals.load("pc_3b", "ai_pc"), prefer="cheapest")
