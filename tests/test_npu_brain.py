"""Phase 6 mandatory verification tests for the real on-NPU fast brain.

Per superpowers/deploy-local-brain-npu.md Phase 6 -- "definition of done"
tests, in addition to the base suite's 9 tests. These need the real
hardware/runtime stack from Phase 1 (.venv-npu, onnxruntime_qnn installed,
the real Phi-3.5-mini-instruct Genie artifact downloaded to
data/npu_model/phi-3.5-mini-instruct/) -- they skip cleanly when that stack
isn't present (e.g. the base .venv, or a clean checkout without the
gitignored artifact), and run for real under .venv-npu on this machine.

Run with:
    .venv-npu\\Scripts\\python.exe -m pytest tests\\test_npu_brain.py -v
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import pytest

from two_brain_router.routing.brains import _NPU_ARTIFACT_DIR

try:
    import onnxruntime_qnn

    _NPU_STACK_AVAILABLE = True
except ImportError:
    _NPU_STACK_AVAILABLE = False

_ARTIFACT_PRESENT = _NPU_ARTIFACT_DIR.exists()

pytestmark = pytest.mark.skipif(
    not (_NPU_STACK_AVAILABLE and _ARTIFACT_PRESENT),
    reason=(
        "real NPU runtime stack not present -- run from .venv-npu with the "
        "Phi-3.5-mini-instruct Genie artifact downloaded "
        "(see superpowers/deploy-local-brain-npu.md Phase 1/2a)"
    ),
)

# Real cold load measured ~13.1s (data/npu_model/.../_real_inference_smoke_log.md,
# Attempt 3b) -- these are generous ceilings to catch a genuine hang/regression,
# not tight perf assertions.
_COLD_LOAD_CEILING_MS = 60_000
_INFERENCE_CEILING_MS = 60_000


def _find_genie_cli() -> str | None:
    """Locate genie-t2t-run.exe from any installed system QAIRT SDK.

    Only the real CLI's `--log info` gives an automatable EP-assignment
    receipt here: Genie's C API logging callback (GenieLog.h) is a
    printf-style `va_list` callback that ctypes cannot marshal reliably --
    attempting it would risk a subtly-wrong capture, which this project's
    receipts rule treats as worse than not measuring at all. This shells
    out to the same real CLI path already proven working in
    _real_inference_smoke_log.md instead.
    """
    candidates = sorted(
        glob.glob(r"C:\Qualcomm\AIStack\QAIRT\*\bin\aarch64-windows-msvc\genie-t2t-run.exe")
    )
    return candidates[-1] if candidates else None


@pytest.fixture
def npu_brain():
    """Function-scoped, not module-scoped: a real run showed a second live
    Genie dialog session (e.g. one opened via `TwoBrainRouter` while this
    fixture's session was still held module-wide) fails to create its HTP
    context binary -- real error `Could not create context from binary for
    context index = 2 : err 1002`, `Failed to free device: 14003`. This
    hardware/runtime does not support two concurrent Genie sessions;
    closing this fixture's session before the next test opens its own
    avoids the collision rather than masking it."""
    from two_brain_router.routing.brains import NpuFastBrain
    from two_brain_router.signals.loader import TierSignals

    signals = TierSignals.load("pc_3b", "ai_pc")
    brain = NpuFastBrain("pc", signals)
    yield brain
    brain.close()


def test_real_inference_smoke(npu_brain):
    """A fixed prompt against the real deployed model returns a non-empty,
    well-formed completion within a generous timeout -- not mocked, not the
    stub string."""
    start = time.perf_counter()
    response = npu_brain.answer("What time zone is Tokyo in?")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.text.strip()
    assert not response.text.startswith("[local:")  # not the LocalFastBrain stub
    assert not response.text.startswith("[cloud:")
    assert elapsed_ms < _INFERENCE_CEILING_MS
    assert response.cost_usd == 0.0


def test_npu_ep_assignment():
    """Inference nodes ran on the QnnHtp (Hexagon NPU) backend, not a CPU
    fallback -- verified from Genie's own real execution log. This is the
    equivalent receipt to an ORT EP-assignment report for this runtime:
    Genie's backend is fixed to QnnHtp per genie_config.json with no
    ORT-style dynamic EP selection to inspect instead (see brains.py's
    module docstring and the Known issue/EP sections of
    _real_inference_smoke_log.md)."""
    genie_cli = _find_genie_cli()
    if genie_cli is None:
        pytest.skip("no system QAIRT SDK with genie-t2t-run.exe found on this machine")

    lib_dir = str(onnxruntime_qnn.LIB_DIR_FULL_PATH)
    cli_dir = os.path.dirname(genie_cli)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([lib_dir, cli_dir, env.get("PATH", "")])

    sample_prompt = _NPU_ARTIFACT_DIR / "sample_prompt.txt"
    result = subprocess.run(
        [
            genie_cli,
            "-c", "genie_config.json",
            "--prompt_file", str(sample_prompt),
            "--log", "info",
            "--action", "ABORT",
            "--sleep", "5000",
        ],
        cwd=str(_NPU_ARTIFACT_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    log = result.stdout + result.stderr
    assert "QnnGraph_execute started" in log
    assert "QnnGraph_execute done" in log
    assert "QnnHtp" in log
    # Genie's backend for this bundle has no CPU-EP-fallback path at all
    # (fixed to QnnHtp) -- absence of these markers is real negative
    # evidence, not just "didn't look".
    assert "ExecutionProvider" not in log
    assert "forced to cpu" not in log.lower()


def test_router_end_to_end_with_real_brain(monkeypatch):
    """Run all three demo queries through TwoBrainRouter with NpuFastBrain
    wired in: the easy query now returns a genuinely generated answer, and
    masking/escalation is otherwise unchanged."""
    monkeypatch.setenv("TWO_BRAIN_NPU_BRAIN", "1")
    from two_brain_router.routing.router import TwoBrainRouter

    router = TwoBrainRouter(tier="pc")
    try:
        easy = router.route("What time zone is Tokyo in?")
        assert easy.tier_answered == "local"
        assert not easy.answer.startswith("[local:")
        assert easy.answer.strip()

        hard = router.route(
            "Derive the time complexity of merge sort step by step and "
            "compare it to quicksort's worst case, then explain the "
            "trade-offs."
        )
        assert hard.tier_answered == "cloud"

        pii = router.route(
            "My email is jane.doe@example.com and my phone is "
            "555-123-4567 -- can you draft a reply telling the sender "
            "their SSN 123-45-6789 was found in an old backup and needs "
            "to be rotated?"
        )
        assert pii.tier_answered == "cloud"
        assert pii.pii_entities_masked >= 1
        sent_off_device = next(n for n in pii.notes if n.startswith("sent off-device"))
        assert "jane.doe@example.com" not in sent_off_device
        assert "jane.doe@example.com" in pii.answer
    finally:
        router.fast_brain.close()


def test_brain_reachable_cleanly():
    """A clean-process smoke test: import NpuFastBrain, load the session,
    and answer one query deterministically end-to-end, with a hard ceiling
    on cold-load time."""
    script = (
        "import time, sys\n"
        "from two_brain_router.routing.brains import NpuFastBrain\n"
        "from two_brain_router.signals.loader import TierSignals\n"
        "signals = TierSignals.load('pc_3b', 'ai_pc')\n"
        "t0 = time.perf_counter()\n"
        "brain = NpuFastBrain('pc', signals)\n"
        "load_ms = (time.perf_counter() - t0) * 1000\n"
        "resp = brain.answer('What time zone is Tokyo in?')\n"
        "brain.close()\n"
        "print('LOAD_MS=%f' % load_ms)\n"
        "sys.exit(0 if resp.text.strip() else 1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=_COLD_LOAD_CEILING_MS / 1000 + 30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    load_ms = None
    for line in result.stdout.splitlines():
        if line.startswith("LOAD_MS="):
            load_ms = float(line.split("=", 1)[1])
    assert load_ms is not None
    assert load_ms < _COLD_LOAD_CEILING_MS
