"""Which local models are installed, and how to build a brain for each.

The router has always had exactly one fast brain, chosen by environment
variable at startup. That is right for a deployment and wrong for a demo: the
interesting question here is *how the tiers compare*, and answering it meant
restarting the server with different env vars. This module makes the choice a
value instead -- something the UI can list and switch between.

**Discovery, not configuration.** A model appears in the list only if its files
are actually on disk (`LocalModel.available`). A dropdown offering something
that will fail to load 17 seconds later is worse than a shorter dropdown.

Paths are resolved against `MODEL_ROOTS`: this checkout's own `data/` plus
anything in `TWO_BRAIN_MODEL_ROOT`. The second exists because the weights are
gitignored and large enough that people keep one copy and point several
checkouts at it -- on this machine they live in a different clone entirely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from two_brain_router.signals.loader import DATA_DIR

#: Where to look for weights, in order. `TWO_BRAIN_MODEL_ROOT` should point at
#: a directory laid out like this repo's `data/` (i.e. containing `npu_model/`
#: and `vlm_gpu_model/`); several roots may be given, os.pathsep-separated.
def _model_roots() -> list[Path]:
    roots = [DATA_DIR]
    extra = os.environ.get("TWO_BRAIN_MODEL_ROOT", "")
    roots.extend(Path(p) for p in extra.split(os.pathsep) if p.strip())
    return roots


def _find(*relative: str) -> Path | None:
    """First existing match for `relative` across the roots, else None."""
    for root in _model_roots():
        candidate = root.joinpath(*relative)
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class LocalModel:
    """One selectable local model, and everything needed to instantiate it."""

    id: str
    label: str
    #: Which backend runs it. `npu` is Genie via ctypes; `gpu` is a
    #: `llama-server` child on the Adreno.
    backend: Literal["npu", "gpu"]
    #: One line for the UI -- size, quantisation, and the measured trade-off,
    #: so the choice is informed rather than a name-guessing game.
    detail: str
    model_rel: tuple[str, ...] = ()
    mmproj_rel: tuple[str, ...] = ()
    #: False forces the vision encoder onto CPU. Mandatory for Qwen3-VL-8B:
    #: its vision tower is head_dim 72 and llama.cpp's OpenCL flash-attention
    #: kernels cover only 64/128, so offloading it segfaults (exit 139, no
    #: clean error) -- see data/vlm_gpu_model/qwen3-vl-8b-instruct/
    #: _real_inference_smoke_log.md, Attempt 1.
    mmproj_offload: bool = True

    @property
    def model_path(self) -> Path | None:
        return _find(*self.model_rel) if self.model_rel else None

    @property
    def mmproj_path(self) -> Path | None:
        return _find(*self.mmproj_rel) if self.mmproj_rel else None

    @property
    def can_see(self) -> bool:
        return bool(self.mmproj_rel) and self.mmproj_path is not None

    @property
    def available(self) -> bool:
        """Whether this model's files are actually present."""
        if self.model_rel and self.model_path is None:
            return False
        if self.mmproj_rel and self.mmproj_path is None:
            return False
        return True

    def as_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "backend": self.backend,
            "detail": self.detail,
            "vision": self.can_see,
            "available": self.available,
        }


#: The catalogue. One Qwen per size, chosen by the repo's own measured eval
#: (`data/vlm_gpu_model/_eval/RESULTS.md`) rather than by size alone:
#:
#:     8B Q4_0  7/7   <- best overall, despite vision forced to CPU
#:     4B Q8_0  6/7   <- best 4B; BF16 ties it at 1.9x the weights
#:     4B BF16  6/7
#:     4B Q4_0  5/7
#:
#: Plus both Phi-3.5-mini builds, which are the *same model* on two backends --
#: the whole point of keeping both is that it isolates device from model when
#: comparing NPU against GPU.
CATALOGUE: tuple[LocalModel, ...] = (
    LocalModel(
        id="phi35-npu",
        label="Phi-3.5-mini - NPU",
        backend="npu",
        detail="3.8B w4a16 - Hexagon NPU - in-process Genie - lowest power",
    ),
    LocalModel(
        id="phi35-gpu",
        label="Phi-3.5-mini - GPU",
        backend="gpu",
        detail="3.8B Q4_0 - Adreno - same model as the NPU build, different device",
        model_rel=("npu_model", "phi-3.5-mini-instruct", "gguf", "Phi-3.5-mini-instruct-Q4_0.gguf"),
    ),
    LocalModel(
        id="qwen3vl-4b",
        label="Qwen3-VL 4B - GPU",
        backend="gpu",
        detail="4B Q8_0 - vision on GPU - 6/7 on the eval - ~39% faster than the 8B",
        model_rel=("vlm_gpu_model", "qwen3-vl-4b-instruct", "raw", "Qwen3-VL-4B-Instruct-Q8_0.gguf"),
        mmproj_rel=("vlm_gpu_model", "qwen3-vl-4b-instruct", "raw", "mmproj-Qwen3VL-4B-Instruct-f16.gguf"),
    ),
    LocalModel(
        id="qwen3vl-8b",
        label="Qwen3-VL 8B - GPU",
        backend="gpu",
        detail="8B Q4_0 - vision on CPU (required) - 7/7 on the eval - longest answers",
        model_rel=("vlm_gpu_model", "qwen3-vl-8b-instruct", "raw", "Qwen3-VL-8B-Instruct-Q4_0.gguf"),
        mmproj_rel=("vlm_gpu_model", "qwen3-vl-8b-instruct", "raw", "mmproj-F16.gguf"),
        mmproj_offload=False,  # mandatory; segfaults otherwise
    ),
)

BY_ID: dict[str, LocalModel] = {m.id: m for m in CATALOGUE}


def available_models() -> list[LocalModel]:
    """The catalogue, filtered to what is actually installed.

    The NPU entry is additionally gated on the runtime being importable: the
    Genie artifact can be present in a checkout whose venv has no
    `onnxruntime_qnn`, and offering it there produces a failure 12 seconds into
    a load rather than an absent option.
    """
    out = []
    for model in CATALOGUE:
        if not model.available:
            continue
        if model.backend == "npu":
            try:
                import onnxruntime_qnn  # noqa: F401
            except ImportError:
                continue
            from two_brain_router.routing.brains import _NPU_ARTIFACT_DIR

            if not _NPU_ARTIFACT_DIR.exists():
                continue
        out.append(model)
    return out


def build_brain(model: LocalModel, tier: str, signals):
    """Instantiate the brain for `model`. Raises if its runtime is missing.

    Vision-capable GPU models go through `GpuLocalBrain.for_vision`'s argument
    shape rather than around it, so `mmproj_offload` cannot be forgotten -- on
    the 8B that would be a segfault, not a degraded answer.
    """
    from two_brain_router.routing.brains import GpuLocalBrain, NpuFastBrain

    if model.backend == "npu":
        return NpuFastBrain(tier, signals)
    return GpuLocalBrain(
        tier,
        signals,
        model_path=model.model_path,
        mmproj_path=model.mmproj_path,
        mmproj_offload=model.mmproj_offload,
    )
