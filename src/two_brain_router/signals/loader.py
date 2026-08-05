"""Loads captured QUAD tool responses from `data/`.

Every file under `data/<tool>/` is a real response shape for one of the four
MCP tools this project drives:

    hardware_detect -> convert_model -> profile_workload -> orchestrate_workload

Some are real captures, some are mocked because the tool genuinely failed --
each `data/*/_real_*_log.md` states which, with the exact calls and error
strings. This module does not care which: it only knows the response *shape*,
so nothing here changes when a mocked file is replaced with a real capture.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# src/two_brain_router/signals/loader.py -> project root is three levels up.
# This holds for a source checkout and for an editable install (`pip install -e .`,
# which is what requirements.txt does). Installed as a plain wheel the package
# sits in site-packages with no data/ beside it -- set TWO_BRAIN_DATA_DIR then.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Override with TWO_BRAIN_DATA_DIR to point at a different capture set (e.g. a
#: directory of real responses recorded on a working server).
DATA_DIR = Path(os.environ.get("TWO_BRAIN_DATA_DIR", PROJECT_ROOT / "data"))


def load_tool_response(tool: str, name: str, data_dir: Path | None = None) -> dict:
    """Read `data/<tool>/<name>.json`."""
    root = data_dir or DATA_DIR
    return json.loads((root / tool / f"{name}.json").read_text())


@dataclass
class TierSignals:
    """The four tool responses that describe one tier (mobile / pc / cloud)."""

    device_tier: str
    hardware: dict
    convert: dict
    profile: dict
    orchestrate: dict | None

    @classmethod
    def load(cls, tier_name: str, hw_file: str, data_dir: Path | None = None) -> "TierSignals":
        root = data_dir or DATA_DIR
        orchestrate_path = root / "orchestrate_workload" / f"{tier_name}.json"
        return cls(
            device_tier=tier_name,
            hardware=load_tool_response("hardware_detect", hw_file, root),
            convert=load_tool_response("convert_model", tier_name, root),
            profile=load_tool_response("profile_workload", tier_name, root),
            # The cloud tier has no orchestrate_workload response -- op placement
            # is an on-device problem and the tool has no cloud target (gap #4).
            orchestrate=load_tool_response("orchestrate_workload", tier_name, root)
            if orchestrate_path.exists()
            else None,
        )
