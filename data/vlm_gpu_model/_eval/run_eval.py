"""Score VLM variants on the eval set. Real inference, objective scoring.

    python run_eval.py --model <path.gguf> --mmproj <path.gguf> [--label Q4_0]

Drives `llama-mtmd-cli` directly (GenieX cannot do VLMs -- see
../_real_geniex_vlm_log.md). Prints a per-item pass/fail and a total, and
appends a JSON row to results.jsonl so runs accumulate for comparison.

Scoring is deliberately lenient on prose and strict on fact: an answer passes an
`any_of` check if it contains any listed phrase, an `ordered` check if the
listed words appear in that relative order, and fails if any `none_of` phrase
appears. It scores what the model *got right*, not how it worded it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[2]
BIN = Path(os.environ.get("TWO_BRAIN_LLAMA_BIN", REPO / ".llama-cpp-opencl" / "extracted"))


def ask(model: Path, mmproj: Path, image: Path, question: str, ngl: int,
        mmproj_offload: bool, image_min_tokens: int | None) -> tuple[str, float]:
    cmd = [
        str(BIN / "llama-mtmd-cli.exe"),
        "-m", str(model), "--mmproj", str(mmproj), "--image", str(image),
        "-p", question, "-c", "4096", "-ngl", str(ngl),
        # llama-mtmd-cli defaults to temp 0.20 with a random seed, so repeated
        # runs disagree. Comparing quantizations on stochastic single samples
        # would measure sampling noise as much as model quality -- pin both.
        "--temp", "0", "--seed", "42",
    ]
    if not mmproj_offload:
        cmd.append("--no-mmproj-offload")
    if image_min_tokens:
        cmd += ["--image-min-tokens", str(image_min_tokens)]

    env = dict(os.environ)
    icd = BIN / "OpenCL_adreno.dll"
    if icd.exists():
        env.setdefault("OCL_ICD_FILENAMES", str(icd))
    env["PATH"] = f"{BIN}{os.pathsep}{env.get('PATH', '')}"

    start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    return proc.stdout.strip(), (time.perf_counter() - start)


def grade(answer: str, checks: list[dict]) -> list[tuple[str, bool]]:
    low = answer.lower()
    out = []
    for chk in checks:
        ok = True
        if "any_of" in chk:
            ok = any(p in low for p in chk["any_of"])
        if ok and "ordered" in chk:
            # every word present, and in the required relative order
            positions = [low.find(w) for w in chk["ordered"]]
            ok = all(p >= 0 for p in positions) and positions == sorted(positions)
        if ok and "none_of" in chk:
            ok = not any(p in low for p in chk["none_of"])
        out.append((chk["name"], ok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--mmproj", required=True, type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--no-mmproj-offload", action="store_true")
    ap.add_argument("--image-min-tokens", type=int, default=None)
    args = ap.parse_args()

    label = args.label or args.model.stem
    items = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))

    passed = total = 0
    elapsed = 0.0
    detail = []
    print(f"\n=== {label} ===")
    for item in items:
        answer, secs = ask(
            args.model, args.mmproj, HERE / "images" / item["image"],
            item["question"], args.ngl, not args.no_mmproj_offload,
            args.image_min_tokens,
        )
        elapsed += secs
        results = grade(answer, item["checks"])
        for name, ok in results:
            passed += ok
            total += 1
            print(f"  [{'PASS' if ok else 'FAIL'}] {item['image']:14s} {name}")
        if not all(ok for _, ok in results):
            flat = re.sub(r"\s+", " ", answer)[:150]
            print(f"         answer: {flat}")
        detail.append({"image": item["image"], "answer": answer,
                       "checks": {n: ok for n, ok in results}, "seconds": round(secs, 1)})

    print(f"  SCORE {passed}/{total}  ({elapsed:.0f}s total)")
    with (HERE / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "label": label, "model": args.model.name, "mmproj": args.mmproj.name,
            "score": passed, "total": total, "seconds": round(elapsed, 1),
            "mmproj_offload": not args.no_mmproj_offload,
            "image_min_tokens": args.image_min_tokens, "detail": detail,
        }) + "\n")


if __name__ == "__main__":
    main()
