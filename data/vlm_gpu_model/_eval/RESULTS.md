# VLM quality eval — which variant to actually use

`_mock: false`. Real inference on this machine, deterministic decoding
(`--temp 0 --seed 42`), 7 objectively-checkable items over 6 generated images
with pixel-derived ground truth. Regenerate with `make_eval_set.py`, re-run with
`run_eval.py`, raw rows in `results.jsonl`.

## Result

| Variant | Weights | Score | Eval time | Failed |
|---|---|---|---|---|
| 4B Q4_0 *(what was deployed)* | 2.38 GB | 5/7 | 86 s | count, digits |
| **4B Q8_0** | 4.28 GB | **6/7** | 94 s | digits |
| 4B BF16 *(full precision)* | 8.05 GB | 6/7 | 149 s | digits |
| **8B Q4_0** *(vision on CPU)* | 4.79 GB | **7/7** | 154 s | — |

### Two findings, both counter-intuitive

**1. Model size beats numeric precision.** The 8B at *4-bit* outscores the 4B at
*full precision*, and does it while being forced to run its vision encoder on
CPU. If the goal is VLM quality, parameters are the lever, not bit-width.

**2. Full precision buys nothing over Q8_0.** BF16 and Q8_0 score identically
(6/7, same single failure) while BF16 is ~1.9x the weights and ~60% slower.
Going past 8-bit is wasted on this workload — the usual result for Q8_0, now
confirmed here rather than assumed.

The step that *does* pay is Q4_0 → Q8_0 on the 4B: it fixed the counting error
(said 4, actual 5) for +1.8 GB and +9% time.

## This corrects an earlier recommendation

`../qwen3-vl-8b-instruct/_real_inference_smoke_log.md` concluded the 8B was "a
downgrade on this hardware on every axis measured" and recommended the 4B. That
was based on **one prompt on a weak test card** — an image whose circle sat at
exactly (0.50W, 0.50H), where "centre" versus "lower-centre" is a judgement
call. On a real eval the 8B is the most accurate variant available here.

The *performance* half of that conclusion still stands: the 8B is slower, needs
`--no-mmproj-offload`, and loses GPU vision acceleration. It is a
quality-versus-speed trade, not a strict downgrade.

## Two flaws found in this harness, and fixed

Worth recording because both would have produced a wrong answer:

1. **Non-deterministic decoding.** `llama-mtmd-cli` defaults to `temp 0.20`
   with a *random seed*, so comparing variants on single samples would partly
   measure sampling noise. Pinned to `--temp 0 --seed 42`.
2. **An under-specified question.** Asked openly where the shapes were, 4B Q8_0
   answered "blue square is on the left, red circle is on the right" — correct,
   but coarser than the check, which scored it wrong for being *vague* rather
   than *mistaken*, and made Q8_0 look worse than Q4_0. The question now demands
   the granularity it grades ("Answer using one of: top-left, top-right,
   bottom-left, bottom-right"). First run before this fix: Q8_0 4/7. After: 6/7.

## Limits of this eval — read before trusting it

- **It saturates.** 8B Q4_0 is already 7/7, so it cannot distinguish anything
  better. Comparing 8B Q4_0 against 8B Q8_0 would need harder items first.
- **n=1 per item.** Deterministic, so re-running gives the same answer, but a
  different seed or slightly reworded prompt could shift a borderline item.
- **Synthetic images.** Flat colours, hard edges, no photographic noise,
  occlusion, lighting or clutter. It measures perception of clean geometry, not
  real-world robustness.
- **The digits item may be unfairly hard.** It renders a 7-segment "47"; every
  4B variant read it as "41". Seven-segment glyphs are out-of-distribution for a
  VLM, so this probes stylised-glyph OCR rather than text reading generally. It
  is the single item separating 8B from 4B, so **that gap rests on one
  item** — do not over-read it.

## Recommendation

- **Best quality: `Qwen3-VL-8B-Instruct-Q4_0` + `--no-mmproj-offload`.** Accept
  vision-on-CPU and ~1.6x the latency.
- **Best balance: `Qwen3-VL-4B-Instruct-Q8_0`.** One point behind, keeps the
  vision encoder on the GPU, and 39% faster.
- **Do not use BF16.** No measured benefit over Q8_0 at twice the size.
- **Do not stay on 4B Q4_0** if quality matters — Q8_0 is a strict improvement
  for +1.8 GB.
