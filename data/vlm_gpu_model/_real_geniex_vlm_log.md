# Driving the VLMs through GenieX -- real log (it does not work)

`_mock: false`. Applies to **both** models under this directory.

**Question:** GenieX is Qualcomm's recommended runtime and its `gpu` device is
the same `ggml-opencl`/`GPUOpenCL` backend these models already run on
(see each model's `_real_inference_smoke_log.md`). Can we drive the VLMs
through GenieX instead of invoking `llama-mtmd-cli` directly, and get a
supported API instead of hand-assembled binaries?

**Answer: no, not in GenieX 0.3.18.** Multimodal generation fails for both the
4B and the 8B with the same error, while the *identical* model, mmproj, image,
device and backend work correctly through `llama-mtmd-cli`. This is a bug in
GenieX's own layer, not in the hardware, the backend, or the models.

## What works

Model resolution and loading are fine, once the right hub is used:

```
geniex-py chat qwen3-vl-8b \
  --hub localfs --local-path <repo>/data/vlm_gpu_model/qwen3-vl-8b-instruct/raw \
  --device gpu -p "Describe this image. C:/.../test_image.png"
```

```
loading qwen3-vl-8b ... done (vlm, 13.2s, llama_cpp:GPUOpenCL)
vlm.cpp:99:create] mmproj loaded: vision=true, audio=false
vlm.cpp:223:generate] successfully loaded image 0: C:/.../test_image.png
vlm.cpp:255:generate] total media files loaded: 1
vlm.cpp:272:generate] using multimodal (mtmd) path with ctx_vision
```

It correctly detects the model as a VLM (`vlm`, not `llm`), finds and loads the
mmproj, loads the image, and selects the mtmd path -- all on `GPUOpenCL`.

**Note the hub matters.** Passing a local `.gguf` path directly is silently
degrading: GenieX logs `invalid model name: '<path>' (must be 'org/repo' ...)`,
never resolves an mmproj, and loads the model as **`llm`** rather than `vlm`.
The image is then never passed, and the model answers from text alone -- in our
run it produced a confident refusal ("I do not have access to your local file
system"), which looks like a model limitation but is actually a silently
dropped image. Use `--hub localfs --local-path <dir>`.

## What fails -- the media marker mismatch

```
mtmd_tokenize: error: number of media markers in text (0) does not match number of bitmaps (1)
vlm.cpp:292:generate] mtmd_tokenize failed
error: GenieXError(-201201): Multimodal generation failed
```

The decisive evidence is the prompt GenieX itself built, from its own debug
output (this run had one marker injected by GenieX plus one added manually, to
test):

```
VlmGenerateInput(prompt_utf8: <|im_start|>user
<__media__><__media__>Describe this image.<|im_end|>
<|im_start|>assistant
, ... image_count: 1, ...)
```

**Two `<__media__>` markers are demonstrably present in the text, and GenieX's
own `mtmd_tokenize` reports finding zero.** So the marker GenieX injects is not
the marker its vendored mtmd is configured to search for. Adding the marker by
hand does not help, which rules out "the user must supply it" -- the count is
reported as 0 regardless of whether the text contains 0, 1 or 2 of them.

That `<__media__>` is the *correct* marker for llama.cpp is independently
confirmed by our direct runs, where `llama-mtmd-cli` logs
`chat_add_and_format: new_msg.content='<__media__>Describe this image.'` and
then encodes the image successfully on the same backend.

## Reproduced on both models

| Model | Loads as VLM? | mmproj found | Result |
|---|---|---|---|
| Qwen3-VL-8B-Instruct | yes (13.2 s) | vision=true | `mtmd_tokenize failed`, exit 1 |
| Qwen3-VL-4B-Instruct | yes (4.8 s) | vision=true | `mtmd_tokenize failed`, exit 1 |

Identical failure on the model that is otherwise fully working end-to-end on
this exact GPU backend. That isolates the defect to GenieX's multimodal
plumbing.

Also logged, harmless but noted:
`thinking mode not supported for llama.cpp VLM; ignoring enable_thinking=true`.

## Consequence

**Keep invoking `llama-mtmd-cli` directly for VLM work.** The GenieX wrapper
buys nothing here today -- it is the same backend underneath (confirmed: both
report device `GPUOpenCL`, Adreno X1-85), so there is no performance or
capability difference to chase, only a supported-API benefit that is currently
unavailable because the feature is broken.

Worth re-testing on a future GenieX release; the signature to watch is
`number of media markers in text (0)` from `mtmd_tokenize` despite markers
being present in `prompt_utf8`. Text-only generation through GenieX is
unaffected and works well (see
`data/npu_model/phi-3.5-mini-instruct/_real_geniex_hybrid_log.md`).
