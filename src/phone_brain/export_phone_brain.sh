#!/usr/bin/env bash
#
# export_phone_brain.sh
#
# Exports Llama-3.2-3B-Instruct to a Genie-ready QNN bundle targeting the
# Snapdragon 8 Elite (Galaxy S25's SoC), at the officially-supported w4a16
# precision (4-bit weights / 16-bit activations, with sensitive layers at
# w8a16 and an 8-bit KV cache).
#
# Why w4a16 and not w4a8: as of this writing, Qualcomm's published Llama
# recipes for the Genie/QNN path ship at w4a16 -- that's what's benchmarked
# and supported for your exact chipset. True W4A8 isn't an offered preset
# for these LLM exports; getting there would mean a custom AIMET PTQ recipe
# outside this script. Swap PRECISION_NOTE below only after you've confirmed
# (via --help, see below) that your installed qai_hub_models version exposes
# a different precision flag for this model.
#
# ---------------------------------------------------------------------------
# PREREQS (one-time, on your dev machine -- not the phone):
#
#   1. Python 3.10-3.13, then:
#        pip install "qai-hub-models[llama-v3-2-3b-instruct]"
#
#   2. Create a Qualcomm AI Hub account at https://aihub.qualcomm.com,
#      grab an API token from Account -> Settings -> API Token, then:
#        qai-hub configure --api_token <YOUR_AI_HUB_TOKEN>
#
#   3. Llama 3.2 weights are gated on Hugging Face. Request access to
#      meta-llama/Llama-3.2-3B-Instruct, then on this machine:
#        huggingface-cli login
#
#   4. Confirm what precision options your installed version actually
#      supports before assuming w4a16 is the only one:
#        python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --help
#
# USAGE:
#   ./export_phone_brain.sh [output_dir]
#
# ---------------------------------------------------------------------------

set -euo pipefail

OUTPUT_DIR="${1:-genie_bundle_l_phone}"
CHIPSET="qualcomm-snapdragon-8-elite"   # Galaxy S25's SoC
CONTEXT_LEN="${CONTEXT_LEN:-2048}"       # trim further if you hit device memory pressure

echo "=================================================================="
echo " Exporting Llama-3.2-3B-Instruct  ->  Genie/QNN bundle"
echo " Chipset:        ${CHIPSET}"
echo " Context length: ${CONTEXT_LEN}"
echo " Output dir:     ${OUTPUT_DIR}"
echo "=================================================================="

python -m qai_hub_models.models.llama_v3_2_3b_instruct.export \
  --chipset "${CHIPSET}" \
  --context-length "${CONTEXT_LEN}" \
  --skip-profiling \
  --output-dir "${OUTPUT_DIR}"

echo
echo "==> Export complete: ${OUTPUT_DIR}/"
echo
echo "Next steps:"
echo "  1. Connect the S25 with USB debugging enabled, then:"
echo "       adb push ${OUTPUT_DIR} /data/local/tmp/genie_bundle_l"
echo "  2. Install/serve via GenieX or the raw Genie/QAIRT runtime on-device."
echo "  3. Run test_phone_brain.py against the served endpoint to sanity-check"
echo "     latency and tokens/sec before wiring in the orchestrator."
