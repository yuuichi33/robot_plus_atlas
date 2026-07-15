#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
ONNX="$MODEL_DIR/best.onnx"
OUTPUT="$MODEL_DIR/task_yolov5n_fp16"
OM="$OUTPUT.om"
EXPECTED_ONNX_SHA256="43ba02b2451656b91bb7b2758c98a612d4c648ee9d49f63597c2b738a92f9bfe"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"

command -v atc >/dev/null 2>&1 || {
  echo "[ERROR] atc not found; source the matching CANN set_env.sh first" >&2
  exit 2
}
command -v sha256sum >/dev/null 2>&1 || {
  echo "[ERROR] sha256sum not found" >&2
  exit 2
}
[[ -s "$ONNX" ]] || {
  echo "[ERROR] source ONNX missing: $ONNX" >&2
  exit 2
}

ACTUAL_ONNX_SHA256="$(sha256sum "$ONNX" | awk '{print $1}')"
if [[ "$ACTUAL_ONNX_SHA256" != "$EXPECTED_ONNX_SHA256" ]]; then
  echo "[ERROR] ONNX SHA-256 mismatch" >&2
  exit 2
fi

TEMP_OUTPUT="$MODEL_DIR/.task_yolov5n_fp16.$$.tmp"
trap 'rm -f "${TEMP_OUTPUT}.om"' EXIT

atc --model="$ONNX" \
    --framework=5 \
    --output="$TEMP_OUTPUT" \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --soc_version="$SOC_VERSION" \
    --input_fp16_nodes="images" \
    --log=info

[[ -s "${TEMP_OUTPUT}.om" ]] || {
  echo "[ERROR] ATC returned success but produced no OM" >&2
  exit 2
}
mv -f "${TEMP_OUTPUT}.om" "$OM"
sha256sum "$OM" > "$OM.sha256"
atc --version > "$MODEL_DIR/atc_version.txt" 2>&1 || true
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info > "$MODEL_DIR/npu_smi_info.txt" 2>&1 || true
else
  echo "npu-smi unavailable during conversion" > "$MODEL_DIR/npu_smi_info.txt"
fi
printf '%s\n' "$SOC_VERSION" > "$MODEL_DIR/soc_version.txt"

echo "OM generated: $OM"
cat "$OM.sha256"
