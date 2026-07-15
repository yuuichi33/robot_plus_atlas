#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/../models" && pwd)"

cd "$MODEL_DIR"
sha256sum -c SOURCE_SHA256SUMS.txt
[[ -s task_yolov5n_fp16.om ]] || {
  echo "[ERROR] task_yolov5n_fp16.om is missing; run scripts/convert_atc.sh" >&2
  exit 2
}
sha256sum -c task_yolov5n_fp16.om.sha256
