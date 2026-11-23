#!/usr/bin/env bash
set -euo pipefail

# Reference implementation living under examples/; the project root is two levels up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPP_DIR="$ROOT_DIR"
PYTHON="${PRED_PYTHON:-/home/sgui/miniforge3/envs/didi_t213/bin/python}"
TEMP_ROOT="${OVBENCH_TEMP_ROOT:-/tmp/ovbench-$(id -u)}"
REPORT="${PRED_REPORT_DIR:-$ROOT_DIR/reports/pred_models_$(date +%Y%m%d_%H%M%S)}"
MODELS="${PRED_MODELS:-26Q2_round13_run2_e40_no_map_attn,ped_26Q1_history_data_gen4_800v2_issue2_e23_refine,vehicle_q2_release_v0_w_leading_25}"
PRECISIONS="${PRED_PRECISIONS:-FP32}"
THREADS="${PRED_THREADS:-6}"
ROUNDS="${PRED_ROUNDS:-10}"
SAMPLES="${PRED_SAMPLES:-10}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON; set PRED_PYTHON to the didi_t213-compatible environment." >&2
    exit 1
fi
if [[ -e "$REPORT" ]]; then
    echo "Report directory already exists: $REPORT" >&2
    exit 1
fi
"$PYTHON" -c 'import openvino as ov; version = ov.__version__; print("OpenVINO", version); assert version.startswith("2025.4.1"), f"Expected OpenVINO 2025.4.1, got {version}"'
mkdir -p "$TEMP_ROOT"
SCRATCH="$(mktemp -d "$TEMP_ROOT/pred-models-XXXXXXXX")"
BINARY="${PRED_BINARY:-$SCRATCH/build/ov_benchmark}"
if [[ -z "${PRED_BINARY:-}" ]]; then
    SDK="$("$PYTHON" -c 'import sys; from pathlib import Path; print(Path(sys.prefix) / "lib/cmake/openvino2025.4.1")')"
    cmake -S "$CPP_DIR" -B "$SCRATCH/build" -DOpenVINO_DIR="$SDK" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SCRATCH/build" --target ov_benchmark -j "${BUILD_JOBS:-4}"
fi

echo "Scratch directory: $SCRATCH"
echo "Report directory: $REPORT"
"$PYTHON" "$CPP_DIR/tools/prepare_pred_models.py" \
    --model-root "${PRED_MODEL_ROOT:-$ROOT_DIR/models/pred_model}" \
    --output "$SCRATCH/prepared" --samples "$SAMPLES"

COMMON=(
    --model-root "$SCRATCH/prepared"
    --manifest "$SCRATCH/prepared/manifest.json"
    --prepared-inputs
    --model "$MODELS"
    --precisions "$PRECISIONS"
    --threads "$THREADS"
    --warmup 0
    --cpu-pinning NO
    --perf-count NO
)
set +e
"$BINARY" "${COMMON[@]}" --mode accuracy --output "$REPORT/accuracy"
accuracy_status=$?
"$BINARY" "${COMMON[@]}" --mode stability --rounds "$ROUNDS" --output "$REPORT/stability"
stability_status=$?
set -e

if [[ $accuracy_status -ne 0 || $stability_status -ne 0 ]]; then
    echo "Prediction validation failed: accuracy=$accuracy_status stability=$stability_status" >&2
    exit 1
fi
echo "Accuracy and stability reports: $REPORT"
