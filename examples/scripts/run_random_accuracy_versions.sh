#!/usr/bin/env bash
set -euo pipefail

REPORT="${1:?Usage: run_random_accuracy_versions.sh NEW_REPORT_DIR}"
shift
EXTRA=("$@")
while [[ $# -gt 0 ]]; do
    option="$1"
    case "$option" in
        --resample-common-cosine-failures) ;;
        --seed)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && ${#2} -le 10 ]] || {
                echo "--seed requires an integer in [0,1000000000]" >&2; exit 2;
            }
            [[ $((10#$2)) -le 1000000000 ]] || {
                echo "--seed requires an integer in [0,1000000000]" >&2; exit 2;
            }
            shift ;;
        *) echo "Unsupported option: $option" >&2; exit 2 ;;
    esac
    shift
done
CPP=/home/sgui/2026/ov_bench
PYTHON=/home/sgui/miniforge3/envs/didi_t213/bin/python
LEGACY_SDK=/tmp/ovbench-1003/sdk-2023.3/runtime/cmake
PATCHED_SDK=/home/sgui/miniforge3/envs/didi_t213/lib/cmake/openvino2025.4.1
PLUGIN=/home/sgui/miniforge3/envs/didi_t213/lib/openvino-2025.4.1/libopenvino_intel_cpu_plugin.so
EXPECTED=a9888e0f50855db414405f9da1a8771f06f8419d575436a07589ff21d719b5c0
[[ ! -e "$REPORT" ]] || { echo "Report directory already exists: $REPORT" >&2; exit 2; }
mkdir -p "$REPORT/source"
if [[ -n "${OVBENCH_REFERENCE_CONFIG:-}" ]]; then
    cp "$OVBENCH_REFERENCE_CONFIG" "$REPORT/reference_config.json"
    export OVBENCH_REFERENCE_CONFIG
    OVBENCH_REFERENCE_CONFIG="$(realpath "$REPORT/reference_config.json")"
fi
exec >"$REPORT/driver.log" 2>&1
echo "$$" >"$REPORT/driver.pid"
echo RUNNING >"$REPORT/status"
date -Iseconds >"$REPORT/started_at"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo ERROR >"$REPORT/status"; echo "$rc" >"$REPORT/driver_exit_code"; fi' EXIT
printf '%s  %s\n' "$EXPECTED" "$PLUGIN" | sha256sum --check
sha256sum "$PLUGIN" >"$REPORT/patched_plugin.sha256"
cp -a "$CPP/src" "$CPP/include" "$CPP/tests" "$CPP/tools" "$CPP/CMakeLists.txt" \
    "$CPP/examples/scripts/run_random_accuracy_background.sh" "$REPORT/source/"
find "$REPORT/source" -type f ! -path '*/__pycache__/*' -exec sha256sum {} + >"$REPORT/source.sha256"
env | grep -E '^(ONEDNN|DNNL|OV_|OMP_|MKL_|OPENBLAS_|LD_LIBRARY_PATH)' >"$REPORT/runtime_environment.txt" || true
for version in 2023.3 2025.4.1-patched; do
    sdk="$LEGACY_SDK"
    if [[ "$version" == 2025.4.1-patched ]]; then
        sdk="$PATCHED_SDK"
        printf '%s  %s\n' "$EXPECTED" "$PLUGIN" | sha256sum --check
    fi
    echo "START $version $(date -Iseconds)"
    OVBENCH_SOURCE_DIR="$REPORT/source" OVBENCH_OPENVINO_DIR="$sdk" \
        OVBENCH_REFERENCE_PROJECT_ROOT="$(dirname "$CPP")" \
        OVBENCH_REFERENCE_PYTHON="$PYTHON" \
        bash "$REPORT/source/run_random_accuracy_background.sh" "$REPORT/$version" "${EXTRA[@]}"
    echo "END $version $(date -Iseconds)"
done
"$PYTHON" "$REPORT/source/tools/summarize_random_accuracy_versions.py" "$REPORT"
date -Iseconds >"$REPORT/finished_at"
echo 0 >"$REPORT/driver_exit_code"
