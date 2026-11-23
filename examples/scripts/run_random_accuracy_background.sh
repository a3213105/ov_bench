#!/usr/bin/env bash
set -euo pipefail

REPORT="${1:?Usage: run_random_accuracy_background.sh NEW_REPORT_DIR}"
shift
SAMPLING=()
SEED=43
while [[ $# -gt 0 ]]; do
    option="$1"
    case "$option" in
        --resample-common-cosine-failures) SAMPLING+=("$option") ;;
        --seed)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && ${#2} -le 10 ]] || {
                echo "--seed requires an integer in [0,1000000000]" >&2; exit 2;
            }
            SEED="$2"
            [[ $((10#$SEED)) -le 1000000000 ]] || {
                echo "--seed requires an integer in [0,1000000000]" >&2; exit 2;
            }
            shift ;;
        *) echo "Unsupported option: $option" >&2; exit 2 ;;
    esac
    shift
done
CPP="${OVBENCH_SOURCE_DIR:-/home/sgui/2026/ov_bench}"
PYTHON="${OVBENCH_REFERENCE_PYTHON:-/home/sgui/miniforge3/envs/didi_t213/bin/python}"
SDK="${OVBENCH_OPENVINO_DIR:-/home/sgui/miniforge3/envs/didi_t213/lib/cmake/openvino2025.4.1}"
MODE="${OVBENCH_TEST_MODE:-random-accuracy}"
case "$MODE" in
    random-accuracy) PRECISIONS=FP32,BF16,FP16; EXTRA=(
        --atol FP32=1e-5,BF16=1e-3,FP16=1e-4 --rtol FP32=1e-4,BF16=1e-2,FP16=1e-3) ;;
    tolerance-search) PRECISIONS=FP32; EXTRA=() ;;
    *) echo "Unsupported test mode: $MODE" >&2; exit 2 ;;
esac
if [[ ${#SAMPLING[@]} -gt 0 && "$MODE" != random-accuracy ]]; then
    echo "--resample-common-cosine-failures requires random-accuracy" >&2
    exit 2
fi
if [[ -n "${OVBENCH_REFERENCE_CONFIG:-}" ]]; then
    [[ -f "$OVBENCH_REFERENCE_CONFIG" ]] || { echo "Reference config does not exist: $OVBENCH_REFERENCE_CONFIG" >&2; exit 2; }
    EXTRA+=(--reference-config "$OVBENCH_REFERENCE_CONFIG")
fi
[[ ! -e "$REPORT" ]] || { echo "Report directory already exists: $REPORT" >&2; exit 2; }
mkdir -p "$REPORT/logs"
exec >"$REPORT/driver.log" 2>&1
echo "Driver PID: $$"
echo "$$" >"$REPORT/driver.pid"
date -Iseconds >"$REPORT/started_at"
echo RUNNING >"$REPORT/status"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo ERROR >"$REPORT/status"; echo "$rc" >"$REPORT/driver_exit_code"; fi' EXIT

SCRATCH="$(mktemp -d /tmp/ovbench-$(id -u)/random-background-XXXXXXXX)"
echo "$SCRATCH" >"$REPORT/scratch_path"
cmake -S "$CPP" -B "$SCRATCH/build" \
    -DOpenVINO_DIR="$SDK" \
    -DCMAKE_BUILD_TYPE=Release >"$REPORT/logs/configure.log" 2>&1
cmake --build "$SCRATCH/build" --target ov_benchmark -j 8 >"$REPORT/logs/build.log" 2>&1
"$SCRATCH/build/ov_benchmark" --mode self-test >"$REPORT/logs/native_tests.log" 2>&1
OVBENCH_BINARY="$SCRATCH/build/ov_benchmark" "$PYTHON" -m unittest discover \
    -s "$CPP/tests" -p 'test_random*.py' -v >"$REPORT/logs/unit_tests.log" 2>&1

run_node() {
    local node="$1"
    shift
    local threads model safe rc
    threads="$(lscpu -p=CORE,NODE | awk -F, -v node="$node" \
        '$0 !~ /^#/ && $2 == node { cores[$1] = 1 } END { print length(cores) }')"
    [[ "$threads" =~ ^[1-9][0-9]*$ ]] || return 2
    echo "node=$node threads=$threads models=$*"
    for model in "$@"; do
        safe="${model//:/_}"
        echo "START $model node=$node threads=$threads $(date -Iseconds)" >>"$REPORT/progress.log"
        rc=0
        OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" \
        timeout --signal=TERM --kill-after=30s 2h \
        numactl --cpunodebind="$node" --membind="$node" "$SCRATCH/build/ov_benchmark" \
            --mode "$MODE" --model "$model" \
            --model-root /home/sgui/2026/models_ov \
            --reference-root /home/sgui/2026/models \
            --reference-python "$PYTHON" \
            --precisions "$PRECISIONS" --iterations 100 --seed "$SEED" "${EXTRA[@]}" "${SAMPLING[@]}" \
            --threads "$threads" --cpu-pinning NO --perf-count NO \
            --threshold 0.99 --step-timeout-seconds 1800 \
            --output "$REPORT/$safe" >"$REPORT/logs/$safe.log" 2>&1 || rc=$?
        echo "$rc" >"$REPORT/logs/$safe.exit_code"
        echo "END $model node=$node exit=$rc $(date -Iseconds)" >>"$REPORT/progress.log"
    done
}

run_node 1 \
    dinov3-convnext-tiny-pretrain-lvd1689m \
    v2.1:640 \
    v2:640 \
    26Q2_round13_run2_e40_no_map_attn >"$REPORT/logs/node1.log" 2>&1 &
p1=$!
run_node 2 \
    dinov3-vits16-pretrain-lvd1689m \
    v2.1:1280 \
    ped_26Q1_history_data_gen4_800v2_issue2_e23_refine >"$REPORT/logs/node2.log" 2>&1 &
p2=$!
run_node 3 \
    dinov3-vits16plus-pretrain-lvd1689m \
    v2:1280 \
    vehicle_q2_release_v0_w_leading_25 >"$REPORT/logs/node3.log" 2>&1 &
p3=$!
echo "$p1 $p2 $p3" >"$REPORT/worker.pids"
worker_failed=0
for pid in "$p1" "$p2" "$p3"; do
    wait "$pid" || worker_failed=1
done
if [[ "$worker_failed" -ne 0 ]]; then
    echo "A node worker exited unexpectedly" >&2
    exit 2
fi
"$PYTHON" "$CPP/tools/summarize_random_accuracy.py" "$REPORT"
date -Iseconds >"$REPORT/finished_at"
echo 0 >"$REPORT/driver_exit_code"
