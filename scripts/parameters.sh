#!/usr/bin/env bash
BINARY="${BINARY:-}"
IR_ROOT="${IR_ROOT:-$PROJECT_ROOT/models_ov}"
MODEL_ROOT_PRED="${MODEL_ROOT_PRED:-$PROJECT_ROOT/models}"
IMAGES_DIR="${IMAGES_DIR:-$PROJECT_ROOT/test_images}"
IMAGE_MANIFEST="${IMAGE_MANIFEST:-}"
PRED_MANIFEST="${PRED_MANIFEST:-}"
DINO_MODELS="${DINO_MODELS:-dinov3-convnext-tiny-pretrain-lvd1689m,dinov3-vits16-pretrain-lvd1689m,dinov3-vits16plus-pretrain-lvd1689m}"
YOLO_MODELS="${YOLO_MODELS:-v2.1:640,v2.1:1280,v2:640,v2:1280}"
PRED_MODELS="${PRED_MODELS:-26Q2_round13_run2_e40_no_map_attn,ped_26Q1_history_data_gen4_800v2_issue2_e23_refine,vehicle_q2_release_v0_w_leading_25}"
UNET_MODELS="${UNET_MODELS:-unet_carvana_scale0.5_epoch2,unet_carvana_scale1.0_epoch2}"
SUITES="${SUITES:-dinov3,yolo-world,pred,unet}"
SUITES_EXPLICIT=0
MODEL_FILES_CSV=""
GENERIC_INPUT_SHAPES=()
PRECISIONS="${PRECISIONS:-FP32,BF16,FP16}"
THREADS="${THREADS:-12}"
BATCH="${BATCH:-0}"
WARMUP="${WARMUP:-1}"
ITERATIONS_EXPLICIT=0
if [[ -n "${ITERATIONS:-}" ]]; then ITERATIONS_EXPLICIT=1; fi
ITERATIONS="${ITERATIONS:-100}"
DURATION="${DURATION:-}"
DURATION_EXPLICIT=0
if [[ -n "$DURATION" ]]; then DURATION_EXPLICIT=1; fi
ROUNDS="${STABILITY_ROUNDS:-10}"
PERF_COUNT="${PERF_COUNT:-NO}"
MODE="${MODE:-single}"
MULTI_INSTANCE_COUNTS="${INSTANCES:-}"
# Core-scaling is part of the default single-process test set; disable it with
# --no-core-scaling or override the physical-core list with --core-scaling CSV.
CORE_SCALING_COUNTS="${CORE_SCALING_COUNTS:-1,2,4,8}"
INPUT_COUNT="${INPUT_COUNT:-}"
MULTI_START_TIMEOUT="${START_TIMEOUT:-1800}"
SYNC_START="${SYNC_START:-0}"
TASKSET_BIND="${TASKSET_BIND:-1}"
SINGLE_NUMA_NODE_OPTION="${SINGLE_NUMA_NODE_OPTION:-last}"
UNBOUND_HT="${HYPER_THREADING:-2}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/reports/cpp_all_models_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN=0
usage() {
	cat <<'EOF'
Usage: <ov_bench test entrypoint> --binary FILE [options]
	--mode single|multi|all       default: single
	--suites CSV                  dinov3,yolo-world,pred,unet (or: generic) by default
	--model-file FILE[,FILE...]    Arbitrary ONNX/XML model(s); selects the generic suite
	--input-shape NAME=DIMS        Optional dynamic input shape for generic models (repeatable)
	--precisions CSV              FP32,BF16,FP16 by default
	--threads N                   OV threads per process (default: 12);
																bound multi workers use N physical cores + all siblings
	--batch 0|N[,0|N...]         Dynamic batch (0) or static batch sizes; default: 0
	--instances CSV               Required for unbound runs (taskset-bind 0 or 2); ignored with 1
	--core-scaling CSV            Single-process performance sweep by physical core count
	                                                   (default: 1,2,4,8, run alongside the regular
	                                                   single-process item in single/all modes)
	--no-core-scaling             Disable the default core-scaling sweep
	--input-count N                Prediction random tensors per process (default: 10; independent of taskset binding)
	--taskset-bind 0|1|2          Single: 0=unbound, 1=last available NUMA node, 2=both; multi keeps its existing policy (default: 1)
	--single-numa-node N|last     NUMA node used by bound single/core-scaling tests (default: last)
	--hyper-threading 0|1|2       Unbound only: 0=OFF, 1=ON (threads x2), 2=both ON then OFF (default: 2)
	--warmup N                    Untimed inference count; also cooldown count (default: 1)
	--iterations N                Timed inference count (default: 100)
	--duration SECONDS            Timed inference duration instead of --iterations; exclusive
	--stability-rounds N          Single-process MD5 repeats (default: 10)
	--start-timeout N             Multi warmup barrier timeout in seconds (default: 1800)
	--sync-start 0|1|2            0: independent; 1: synchronized; 2: both 0 then 1 (default: 0)
	--perf-count YES|NO           OV profiling (default: NO)
	--ir-root DIR                 DINO/YOLO/UNet IR root (default: ../models_ov)
	--model-root DIR              Prediction ONNX root (default: ../models)
	--images-dir FILE_OR_DIR      Input images (default: ../test_images)
	--image-manifest FILE         Optional independent image reference
	--pred-manifest FILE          Optional prepared prediction input/reference
	--dinov3-models CSV           DINO model selection
	--yolo-world-models CSV       YOLO model selection
	--pred-models CSV             Prediction model selection
	--unet-models CSV             UNet model selection
	--run-dir DIR                 Report directory
	--dry-run                     Validate and print settings without execution
	-h, --help                    Show help
EOF
}

while (($#)); do
	case "$1" in
		--dry-run) DRY_RUN=1; shift; continue ;;
			--no-core-scaling) CORE_SCALING_COUNTS=""; shift; continue ;;
			--taskset-bind=*) TASKSET_BIND="${1#*=}"; shift; continue ;;
			--numactl-bind=*) TASKSET_BIND="${1#*=}"; shift; continue ;;
			--single-numa-node=*) SINGLE_NUMA_NODE_OPTION="${1#*=}"; shift; continue ;;
			--model-file=*) MODEL_FILES_CSV="${1#*=}"; shift; continue ;;
			--input-shape=*) GENERIC_INPUT_SHAPES+=("${1#*=}"); shift; continue ;;
			--core-scaling=*) CORE_SCALING_COUNTS="${1#*=}"; shift; continue ;;
		-h|--help) usage; exit 0 ;;
	esac
	(($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
	[[ "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
	case "$1" in
		--binary) BINARY="$2" ;;
		--mode) MODE="$2" ;;
		--run-dir) RUN_DIR="$2" ;;
		--ir-root) IR_ROOT="$2" ;;
		--model-root) MODEL_ROOT_PRED="$2" ;;
		--images-dir) IMAGES_DIR="$2" ;;
		--image-manifest) IMAGE_MANIFEST="$2" ;;
		--pred-manifest) PRED_MANIFEST="$2" ;;
		--suites) SUITES="$2"; SUITES_EXPLICIT=1 ;;
		--model-file) MODEL_FILES_CSV="$2" ;;
		--input-shape) GENERIC_INPUT_SHAPES+=("$2") ;;
		--dinov3-models) DINO_MODELS="$2" ;;
		--yolo-world-models) YOLO_MODELS="$2" ;;
		--pred-models) PRED_MODELS="$2" ;;
		--unet-models) UNET_MODELS="$2" ;;
		--precisions) PRECISIONS="$2" ;;
		--threads) THREADS="$2" ;;
		--batch) BATCH="$2" ;;
		--warmup) WARMUP="$2" ;;
		--iterations) ITERATIONS="$2"; ITERATIONS_EXPLICIT=1 ;;
		--duration) DURATION="$2"; DURATION_EXPLICIT=1 ;;
		--stability-rounds) ROUNDS="$2" ;;
		--perf-count) PERF_COUNT="$2" ;;
		--instances) MULTI_INSTANCE_COUNTS="$2" ;;
		--core-scaling) CORE_SCALING_COUNTS="$2" ;;
		--input-count) INPUT_COUNT="$2" ;;
		--taskset-bind|--numactl-bind) TASKSET_BIND="$2" ;;
		--single-numa-node) SINGLE_NUMA_NODE_OPTION="$2" ;;
		--hyper-threading) UNBOUND_HT="$2" ;;
		--start-timeout) MULTI_START_TIMEOUT="$2" ;;
		--sync-start) SYNC_START="$2" ;;
		*) echo "Unsupported option: $1; see --help for the current interface" >&2; exit 2 ;;
	esac
	shift 2
done

MODEL_ROOT_IMAGES="$IR_ROOT"
MODEL_ROOT_UNET="$IR_ROOT"
INSTANCE_THREADS="$THREADS"
CPU_PINNING=NO
MULTI_INSTANCE_ONLY=0
[[ "$MODE" != multi ]] || MULTI_INSTANCE_ONLY=1
AVAILABLE_CPUS=()
WORKER_CPU_LISTS=()
WORKER_NUMA_NODES=()
NODE_NUMBERS=()
declare -A NODE_CPU_LISTS=() AVAILABLE_CPU_SET=()

# Normalize the arbitrary-model (--model-file) selection and reconcile it with
# the "generic" suite. When files are supplied and no explicit --suites was given,
# only the generic suite runs; otherwise ensure "generic" is (not) present as needed.
MODEL_FILES=()
if [[ -n "$MODEL_FILES_CSV" ]]; then
	IFS=',' read -r -a MODEL_FILES <<< "${MODEL_FILES_CSV// /}"
fi
_suites_clean=",$(echo "$SUITES" | tr -d ' '),"
if [[ ${#MODEL_FILES[@]} -gt 0 ]]; then
	if [[ "$SUITES_EXPLICIT" -eq 0 ]]; then
		SUITES="generic"
	elif [[ "$_suites_clean" != *",generic,"* ]]; then
		SUITES="${SUITES:+$SUITES,}generic"
	fi
else
	_out=""
	IFS=',' read -r -a _suites_arr <<< "$(echo "$SUITES" | tr -d ' ')"
	for _s in "${_suites_arr[@]}"; do
		[[ "$_s" == "generic" || -z "$_s" ]] && continue
		[[ -n "$_out" ]] && _out+=","
		_out="${_out}${_s}"
	done
	SUITES="${_out:-dinov3,yolo-world,pred,unet}"
fi
