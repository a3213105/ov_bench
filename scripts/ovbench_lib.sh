#!/usr/bin/env bash
# Shared library sourced by the ov_bench test-entrypoint scripts under tools/.
# It is not an entry point itself. Source it AFTER setting OVB_STAGE, and it will
# parse "$@" (options) into the shared globals and load the helper libraries.
#
#   OVB_STAGE=<single|corescaling|multi|summary>
#   source "$(dirname ...)/scripts/ovbench_lib.sh"
#   ovb_init            # validate, prepare RUN_DIR and shared paths
#   <run that stage>

OV_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OV_BENCH_ROOT="$(dirname "$OV_LIB_DIR")"
CPP_DIR="$OV_BENCH_ROOT"
PROJECT_ROOT="$(dirname "$OV_BENCH_ROOT")"
export OV_BENCH_ROOT CPP_DIR PROJECT_ROOT

# Which orchestration stage this invocation owns. Callers (each item entrypoint
# and the all-items orchestrator) set it before sourcing this library; it always
# defaults to "all" so it is never unbound.
OVB_STAGE_ACTIVE="${OVB_STAGE_ACTIVE:-all}"
export OVB_STAGE_ACTIVE

source "$OV_LIB_DIR/parameters.sh"
source "$OV_LIB_DIR/validate.sh"
source "$OV_LIB_DIR/numa.sh"
source "$OV_LIB_DIR/summary.sh"

log() {
	local msg="$1"
	if [[ -n "${MASTER_LOG:-}" && -d "$(dirname "$MASTER_LOG")" ]]; then
		printf '[%s] %s\n' "$(date +%F' '%T)" "$msg" | tee -a "$MASTER_LOG"
	else
		printf '[%s] %s\n' "$(date +%F' '%T)" "$msg" >&2
	fi
}

# Whether the multi-instance stage is responsible in the current invocation.
ovb_multi_active() {
	if [[ "$OVB_STAGE_ACTIVE" == "multi" ]]; then return 0; fi
	if [[ "$OVB_STAGE_ACTIVE" == "all" && ( "$MODE" == "multi" || "$MODE" == "all" ) ]]; then return 0; fi
	return 1
}

# Populate MODEL_CMD_ARGS with the model-selection arguments for the current
# CURRENT_SUITE / model. Registered suites pass --model/--model-root, while the
# synthetic "generic" suite rewrites unknown models to --model-file (plus any
# --input-shape overrides captured from the command line).
set_model_args() {
	local model="$1"
	MODEL_CMD_ARGS=()
	if [[ "$CURRENT_SUITE" == "generic" ]]; then
		MODEL_CMD_ARGS=(--model-file "$model")
		local shape
		for shape in "${GENERIC_INPUT_SHAPES[@]}"; do MODEL_CMD_ARGS+=(--input-shape "$shape"); done
	else
		if [[ "$CURRENT_SUITE" == "pred" ]]; then
			MODEL_CMD_ARGS=(--model "$model" --model-root "$MODEL_ROOT_PRED")
		else
			MODEL_CMD_ARGS=(--model "$model" --model-root "$MODEL_ROOT_IMAGES")
		fi
	fi
}

# Populate INPUT_CMD_ARGS with the per-suite input-source arguments. Generic models
# always rely on the natively generated random tensors and therefore add nothing
# here (no --images, no manifests, no --input-count).
set_input_args() {
	local suite="$1"
	INPUT_CMD_ARGS=()
	case "$suite" in
		generic) : ;;
		pred)
			INPUT_CMD_ARGS=(--images "$IMAGES_DIR")
			if [[ -n "$PRED_MANIFEST" ]]; then
				INPUT_CMD_ARGS+=(--manifest "$PRED_MANIFEST" --prepared-inputs)
			elif [[ -n "$INPUT_COUNT" ]]; then
				INPUT_CMD_ARGS+=(--input-count "$INPUT_COUNT")
			fi
			;;
		dinov3|yolo-world|unet)
			INPUT_CMD_ARGS=(--images "$IMAGES_DIR")
			if [[ ( "$suite" == "dinov3" || "$suite" == "yolo-world" ) && -n "$IMAGE_MANIFEST" ]]; then
				INPUT_CMD_ARGS+=(--manifest "$IMAGE_MANIFEST")
			fi
			;;
		*) echo "Unknown suite: $suite" >&2; return 2 ;;
	esac
}

# Emit the CSV of selected models for a given suite (used by the stage runners).
suite_models() {
	local suite="$1"
	case "$suite" in
		dinov3) printf '%s' "$DINO_MODELS" ;;
		yolo-world) printf '%s' "$YOLO_MODELS" ;;
		pred) printf '%s' "$PRED_MODELS" ;;
		unet) printf '%s' "$UNET_MODELS" ;;
		generic) printf '%s' "$MODEL_FILES_CSV" ;;
	esac
}

# A filesystem-safe directory name for a model identifier or file path.
model_dir_name() {
	local model="$1" suite="${2:-}"
	if [[ "$suite" == "generic" ]]; then
		local base="${model##*/}"
		printf '%s' "${base//[^A-Za-z0-9._-]/_}"
	else
		printf '%s' "${model//:/_}"
	fi
}

ovb_dry_run() {
	echo "[DRY-RUN] STAGE=$OVB_STAGE_ACTIVE"
	echo "[DRY-RUN] RUN_DIR=$RUN_DIR"
	echo "[DRY-RUN] BINARY=$BINARY"
	echo "[DRY-RUN] MODEL_FILES=$MODEL_FILES_CSV"
	echo "[DRY-RUN] SUITES=$SUITES DINO=$DINO_MODELS YOLO=$YOLO_MODELS PRED=$PRED_MODELS UNET=$UNET_MODELS"
	echo "[DRY-RUN] PRECISIONS=$PRECISIONS THREADS=$THREADS WARMUP=$WARMUP ITERATIONS=$ITERATIONS ROUNDS=$ROUNDS"
	echo "[DRY-RUN] BATCH=$BATCH MODE=$MODE TASKSET_BIND=$TASKSET_BIND INSTANCES=$MULTI_INSTANCE_COUNTS CORE_SCALING=$CORE_SCALING_COUNTS"
}

# Validate the current stage, prepare RUN_DIR and the shared summary paths.
ovb_init() {
	ovb_validate
	if (( DRY_RUN == 1 )); then
		ovb_dry_run
		return 0
	fi
	SUMMARY_ALL="$RUN_DIR/summary_all.tsv"
	MULTI_SUMMARY="$RUN_DIR/multi_instance_summary.tsv"
	CORE_SCALING_SUMMARY="$RUN_DIR/core_scaling_summary.tsv"
	SINGLE_ENTRIES_FILE="$RUN_DIR/single_summary_entries.tsv"
	SUMMARY_FILES=()

	if [[ "${OVB_STAGE_ACTIVE:-}" == "summary" ]]; then
		# Merge an existing run: reuse the persisted run configuration so the unified
		# summary columns match what the items actually recorded.
		if [[ -f "$RUN_DIR/ovb_config.env" ]]; then source "$RUN_DIR/ovb_config.env"; fi
		if [[ -f "$SINGLE_ENTRIES_FILE" ]]; then mapfile -t SUMMARY_FILES < "$SINGLE_ENTRIES_FILE"; fi
		return 0
	fi

	[[ -x "$BINARY" ]] || { echo "ov_bench binary not executable: $BINARY" >&2; return 1; }
	mkdir -p "$RUN_DIR"
	MASTER_LOG="$RUN_DIR/master.log"
	touch "$MASTER_LOG"
	# Persist the run configuration for the summary stage / reruns.
	{
		printf '# ov_bench run configuration (sourced by the summary stage)\n'
		printf 'THREADS="%s"\n' "$THREADS"
		printf 'WARMUP="%s"\n' "$WARMUP"
		printf 'ITERATIONS="%s"\n' "$ITERATIONS"
		printf 'DURATION="%s"\n' "$DURATION"
		printf 'PERF_COUNT="%s"\n' "$PERF_COUNT"
		printf 'ROUNDS="%s"\n' "$ROUNDS"
		printf 'CPU_PINNING="%s"\n' "$CPU_PINNING"
		printf 'BATCH="%s"\n' "$BATCH"
		printf 'PRECISIONS="%s"\n' "$PRECISIONS"
	} > "$RUN_DIR/ovb_config.env"
	log "STAGE=$OVB_STAGE_ACTIVE RUN_DIR=$RUN_DIR"
	log "BINARY=$BINARY"
	log "MODEL_FILES=$MODEL_FILES_CSV SUITES=$SUITES PRECISIONS=$PRECISIONS THREADS=$THREADS WARMUP=$WARMUP BATCH=$BATCH"
	return 0
}
