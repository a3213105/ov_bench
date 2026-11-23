#!/usr/bin/env bash
# ov_bench SINGLE-PROCESS test item: for each selected model runs performance +
# correctness (accuracy) + MD5 stability in one native `ov_benchmark --mode all`
# pass, across the requested taskset binding modes. Persists its summary entries
# for the summary stage.
set -euo pipefail
OVB_STAGE_ACTIVE="single"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/ovbench_lib.sh"

ovb_init || exit 1
(( DRY_RUN == 1 )) && exit 0

run_single_process() {
	local log_file="$1"
	shift
	if (( SINGLE_TASKSET_BIND == 1 )); then
		taskset --cpu-list "$SINGLE_NUMA_CPU_LIST" "$@" > "$log_file" 2>&1
	else
		"$@" > "$log_file" 2>&1
	fi
}

single_output_base() {
	case "$1" in
		yolo-world) printf '%s' "yolo_world_all" ;;
		generic) printf '%s' "generic_all" ;;
		*) printf '%s' "${1}_all" ;;
	esac
}

# Build the per-suite binary invocation. Registered suites send the whole suite CSV
# in one call; the generic suite issues one call per --model-file target.
run_suite_single() {
	local suite="$1" out="$2"
	local -a cmd=()
	local model
	if [[ "$suite" == "generic" ]]; then
		IFS=',' read -r -a model_list <<< "$(trim_csv "$(suite_models "$suite")")"
		for model in "${model_list[@]}"; do
			CURRENT_SUITE="$suite"
			set_model_args "$model"; set_input_args "$suite"
			cmd=("$BINARY" --mode all "${MODEL_CMD_ARGS[@]}" "${INPUT_CMD_ARGS[@]}"
				--precisions "$PRECISIONS" -batch "$BATCH" --threads "$THREADS" --warmup "$WARMUP"
				"${PERFORMANCE_ARGS[@]}" --rounds "$ROUNDS" --cpu-pinning "$CPU_PINNING" --perf-count "$PERF_COUNT"
				--output "$out")
			run_single_process "$SINGLE_RUN_DIR/${suite}_$(model_dir_name "$model" "$suite").log" "${cmd[@]}"
		done
	else
		local csv; csv="$(trim_csv "$(suite_models "$suite")")"
		CURRENT_SUITE="$suite"
		set_model_args "$csv"; set_input_args "$suite"
		cmd=("$BINARY" --mode all "${MODEL_CMD_ARGS[@]}" "${INPUT_CMD_ARGS[@]}"
			--precisions "$PRECISIONS" -batch "$BATCH" --threads "$THREADS" --warmup "$WARMUP"
			"${PERFORMANCE_ARGS[@]}" --rounds "$ROUNDS" --cpu-pinning "$CPU_PINNING" --perf-count "$PERF_COUNT"
			--output "$out")
		run_single_process "$SINGLE_RUN_DIR/${suite}_all.log" "${cmd[@]}"
	fi
}

run_single_instance_tests() {
	local -a binding_modes=()
	case "$TASKSET_BIND" in
		0) binding_modes=(0) ;;
		1) binding_modes=(1) ;;
		2) binding_modes=(1 0) ;;
	esac
	(( MULTI_INSTANCE_ONLY == 0 )) || { log "Single-process stage skipped (multi-only mode)"; return 0; }
	local SINGLE_TASKSET_BIND
	for SINGLE_TASKSET_BIND in "${binding_modes[@]}"; do
		if (( SINGLE_TASKSET_BIND == 1 )); then
			prepare_single_numa_binding
			CPU_PINNING=YES
		else
			SINGLE_NUMA_CPU_LIST=""
			CPU_PINNING=NO
		fi
		SINGLE_RUN_DIR="$RUN_DIR/single/bind_${SINGLE_TASKSET_BIND}"
		mkdir -p "$SINGLE_RUN_DIR"
		log "SINGLE BINDING BEGIN taskset_bind=$SINGLE_TASKSET_BIND cpu_pinning=$CPU_PINNING"
		local suite out
		for suite in "${suite_list[@]}"; do
			contains_suite "$suite" || continue
			IFS=',' read -r -a n_models <<< "$(trim_csv "$(suite_models "$suite")")"
			(( ${#n_models[@]} > 0 )) || continue
			out="$SINGLE_RUN_DIR/$(single_output_base "$suite")"
			log "Running $suite suite"
			run_suite_single "$suite" "$out"
			if [[ -f "$out/summary.tsv" ]]; then
				local entry="$suite-bind_${SINGLE_TASKSET_BIND}:$out/summary.tsv"
				SUMMARY_FILES+=("$entry")
				printf '%s\n' "$entry" >> "$SINGLE_ENTRIES_FILE"
			fi
		done
		log "SINGLE BINDING END taskset_bind=$SINGLE_TASKSET_BIND"
	done
	log "Single-process summary entries: ${#SUMMARY_FILES[@]}"
}

run_single_instance_tests
exit 0
