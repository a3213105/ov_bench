#!/usr/bin/env bash
# ov_bench CORE-SCALING test item: single-process performance swept by physical-core
# count (taskset-bound), replacing the regular single-process stage when requested.
set -euo pipefail
OVB_STAGE_ACTIVE="corescaling"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/ovbench_lib.sh"

ovb_init || exit 1
(( DRY_RUN == 1 )) && exit 0
(( ${#CORE_SCALING_LIST[@]} > 0 )) || { echo "--core-scaling is required for this item" >&2; exit 2; }

run_core_scaling_tests() {
	local suite modelsCsv model model_root model_dir precision count cpu_list output_dir log_file test_threads
	local -a precision_list=() input_args=() binding_modes=()
	case "$TASKSET_BIND" in
		0) binding_modes=(0) ;;
		1) binding_modes=(1) ;;
		2) binding_modes=(1 0) ;;
	esac
	IFS=',' read -r -a precision_list <<< "$(trim_csv "$PRECISIONS")"
	printf 'suite\ttaskset_bind\tmodel\tprecision\tbatch\tphysical_cores\tcpu_list\tthreads\teffective_threads\timages\tmean_ms\tp99_ms\tthroughput_fps\ttimed_inferences\telapsed_seconds\tstatus\tresult_dir\n' > "$CORE_SCALING_SUMMARY"

	local SINGLE_TASKSET_BIND
	for SINGLE_TASKSET_BIND in "${binding_modes[@]}"; do
		if (( SINGLE_TASKSET_BIND == 1 )); then
			prepare_single_numa_binding
			prepare_core_scaling_cpu_lists "$SINGLE_NUMA_NODE"
			CPU_PINNING=YES
		else
			CORE_SCALING_CPU_LISTS=()
			CPU_PINNING=NO
		fi
		for suite in "${suite_list[@]}"; do
			contains_suite "$suite" || continue
			modelsCsv="$(trim_csv "$(suite_models "$suite")")"
			case "$suite" in
				dinov3|yolo-world|unet) model_root="$MODEL_ROOT_IMAGES" ;;
				pred) model_root="$MODEL_ROOT_PRED" ;;
				generic) model_root="" ;;
				*) continue ;;
			esac
			IFS=',' read -r -a model_list <<< "$modelsCsv"
			(( ${#model_list[@]} > 0 )) || continue
			for model in "${model_list[@]}"; do
				model_dir="$(model_dir_name "$model" "$suite")"
				CURRENT_SUITE="$suite"
				set_model_args "$model"; set_input_args "$suite"
				for precision in "${precision_list[@]}"; do
					for count in "${CORE_SCALING_LIST[@]}"; do
						test_threads="$count"
						cpu_list=UNBOUND
						(( SINGLE_TASKSET_BIND == 1 )) && cpu_list="${CORE_SCALING_CPU_LISTS[$count]}"
						output_dir="$RUN_DIR/core_scaling/bind_${SINGLE_TASKSET_BIND}/$suite/$model_dir/cores_${count}/$precision"
						log_file="$RUN_DIR/core_scaling/bind_${SINGLE_TASKSET_BIND}/$suite/$model_dir/cores_${count}/${precision}.log"
						mkdir -p "$(dirname "$log_file")"
						cmd=("$BINARY" --mode performance "${MODEL_CMD_ARGS[@]}" "${INPUT_CMD_ARGS[@]}"
							--precisions "$precision" --threads "$test_threads" -batch "$BATCH"
							--warmup "$WARMUP" "${PERFORMANCE_ARGS[@]}" --cpu-pinning "$CPU_PINNING"
							--perf-count "$PERF_COUNT" --output "$output_dir" --no-performance-output-files)
						log "CORE SCALING START suite=$suite model=$model precision=$precision physical_cores=$count threads=$test_threads taskset_bind=$SINGLE_TASKSET_BIND cpus=$cpu_list"
						if (( SINGLE_TASKSET_BIND == 1 )); then
							taskset --cpu-list "$cpu_list" "${cmd[@]}" > "$log_file" 2>&1
						else
							"${cmd[@]}" > "$log_file" 2>&1
						fi
						awk -F '\t' -v OFS='\t' -v suite="$suite" -v binding="$SINGLE_TASKSET_BIND" -v cores="$count" \
							-v cpus="$cpu_list" -v threads="$test_threads" -v result_dir="$output_dir" \
							-v configured_batch="$BATCH" '
							function field(name) { return name in columns ? $(columns[name]) : "" }
							function na(value) { return value == "" ? "N/A" : value }
							NR == 1 { for (i = 1; i <= NF; i++) columns[$i] = i; next }
							NF {
								batch = field("batch")
								if (batch == "") batch = configured_batch
								print na(suite), na(binding), na(field("model")), na(field("precision")), na(batch), na(cores), na(cpus),
									na(threads), na(field("effective_threads")), na(field("images")), na(field("mean_ms")),
									na(field("p99_ms")), na(field("throughput_fps")), na(field("timed_inferences")),
									na(field("elapsed_seconds")), na(field("status")), na(result_dir)
							}' "$output_dir/summary.tsv" >> "$CORE_SCALING_SUMMARY"
						log "CORE SCALING DONE suite=$suite model=$model precision=$precision physical_cores=$count"
					done
				done
			done
		done
	done
	log "Core-scaling summary: $CORE_SCALING_SUMMARY"
}

run_core_scaling_tests
exit 0
