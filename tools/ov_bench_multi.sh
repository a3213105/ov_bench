#!/usr/bin/env bash
# ov_bench MULTI-INSTANCE test item: launches waves of independent ov_bench
# workers (NUMA-group-bound or unbound), with optional synchronized starts and
# hyper-threading sweeps, then aggregates per-instance metrics.
set -euo pipefail
OVB_STAGE_ACTIVE="multi"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/ovbench_lib.sh"

ovb_init || exit 1
(( DRY_RUN == 1 )) && exit 0

run_multi_instance_model() {
	local suite="$1" model="$2" instance_count="$3" precision="$4" case_root="$5"
	local wave_dir="$case_root" ready_count start_time index cpu_list numa_node
	local -a pids=() worker_dirs=()
	local worker_dir worker_log output_dir pid rc failed input_count
	# A comma-separated --batch list runs one independent wave per batch value
	# under every synchronization mode (sync-start 0, 1, and 2 via the outer
	# MODE dispatch), each writing to its own batch_<N|dynamic> subtree. Every
	# worker in a wave is launched with a single concrete batch value.
	if [[ "$BATCH" == *,* ]]; then
		local original_batch="$BATCH" batch_value batch_dir rc_approx
		local -a batch_values=()
		IFS=',' read -r -a batch_values <<< "$BATCH"
		rc_approx=0
		for batch_value in "${batch_values[@]}"; do
			BATCH="$batch_value"
			if [[ "$batch_value" == 0 ]]; then batch_dir=batch_dynamic; else batch_dir="batch_$batch_value"; fi
			run_multi_instance_model "$suite" "$model" "$instance_count" "$precision" "$case_root/$batch_dir" || rc_approx=1
		done
		BATCH="$original_batch"
		return $rc_approx
	fi
	mkdir -p "$wave_dir"
	if (( TASKSET_BIND == 1 && ${#WORKER_CPU_LISTS[@]} < instance_count )); then
		echo "Not enough NUMA-local worker slots: need $instance_count, have ${#WORKER_CPU_LISTS[@]} at $INSTANCE_THREADS physical cores each" >&2
		return 2
	fi
	CURRENT_SUITE="$suite"
	set_model_args "$model"; set_input_args "$suite"
	input_count=N/A
	if [[ "$suite" == "pred" ]]; then
		if [[ -n "$PRED_MANIFEST" ]]; then input_count=manifest; else input_count="${INPUT_COUNT:-10}"; fi
	fi
	local -a cmd=()
	for ((index = 0; index < instance_count; index++)); do
		worker_dir="$wave_dir/instance_$(printf '%03d' $((index + 1)))"
		output_dir="$worker_dir/results"
		worker_log="$worker_dir/worker.log"
		mkdir -p "$worker_dir"
		cpu_list=""; numa_node=""
		if (( TASKSET_BIND == 1 )); then
			cpu_list="${WORKER_CPU_LISTS[$index]}"
			numa_node="${WORKER_NUMA_NODES[$index]}"
		fi
		worker_dirs+=("$worker_dir")
		cmd=(
			"$BINARY" --mode performance "${MODEL_CMD_ARGS[@]}" "${INPUT_CMD_ARGS[@]}"
			--precisions "$precision" --threads "$WAVE_THREADS"
			--warmup "$WARMUP" "${PERFORMANCE_ARGS[@]}" -batch "$BATCH"
			--cpu-pinning "$([[ "$TASKSET_BIND" == "1" ]] && echo YES || echo NO)"
			--perf-count "$PERF_COUNT" --output "$output_dir" --no-performance-output-files
		)
		if (( TASKSET_BIND == 0 )); then cmd+=(--hyper-threading "$WAVE_HYPER_THREADING"); fi
		if (( SYNC_START == 1 )); then cmd+=(--start-signal "$wave_dir/start.signal"); fi
		log "MULTI START suite=$suite model=$model precision=$precision batch=$BATCH input_count=$input_count instances=$instance_count threads_per_instance=$WAVE_THREADS base_threads=$INSTANCE_THREADS hyper_threading=$WAVE_HYPER_THREADING worker=$((index + 1)) taskset_bind=$TASKSET_BIND sync_start=$SYNC_START numa=${numa_node:-<none>} cpus=${cpu_list:-<unbound>}"
		if (( TASKSET_BIND == 1 )); then
			taskset --cpu-list "$cpu_list" "${cmd[@]}" >"$worker_log" 2>&1 &
		else
			"${cmd[@]}" >"$worker_log" 2>&1 &
		fi
		pids+=("$!")
	done
	if (( SYNC_START == 1 )); then
		ready_count=0
		start_time=$SECONDS
		while (( ready_count < instance_count )); do
			ready_count=0
			for ((index = 0; index < instance_count; index++)); do
				worker_dir="${worker_dirs[$index]}"
				if [[ -f "$worker_dir/results/ready.json" ]]; then
					((ready_count += 1))
				elif [[ -f "$worker_dir/results/results.json" ]] && grep -q '"status": "ERROR"' "$worker_dir/results/results.json"; then
					echo "Worker failed before synchronized start: $worker_dir" >&2; touch "$wave_dir/abort.signal"
					for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; return 1
				elif grep -q '^ERROR:' "$worker_dir/worker.log" 2>/dev/null; then
					echo "Worker failed before synchronized start: $worker_dir" >&2; touch "$wave_dir/abort.signal"
					for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; return 1
				elif ! kill -0 "${pids[$index]}" 2>/dev/null; then
					echo "Worker exited before synchronized start: $worker_dir" >&2; touch "$wave_dir/abort.signal"
					for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; return 1
				fi
			done
			(( ready_count == instance_count )) && break
			if (( SECONDS - start_time >= MULTI_START_TIMEOUT )); then
				echo "Timed out waiting for workers to warm up: ready=$ready_count/$instance_count" >&2; touch "$wave_dir/abort.signal"
				for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; return 1
			fi
			sleep 1
		done
		log "MULTI READY suite=$suite model=$model precision=$precision instances=$instance_count; releasing synchronized start"
		touch "$wave_dir/start.signal"
	else
		log "MULTI synchronization disabled; workers start timing immediately after their own warmup"
	fi
	failed=0
	for index in "${!pids[@]}"; do
		pid="${pids[$index]}"
		set +e; wait "$pid"; rc=$?; set -e
		if (( rc != 0 )); then log "MULTI FAIL worker=${worker_dirs[$index]} rc=$rc"; failed=1; fi
	done
	(( failed == 0 )) || return 1
	aggregate_instance_results "$case_root" "$suite" "$model" "$precision" "$instance_count" "${worker_dirs[@]}"
	log "MULTI DONE suite=$suite model=$model precision=$precision instances=$instance_count threads_per_instance=$WAVE_THREADS hyper_threading=$WAVE_HYPER_THREADING"
}

run_multi_instance_matrix() {
	local suite modelsCsv model model_output case_root
	local requested_binding="$TASKSET_BIND"
	local TASKSET_BIND="$requested_binding"
	local requested_sync="$SYNC_START"
	local SYNC_START="$requested_sync"
	local WAVE_HYPER_THREADING=DEFAULT WAVE_THREADS="$INSTANCE_THREADS"
	local -a model_list=() precision_list=() binding_modes=("$requested_binding")
	local -a ht_modes=() instance_counts=()
	local -a sync_modes=("$requested_sync")
	if (( requested_sync == 2 )); then sync_modes=(0 1); fi
	IFS=',' read -r -a precision_list <<< "$(trim_csv "$PRECISIONS")"
	if (( requested_binding == 2 )); then binding_modes=(1 0); fi
	if (( requested_binding != 0 )); then TASKSET_BIND=1; fi
	prepare_worker_cpu_assignments
	if [[ ! -f "$MULTI_SUMMARY" ]]; then
		printf 'suite\tmodel\tprecision\tbatch\tinstances\tthreads_per_instance\twarmup\titerations\tinstances_completed\ttotal_throughput_fps\tthroughput_per_instance_fps\tmax_instance_p99_ms\tstatus\ttaskset_bind\thyper_threading\tbase_threads_per_instance\tduration_seconds\ttotal_timed_inferences\tsync_start\tmean_ms\tmax_instance_elapsed_seconds\tinput_samples_per_instance\tresult_dir\n' > "$MULTI_SUMMARY"
	fi
	local count precision
	for suite in "${suite_list[@]}"; do
		contains_suite "$suite" || continue
		modelsCsv="$(trim_csv "$(suite_models "$suite")")"
		case "$suite" in
			dinov3|yolo-world|unet) : ;;
			pred) : ;;
			generic) : ;;
			*) continue ;;
		esac
		IFS=',' read -r -a model_list <<< "$modelsCsv"
		(( ${#model_list[@]} > 0 )) || continue
		for model in "${model_list[@]}"; do
			log "MULTI MODEL BEGIN suite=$suite model=$model"
			for TASKSET_BIND in "${binding_modes[@]}"; do
				instance_counts=("${MULTI_INSTANCE_LIST[@]}")
				if (( TASKSET_BIND == 1 )); then instance_counts=("${#WORKER_CPU_LISTS[@]}"); fi
				log "MULTI BINDING BEGIN suite=$suite model=$model taskset_bind=$TASKSET_BIND"
				ht_modes=(DEFAULT)
				if (( TASKSET_BIND == 0 )); then
					case "$UNBOUND_HT" in
						0) ht_modes=(NO) ;;
						1) ht_modes=(YES) ;;
						2) ht_modes=(YES NO) ;;
					esac
				fi
				for WAVE_HYPER_THREADING in "${ht_modes[@]}"; do
					WAVE_THREADS="$INSTANCE_THREADS"
					[[ "$WAVE_HYPER_THREADING" == "YES" ]] && WAVE_THREADS=$((INSTANCE_THREADS * 2))
					for count in "${instance_counts[@]}"; do
						for precision in "${precision_list[@]}"; do
							model_output="$RUN_DIR/multi_instance/$suite/$(model_dir_name "$model" "$suite")/bind_${TASKSET_BIND}"
							(( TASKSET_BIND == 0 )) && model_output+="/ht_${WAVE_HYPER_THREADING}"
							for SYNC_START in "${sync_modes[@]}"; do
								case_root="$model_output/i${count}/${precision}/sync_${SYNC_START}"
								run_multi_instance_model "$suite" "$model" "$count" "$precision" "$case_root"
							done
						done
					done
				done
				log "MULTI BINDING END suite=$suite model=$model taskset_bind=$TASKSET_BIND"
			done
			log "MULTI MODEL END suite=$suite model=$model"
		done
	done
}

run_multi_instance_matrix
exit 0
