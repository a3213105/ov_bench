#!/usr/bin/env bash

append_unified_summary() {
	local test_type="$1" suite="$2" source="$3"
	awk -F '\t' -v OFS='\t' -v kind="$test_type" -v suite_name="$suite" \
		-v source_summary="$source" -v threads="$THREADS" -v warmup="$WARMUP" \
		-v iterations="$ITERATIONS" -v duration="$DURATION" -v profiling="$PERF_COUNT" \
		-v stability_rounds="$ROUNDS" -v pinning="$CPU_PINNING" -v configured_batch="$BATCH" '
		function field(name) { return name in columns ? $(columns[name]) : "" }
		function na(value) { return value == "" ? "N/A" : value }
		function parent(path) { sub(/\/[^/]*$/, "", path); return path }
		NR == 1 {
			for (column_index = 1; column_index <= NF; column_index++) columns[$column_index] = column_index
			next
		}
		NF {
			if (kind == "single") {
				suite = suite_name
				binding = 0
				if (suite_name ~ /-bind_[01]$/) {
					binding = substr(suite_name, length(suite_name), 1)
					sub(/-bind_[01]$/, "", suite)
				}
				instances = completed = 1
				base_threads = worker_threads = threads
				effective_threads = field("effective_threads")
				synchronization = 0
				ht = "DEFAULT"
				cpu_pin = binding == 1 ? "YES" : "NO"
				warmup_calls = cooldown_calls = warmup
				requested_iterations = iterations
				requested_duration = duration
				calls = field("timed_inferences")
				elapsed = field("elapsed_seconds")
				mean = field("mean_ms")
				p99 = field("p99_ms")
				fps = per_instance_fps = field("throughput_fps")
				images = field("images")
				accuracy = field("accuracy_passed")
				md5 = field("md5_consistent")
				finite = field("stable_outputs_finite")
				rounds = stability_rounds
				model_dir = field("model")
				gsub(/:/, "_", model_dir)
				batch = field("batch")
				if (batch == "") batch = configured_batch
				batch_dir = field("batch_directory")
				if (batch_dir == "" && field("batch") != "")
					batch_dir = field("batch") == 0 ? "batch_dynamic" : "batch_" field("batch")
				result_dir = parent(source_summary) "/" (batch_dir == "" ? "" : batch_dir "/") \
					model_dir "/" field("precision")
			} else {
				suite = field("suite")
				instances = field("instances")
				completed = field("instances_completed")
				base_threads = field("base_threads_per_instance")
				worker_threads = field("threads_per_instance")
				effective_threads = worker_threads
				binding = field("taskset_bind")
				synchronization = field("sync_start")
				ht = field("hyper_threading")
				cpu_pin = binding == 1 ? "YES" : "NO"
				warmup_calls = cooldown_calls = field("warmup")
				requested_iterations = field("iterations")
				requested_duration = field("duration_seconds")
				calls = field("total_timed_inferences")
				elapsed = field("max_instance_elapsed_seconds")
				mean = field("mean_ms")
				p99 = field("max_instance_p99_ms")
				fps = field("total_throughput_fps")
				per_instance_fps = field("throughput_per_instance_fps")
				images = field("input_samples_per_instance")
				accuracy = md5 = finite = rounds = ""
				result_dir = field("result_dir")
				batch = field("batch")
				if (batch == "") batch = configured_batch
			}
			measurement = requested_duration != "" ? "duration" : "iterations"
			throughput_basis = measurement == "duration" ? "wall_clock" : "inference_only"
			# Empty cells (e.g. accuracy/MD5 for performance-only rows, the inactive
			# measurement budget) are reported as N/A instead of left blank.
			print na(kind), na(suite), na(field("model")), na(field("precision")), na(batch), na(binding), na(cpu_pin),
				na(ht), na(synchronization), na(instances), na(completed), na(base_threads), na(worker_threads), na(effective_threads),
				na(warmup_calls), na(cooldown_calls), na(measurement), na(requested_iterations), na(requested_duration),
				na(profiling), na(rounds), na(images), na(calls), na(elapsed), na(mean), na(p99), na(fps), na(per_instance_fps),
				na(throughput_basis), na(accuracy), na(md5), na(finite), na(field("status")), na(source_summary), na(result_dir)
		}' "$source"
}

summarize_results() {
for summary_entry in "${SUMMARY_FILES[@]}"; do
	summary="${summary_entry#*:}"
	if [[ ! -f "$summary" ]]; then echo "Missing summary file: $summary" >&2; exit 1; fi
done

{
	printf 'test_type\tsuite\tmodel\tprecision\tbatch\ttaskset_bind\tcpu_pinning\thyper_threading\tsync_start\tinstances\tinstances_completed\tbase_threads_per_instance\tthreads_per_instance\teffective_threads\twarmup\tcooldown\tmeasurement_mode\titerations\tduration_seconds\tperf_count\tstability_rounds\tinput_samples_per_instance\ttotal_timed_inferences\tmax_instance_elapsed_seconds\tmean_ms\tmax_instance_p99_ms\ttotal_throughput_fps\tthroughput_per_instance_fps\tthroughput_basis\taccuracy_passed\tmd5_consistent\tstable_outputs_finite\tstatus\tsource_summary\tresult_dir\n'
	for summary_entry in "${SUMMARY_FILES[@]}"; do
		group="${summary_entry%%:*}"
		summary="${summary_entry#*:}"
		append_unified_summary single "$group" "$summary"
	done
	if [[ -f "$MULTI_SUMMARY" ]]; then append_unified_summary multi "" "$MULTI_SUMMARY"; fi
} > "$SUMMARY_ALL"

log "Completed unified C++ matrix"
for summary_entry in "${SUMMARY_FILES[@]}"; do log "Summary: ${summary_entry#*:}"; done
log "Merged summary: $SUMMARY_ALL"
if [[ -f "$MULTI_SUMMARY" ]]; then log "Multi-instance summary: $MULTI_SUMMARY"; fi

echo "RUN_DIR=$RUN_DIR"
echo "BINARY=$BINARY"
for summary_entry in "${SUMMARY_FILES[@]}"; do
	group="${summary_entry%%:*}"
	printf '%s_RESULTS=%s\n' "${group^^}" "$(dirname "${summary_entry#*:}")"
done
echo "SUMMARY_ALL=$SUMMARY_ALL"
if [[ -f "$MULTI_SUMMARY" ]]; then echo "MULTI_INSTANCE_SUMMARY=$MULTI_SUMMARY"; fi
}

aggregate_instance_results() {
	local case_root="$1" suite="$2" model="$3" precision="$4" instance_count="$5"
	shift 5
	local -a worker_dirs=("$@")
	local index
	local metrics_file="$case_root/instance_metrics.tsv"
	: > "$metrics_file"
	for index in "${!worker_dirs[@]}"; do
		awk -F '\t' -v OFS='\t' -v configured_batch="$BATCH" '
			function field(name) { return name in columns ? $(columns[name]) : "" }
			NR == 1 { for (i = 1; i <= NF; i++) columns[$i] = i; next }
			NF {
				batch = field("batch")
				if (batch == "") batch = configured_batch
				print batch, field("throughput_fps"), field("timed_inferences"), field("mean_ms"),
					field("p99_ms"), field("status"), field("elapsed_seconds"), field("images")
			}' \
			"${worker_dirs[$index]}/results/summary.tsv" >> "$metrics_file"
	done
	awk -F '\t' -v OFS='\t' -v suite="$suite" -v model="$model" -v precision="$precision" \
		-v instances="$instance_count" -v threads="$WAVE_THREADS" \
		-v warmup="$WARMUP" -v iterations="$ITERATIONS" -v binding="$TASKSET_BIND" \
		-v ht="$WAVE_HYPER_THREADING" -v base_threads="$INSTANCE_THREADS" -v duration="$DURATION" \
		-v sync_start="$SYNC_START" -v result_dir="$case_root" '
		{
			batch = $1
			if (!(batch in seen)) { seen[batch] = 1; order[++variant_count] = batch }
			count[batch]++
			calls = $3 + 0
			total_fps[batch] += $2 + 0
			total_calls[batch] += calls
			weighted_latency[batch] += ($4 + 0) * calls
			if (($5 + 0) > max_p99[batch]) max_p99[batch] = $5 + 0
			if (($7 + 0) > max_elapsed[batch]) max_elapsed[batch] = $7 + 0
			if ($6 != "PASS") failed[batch]++
			if (count[batch] == 1) inputs[batch] = $8
			else if (inputs[batch] != $8) inputs[batch] = "MIXED"
		}
		END {
			if (variant_count == 0) exit 2
			# The inactive measurement budget (iterations vs duration) and any cell
			# that could not be computed are reported as N/A, never left blank.
			iterations_str = iterations == "" ? "N/A" : iterations
			duration_str = duration == "" ? "N/A" : duration
			for (i = 1; i <= variant_count; i++) {
				batch = order[i]
				if (count[batch] != instances) exit 2
				status = failed[batch] ? "FAIL" : "PASS"
				mean = total_calls[batch] > 0 ?
					sprintf("%.9f", weighted_latency[batch] / total_calls[batch]) : "N/A"
				printf "%s\t%s\t%s\t%s\t%d\t%d\t%d\t%s\t%d\t%.9f\t%.9f\t%.9f\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
					suite, model, precision, batch, instances, threads, warmup, iterations_str,
					count[batch], total_fps[batch], total_fps[batch] / instances, max_p99[batch],
					status, binding, ht, base_threads, duration_str, total_calls[batch], sync_start,
					mean, max_elapsed[batch], inputs[batch], result_dir
			}
		}' "$metrics_file" > "$case_root/aggregate_row.tsv"
	cat "$case_root/aggregate_row.tsv" >> "$MULTI_SUMMARY"
}
