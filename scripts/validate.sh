#!/usr/bin/env bash
# Shared validation library, sourced by scripts/ovbench_lib.sh. Exposes per-stage
# validation helpers so each ov_bench test entrypoint only validates what it needs:
#   validate_common      - scalar + suite checks shared by every run stage
#   resolve_core_scaling - build CORE_SCALING_LIST (core-scaling stage)
#   resolve_instances    - build MULTI_INSTANCE_LIST (multi stage, stage-aware)
#   validate_paths       - required path checks (incl. arbitrary model files)
#   ovb_validate         - stage-aware dispatcher used by ovb_init

trim_csv() {
	echo "$1" | tr -d ' '
}

contains_suite() {
	local suite="$1" csv
	csv=",$(trim_csv "$SUITES"),"
	[[ "$csv" == *",$suite,"* ]]
}

validate_suites() {
	local csv item
	csv="$(trim_csv "$SUITES")"
	[[ -n "$csv" ]] || { echo "--suites must not be empty" >&2; exit 2; }
	local -A seen=()
	IFS=',' read -r -a suite_list <<< "$csv"
	(( ${#suite_list[@]} > 0 )) || { echo "--suites must not be empty" >&2; exit 2; }
	for item in "${suite_list[@]}"; do
		case "$item" in
			dinov3|yolo-world|pred|unet|generic) ;;
			*) echo "Unsupported suite: $item (choose dinov3,yolo-world,pred,unet,generic)" >&2; exit 2 ;;
		esac
		[[ -z "${seen[$item]+x}" ]] || { echo "Duplicate suite: $item" >&2; exit 2; }
		seen["$item"]=1
	done
}

validate_common() {
	local batch_value
	local -A seen_batch_values=()
	if [[ -z "$BINARY" ]]; then echo "--binary is required" >&2; exit 2; fi
	if [[ ! "$THREADS" =~ ^[1-9][0-9]*$ ]]; then echo "--threads must be positive, got: $THREADS" >&2; exit 2; fi

	if [[ "$BATCH" == -1 ]]; then BATCH=0; fi
	if [[ ! "$BATCH" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
		echo "--batch must be 0 (dynamic) or a CSV of positive integers, got: $BATCH" >&2; exit 2
	fi
	IFS=',' read -r -a BATCH_LIST <<< "$BATCH"
	for batch_value in "${BATCH_LIST[@]}"; do
		[[ -n "${seen_batch_values[$batch_value]+x}" ]] && { echo "Duplicate --batch value: $batch_value" >&2; exit 2; }
		seen_batch_values["$batch_value"]=1
	done

	if [[ ! "$WARMUP" =~ ^[0-9]+$ ]]; then echo "--warmup must be non-negative, got: $WARMUP" >&2; exit 2; fi

	if (( DURATION_EXPLICIT == 1 )); then
		if (( ITERATIONS_EXPLICIT == 1 )); then echo "--iterations and --duration are mutually exclusive" >&2; exit 2; fi
		if [[ ! "$DURATION" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
			! LC_ALL=C awk -v value="$DURATION" 'BEGIN { number=value+0; text=tolower(sprintf("%.17g",number)); exit !(number>0 && text !~ /inf|nan/) }'; then
			echo "--duration must be finite and positive seconds, got: $DURATION" >&2; exit 2
		fi
		ITERATIONS=""
		PERFORMANCE_ARGS=(--duration "$DURATION")
	else
		if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then echo "--iterations must be positive, got: $ITERATIONS" >&2; exit 2; fi
		PERFORMANCE_ARGS=(--iterations "$ITERATIONS")
	fi

	if [[ ! "$ROUNDS" =~ ^([2-9]|[1-9][0-9]+)$ ]]; then echo "--stability-rounds must be >=2, got: $ROUNDS" >&2; exit 2; fi
	if [[ "$CPU_PINNING" != "YES" && "$CPU_PINNING" != "NO" ]]; then echo "--cpu-pinning must be YES or NO" >&2; exit 2; fi
	if [[ "$PERF_COUNT" != "YES" && "$PERF_COUNT" != "NO" ]]; then echo "--perf-count must be YES or NO" >&2; exit 2; fi
	if [[ ! "$MULTI_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then echo "--start-timeout must be positive, got: $MULTI_START_TIMEOUT" >&2; exit 2; fi
	if [[ ! "$SYNC_START" =~ ^[012]$ ]]; then echo "--sync-start must be 0, 1 or 2, got: $SYNC_START" >&2; exit 2; fi
	if [[ "$MODE" != single && "$MODE" != multi && "$MODE" != all ]]; then echo "--mode must be single, multi or all" >&2; exit 2; fi
	if [[ ! "$TASKSET_BIND" =~ ^[012]$ ]]; then echo "--taskset-bind must be 0, 1 or 2, got: $TASKSET_BIND" >&2; exit 2; fi
	if [[ "$SINGLE_NUMA_NODE_OPTION" != last && ! "$SINGLE_NUMA_NODE_OPTION" =~ ^[0-9]+$ ]]; then
		echo "--single-numa-node must be a non-negative NUMA node ID or last, got: $SINGLE_NUMA_NODE_OPTION" >&2; exit 2
	fi
	if [[ ! "$UNBOUND_HT" =~ ^[012]$ ]]; then echo "--hyper-threading must be 0, 1 or 2, got: $UNBOUND_HT" >&2; exit 2; fi

	validate_suites
	if contains_suite generic; then
		(( ${#MODEL_FILES[@]} > 0 )) || { echo "--model-file is required for the generic suite" >&2; exit 2; }
	fi
	if [[ -n "$INPUT_COUNT" ]]; then
		if [[ ! "$INPUT_COUNT" =~ ^[1-9][0-9]*$ ]]; then echo "--input-count must be a positive integer, got: $INPUT_COUNT" >&2; exit 2; fi
		contains_suite pred || { echo "--input-count requires the pred suite" >&2; exit 2; }
		[[ -z "$PRED_MANIFEST" ]] || { echo "--input-count cannot be combined with --pred-manifest" >&2; exit 2; }
	fi
	return 0
}

resolve_core_scaling() {
	CORE_SCALING_LIST=()
	if [[ -z "$CORE_SCALING_COUNTS" ]]; then return 0; fi
	if [[ "$MODE" == multi ]]; then
		# Core scaling is a single-process item; in a multi-only run it is
		# ignored rather than fatal (it is also enabled by default, which must
		# not break plain --mode multi invocations).
		echo "NOTE: --core-scaling ignored with --mode multi (single-process item)" >&2
		return 0
	fi
	IFS=',' read -r -a CORE_SCALING_LIST <<< "$(trim_csv "$CORE_SCALING_COUNTS")"
	local -A seen=()
	local count
	for count in "${CORE_SCALING_LIST[@]}"; do
		if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid core-scaling count: $count" >&2; exit 2; fi
		if [[ -n "${seen[$count]+x}" ]]; then echo "Duplicate core-scaling count: $count" >&2; exit 2; fi
		seen["$count"]=1
	done
	return 0
}

resolve_instances() {
	MULTI_INSTANCE_LIST=()
	if ! ovb_multi_active; then
		[[ -z "$MULTI_INSTANCE_COUNTS" ]] || { echo "--instances is only used by the multi stage" >&2; exit 2; }
		return 0
	fi
	if (( TASKSET_BIND == 1 )); then
		if [[ -n "$MULTI_INSTANCE_COUNTS" ]]; then
			echo "[INFO] --instances is ignored with --taskset-bind 1; instance count comes from complete physical-core groups"
		fi
		return 0
	fi
	if [[ -z "$MULTI_INSTANCE_COUNTS" ]]; then echo "Unbound multi stage requires --instances" >&2; exit 2; fi
	IFS=',' read -r -a MULTI_INSTANCE_LIST <<< "$(trim_csv "$MULTI_INSTANCE_COUNTS")"
	(( ${#MULTI_INSTANCE_LIST[@]} > 0 )) || { echo "--instances must not be empty" >&2; exit 2; }
	local -A seen=()
	local count
	for count in "${MULTI_INSTANCE_LIST[@]}"; do
		if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid instance count: $count" >&2; exit 2; fi
		if [[ -n "${seen[$count]+x}" ]]; then echo "Duplicate instance count: $count" >&2; exit 2; fi
		seen["$count"]=1
	done
	return 0
}

validate_paths() {
	local path
	for path in "$BINARY"; do
		if [[ ! -e "$path" ]]; then echo "Required path not found: $path" >&2; exit 1; fi
	done
	if contains_suite generic; then
		for path in "${MODEL_FILES[@]}"; do
			if [[ ! -f "$path" ]]; then echo "Model file not found: $path" >&2; exit 1; fi
		done
	fi
	if contains_suite dinov3 || contains_suite yolo-world; then
		for path in "$MODEL_ROOT_IMAGES" "$IMAGES_DIR"; do
			if [[ ! -e "$path" ]]; then echo "Required path not found: $path" >&2; exit 1; fi
		done
		if [[ -n "$IMAGE_MANIFEST" && ! -f "$IMAGE_MANIFEST" ]]; then echo "Image manifest not found: $IMAGE_MANIFEST" >&2; exit 1; fi
	fi
	if contains_suite pred; then
		for path in "$MODEL_ROOT_PRED"; do
			if [[ ! -e "$path" ]]; then echo "Required path not found: $path" >&2; exit 1; fi
		done
		if [[ -n "$PRED_MANIFEST" && ! -f "$PRED_MANIFEST" ]]; then echo "Prediction manifest not found: $PRED_MANIFEST" >&2; exit 1; fi
	fi
	if contains_suite unet; then
		for path in "$MODEL_ROOT_UNET" "$IMAGES_DIR"; do
			if [[ ! -e "$path" ]]; then echo "Required path not found: $path" >&2; exit 1; fi
		done
	fi
	return 0
}

# Stage-aware dispatcher used by ovb_init. The summary stage needs no binary or
# model paths, so it is validated to the minimum and loads the run configuration.
ovb_validate() {
	if [[ "${OVB_STAGE_ACTIVE:-}" == "summary" ]]; then
		[[ -n "$RUN_DIR" ]] || { echo "--run-dir is required for the summary stage" >&2; exit 2; }
		[[ -d "$RUN_DIR" ]] || { echo "Run directory not found: $RUN_DIR" >&2; exit 2; }
		return 0
	fi
	validate_common
	case "${OVB_STAGE_ACTIVE:-all}" in
		single) : ;;
		corescaling) resolve_core_scaling ;;
		multi) resolve_instances ;;
		summary) : ;;
		* ) resolve_core_scaling; resolve_instances ;;
	esac
	validate_paths
}
