#!/usr/bin/env bash
expand_cpu_list() {
	local spec="$1" output_name="$2" part first last cpu
	local -n output="$output_name"
	IFS=',' read -r -a parts <<< "$spec"
	for part in "${parts[@]}"; do
		if [[ "$part" == *-* ]]; then
			first="${part%-*}"
			last="${part#*-}"
			for ((cpu = first; cpu <= last; cpu++)); do output+=("$cpu"); done
		else
			output+=("$part")
		fi
	done
}

prepare_single_numa_binding() {
	local cpu_spec cpu node_cpus_spec cpu_list node_dir node selected_node=-1 available
	local requested_node="${SINGLE_NUMA_NODE_OPTION:-last}"
	local -a node_cpus=() allowed_cpus=()
	local -A allowed_cpu_set=()
	command -v taskset >/dev/null || { echo "taskset is required for NUMA binding" >&2; return 1; }
	cpu_spec="$(taskset -pc "$$" | awk -F': ' '{print $NF}')"
	expand_cpu_list "$cpu_spec" allowed_cpus
	for cpu in "${allowed_cpus[@]}"; do allowed_cpu_set["$cpu"]=1; done
	for node_dir in /sys/devices/system/node/node[0-9]*; do
		[[ -r "$node_dir/cpulist" ]] || continue
		node="${node_dir##*node}"
		[[ "$requested_node" == last || "$node" == "$requested_node" ]] || continue
		node_cpus=()
		expand_cpu_list "$(<"$node_dir/cpulist")" node_cpus
		available=0
		for cpu in "${node_cpus[@]}"; do
			if [[ -n "${allowed_cpu_set[$cpu]+x}" ]]; then available=1; break; fi
		done
		if (( available == 1 && node > selected_node )); then selected_node="$node"; fi
	done
	if (( selected_node < 0 )); then
		echo "Requested NUMA node '$requested_node' has no CPUs available to this process" >&2
		return 2
	fi
	SINGLE_NUMA_NODE="$selected_node"
	node_cpus_spec="$(<"/sys/devices/system/node/node${SINGLE_NUMA_NODE}/cpulist")"
	node_cpus=()
	expand_cpu_list "$node_cpus_spec" node_cpus
	SINGLE_NUMA_CPU_LIST=""
	for cpu in "${node_cpus[@]}"; do
		[[ -n "${allowed_cpu_set[$cpu]+x}" ]] || continue
		[[ -z "$SINGLE_NUMA_CPU_LIST" ]] || SINGLE_NUMA_CPU_LIST+=","
		SINGLE_NUMA_CPU_LIST+="$cpu"
	done
	[[ -n "$SINGLE_NUMA_CPU_LIST" ]] || {
		echo "No CPUs from NUMA node $SINGLE_NUMA_NODE are available to this process" >&2
		return 2
	}
	log "Single-process NUMA binding enabled; node=$SINGLE_NUMA_NODE cpus=$SINGLE_NUMA_CPU_LIST"
}

prepare_core_scaling_cpu_lists() {
	local cpu_spec node_dir node cpu sibling group complete count index cpu_list max_count target_node="${1:-}"
	local -a node_cpus=() siblings=() core_groups=() reversed_core_groups=()
	local -A allowed_cpu_set=() node_cpu_set=() seen_core_groups=()
	declare -gA CORE_SCALING_CPU_LISTS=()
	AVAILABLE_CPUS=()
	max_count=0
	for count in "${CORE_SCALING_LIST[@]}"; do
		(( count <= max_count )) || max_count="$count"
	done
	command -v taskset >/dev/null || { echo "taskset is required for core scaling" >&2; return 1; }
	cpu_spec="$(taskset -pc "$$" | awk -F': ' '{print $NF}')"
	expand_cpu_list "$cpu_spec" AVAILABLE_CPUS
	for cpu in "${AVAILABLE_CPUS[@]}"; do allowed_cpu_set["$cpu"]=1; done
	for node_dir in /sys/devices/system/node/node[0-9]*; do
		[[ -d "$node_dir" ]] || continue
		node="${node_dir##*node}"
		[[ -z "$target_node" || "$node" == "$target_node" ]] || continue
		node_cpus=()
		expand_cpu_list "$(<"$node_dir/cpulist")" node_cpus
		node_cpu_set=()
		for cpu in "${node_cpus[@]}"; do
			[[ -n "${allowed_cpu_set[$cpu]+x}" ]] && node_cpu_set["$cpu"]=1
		done
		core_groups=()
		for cpu in "${node_cpus[@]}"; do
			[[ -n "${node_cpu_set[$cpu]+x}" ]] || continue
			group="$(<"/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list")"
			[[ -z "${seen_core_groups[$group]+x}" ]] || continue
			siblings=()
			expand_cpu_list "$group" siblings
			complete=1
			for sibling in "${siblings[@]}"; do
				if [[ -z "${node_cpu_set[$sibling]+x}" ]]; then complete=0; break; fi
			done
			(( complete == 1 )) || continue
			seen_core_groups["$group"]=1
			core_groups+=("$(IFS=,; echo "${siblings[*]}")")
		done
		reversed_core_groups=()
		for ((index = ${#core_groups[@]} - 1; index >= 0; index--)); do
			reversed_core_groups+=("${core_groups[$index]}")
		done
		core_groups=("${reversed_core_groups[@]}")
		(( max_count <= ${#core_groups[@]} )) || continue
		for count in "${CORE_SCALING_LIST[@]}"; do
			cpu_list=""
			for ((index = 0; index < count; index++)); do
				[[ -z "$cpu_list" ]] || cpu_list+=","
				cpu_list+="${core_groups[$index]}"
			done
			CORE_SCALING_CPU_LISTS["$count"]="$cpu_list"
		done
		log "CORE SCALING NUMA node=$node physical_cores=${#core_groups[@]}"
		break
	done
	for count in "${CORE_SCALING_LIST[@]}"; do
		if [[ -z "${CORE_SCALING_CPU_LISTS[$count]+x}" ]]; then
			echo "No NUMA node has $count complete available physical cores for core scaling" >&2
			return 2
		fi
		log "CORE SCALING cores=$count cpus=${CORE_SCALING_CPU_LISTS[$count]}"
	done
}

prepare_worker_cpu_assignments() {
	local cpu_spec node_dir node cpulist cpu cpu_list offset sibling group complete
	local -a node_cpus=() siblings=() core_groups=()
	local -A node_cpu_set=() seen_core_groups=()
	AVAILABLE_CPUS=()
	WORKER_CPU_LISTS=()
	WORKER_NUMA_NODES=()
	NODE_NUMBERS=()
	NODE_CPU_LISTS=()
	AVAILABLE_CPU_SET=()
	if (( TASKSET_BIND == 0 )); then
		log "Multi-instance NUMA binding disabled; OpenVINO CPU pinning=NO"
		return 0
	fi
	command -v taskset >/dev/null || { echo "taskset is required for CPU binding" >&2; return 1; }
	cpu_spec="$(taskset -pc "$$" | awk -F': ' '{print $NF}')"
	expand_cpu_list "$cpu_spec" AVAILABLE_CPUS
	for cpu in "${AVAILABLE_CPUS[@]}"; do AVAILABLE_CPU_SET["$cpu"]=1; done
	for node_dir in /sys/devices/system/node/node[0-9]*; do
		[[ -d "$node_dir" ]] || continue
		node="${node_dir##*node}"
		cpulist="$(<"$node_dir/cpulist")"
		node_cpus=()
		expand_cpu_list "$cpulist" node_cpus
		cpu_list=""
		for cpu in "${node_cpus[@]}"; do
			[[ -n "${AVAILABLE_CPU_SET[$cpu]+x}" ]] || continue
			if [[ -n "$cpu_list" ]]; then cpu_list+=","; fi
			cpu_list+="$cpu"
		done
		if [[ -n "$cpu_list" ]]; then
			NODE_CPU_LISTS["$node"]="$cpu_list"
			NODE_NUMBERS+=("$node")
		fi
	done
	if (( ${#NODE_NUMBERS[@]} == 0 )); then
		echo "No allowed CPUs found in NUMA node topology" >&2
		return 1
	fi
	mapfile -t NODE_NUMBERS < <(printf '%s\n' "${NODE_NUMBERS[@]}" | sort -n -u)
	for node in "${NODE_NUMBERS[@]}"; do
		node_cpus=()
		expand_cpu_list "${NODE_CPU_LISTS[$node]}" node_cpus
		node_cpu_set=()
		core_groups=()
		for cpu in "${node_cpus[@]}"; do node_cpu_set["$cpu"]=1; done
		for cpu in "${node_cpus[@]}"; do
			group="$(<"/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list")"
			[[ -z "${seen_core_groups[$group]+x}" ]] || continue
			siblings=()
			expand_cpu_list "$group" siblings
			complete=1
			for sibling in "${siblings[@]}"; do
				if [[ -z "${node_cpu_set[$sibling]+x}" ]]; then complete=0; break; fi
			done
			(( complete == 1 )) || continue
			seen_core_groups["$group"]=1
			core_groups+=("$(IFS=,; echo "${siblings[*]}")")
		done
		for ((offset = 0; offset + INSTANCE_THREADS <= ${#core_groups[@]}; offset += INSTANCE_THREADS)); do
			cpu_list=""
			for ((cpu = 0; cpu < INSTANCE_THREADS; cpu++)); do
				if [[ -n "$cpu_list" ]]; then cpu_list+=","; fi
				cpu_list+="${core_groups[$((offset + cpu))]}"
			done
			WORKER_CPU_LISTS+=("$cpu_list")
			WORKER_NUMA_NODES+=("$node")
		done
		log "NUMA node=$node complete_physical_cores=${#core_groups[@]} idle_physical_cores=$((${#core_groups[@]} % INSTANCE_THREADS))"
	done
	if (( ${#WORKER_CPU_LISTS[@]} == 0 )); then
		echo "No NUMA node has $INSTANCE_THREADS complete available physical cores" >&2
		return 2
	fi
	log "Multi-instance NUMA binding enabled; OpenVINO CPU pinning=YES; physical_cores_per_instance=$INSTANCE_THREADS; auto_instances=${#WORKER_CPU_LISTS[@]}"
}

