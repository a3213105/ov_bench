#!/usr/bin/env bash
# ov_bench ALL-ITEMS entrypoint: the general driver that runs every test item for a
# given model selection (registered families and/or arbitrary --model-file targets)
# by invoking each item entrypoint with a shared report directory:
#
#   single-process  (ov_bench_single.sh)      -- when --mode includes the single stage
#   core-scaling    (ov_bench_corescaling.sh) -- in single/all modes, when the
#                                                 core list is non-empty (default
#                                                 1,2,4,8; --no-core-scaling to
#                                                 disable); runs AFTER the
#                                                 single-process item
#   multi-instance  (ov_bench_multi.sh)       -- when --mode includes the multi stage
#   summary         (ov_bench_summary.sh)     -- runs when any of the items above produced mergeable output
#
# Registered-model runs use --suites dinov3,yolo-world,pred,unet (or a subset);
# arbitrary models use --model-file. Build the native binary first with
# scripts/build_cpp.sh and pass it via --binary.
set -euo pipefail
OVB_STAGE_ACTIVE="all"
# Capture the raw arguments BEFORE the library parses them (parameters.sh
# rewrites the positional parameters), so they can be forwarded verbatim.
ALL_ARGS=("$@")
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts"
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIB_DIR/ovbench_lib.sh"

ovb_init || exit 1
if (( DRY_RUN == 1 )); then
	echo "[DRY-RUN] Dispatching item entrypoints under RUN_DIR=$RUN_DIR"
	echo "[DRY-RUN] Forwarded args: ${ALL_ARGS[*]:-} (plus --run-dir $RUN_DIR)"
	exit 0
fi

# Re-emit the captured args to each item entrypoint, replacing any --run-dir with
# one canonical value so every stage shares the same report tree.
forward_args=()
_skip=0
for _a in "${ALL_ARGS[@]}"; do
	if (( _skip )); then _skip=0; continue; fi
	case "$_a" in
		--run-dir) _skip=1; continue ;;
		--run-dir=*) continue ;;
	esac
	forward_args+=("$_a")
done
forward_args=(--run-dir "$RUN_DIR" "${forward_args[@]}")

run_item() {
	local name="$1"
	shift
	log "ITEM BEGIN tools/$name"
	bash "$TOOLS_DIR/$name" "${forward_args[@]}"
	log "ITEM END tools/$name"
}

# The single-process and core-scaling items are independent: the regular
# single-process item always runs in single/all modes, and the core-scaling sweep
# (default CSV 1,2,4,8; disable with --no-core-scaling) runs as an additional
# item whenever a core list is present. The summary item only merges
# single-process and multi-instance output, so it is skipped when neither ran.
summary_items_ran=0
if [[ "$MODE" == "single" || "$MODE" == "all" ]]; then
	run_item "ov_bench_single.sh"
	summary_items_ran=1
fi

if [[ "$MODE" == "single" || "$MODE" == "all" ]] && ((${#CORE_SCALING_LIST[@]} > 0)); then
	run_item "ov_bench_corescaling.sh"
fi

if ovb_multi_active; then
	run_item "ov_bench_multi.sh"
	summary_items_ran=1
fi

if (( summary_items_ran )); then
	run_item "ov_bench_summary.sh"
else
	log "No single-process or multi-instance items ran; skipping summary stage (core scaling reported in core_scaling_summary.tsv)"
fi

log "All ov_bench test items complete under $RUN_DIR"
exit 0
