#!/usr/bin/env bash
# ov_bench SUMMARY test item: merges the persisted single-process entries and the
# multi-instance aggregate into the unified summary_all.tsv. Reads the run config
# snapshot written by the other items so column values stay consistent.
set -euo pipefail
OVB_STAGE_ACTIVE="summary"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/ovbench_lib.sh"

ovb_init || exit 1
(( DRY_RUN == 1 )) && exit 0

# A merge is only meaningful once at least one run stage has produced output.
if [[ -z "$SINGLE_ENTRIES_FILE" || ( ! -f "$SINGLE_ENTRIES_FILE" && ! -f "$MULTI_SUMMARY" ) ]]; then
	echo "No run output found in $RUN_DIR; run a test item first" >&2
	exit 2
fi
summarize_results
exit 0
