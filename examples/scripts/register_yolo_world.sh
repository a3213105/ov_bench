#!/usr/bin/env bash
# Register (and optionally run) the YOLO-World image-model suite: the four
# v2 / v2.1 letterboxed IRs. Overrides and pipeline options are forwarded to
# tools/register_suite.sh / tools/ov_bench_all.sh.
set -euo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/register_suite.sh"
exec bash "$TOOLS" --suite yolo-world "$@"
