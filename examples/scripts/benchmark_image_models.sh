#!/usr/bin/env bash
# Example: full DINOv3 / YOLO-World / UNet test pipeline.
#
# Selects the three documented image-model suites and runs every test item
# through the general all-items orchestrator. The native binary must be
# built first:
#
#   bash scripts/build_cpp.sh --openvino-dir /path/to/openvino/cmake
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN="${OVBENCH_CPP_BIN:-$REPO_ROOT/build/ov_bench}"

usage() {
	cat <<EOF
Usage: bash examples/scripts/benchmark_image_models.sh [pipeline options]

Run the complete DINOv3, YOLO-World and UNet test pipeline, i.e.
tools/ov_bench_all.sh with --suites dinov3,yolo-world,unet --mode all
(single-process performance/accuracy/stability, the core-scaling sweep,
multi-instance performance and the combined summary).

1) Build the native binary (once):
     bash $REPO_ROOT/scripts/build_cpp.sh --openvino-dir /path/to/openvino/cmake

2) Run this example:
     bash examples/scripts/benchmark_image_models.sh --threads 12 --precisions FP32,BF16,FP16 ...

The binary is taken from \$OVBENCH_CPP_BIN, otherwise $REPO_ROOT/build/ov_bench.
All other options are forwarded to tools/ov_bench_all.sh (see its --help).
EOF
}

for arg in "$@"; do
	case "$arg" in
		-h|--help) usage; exit 0 ;;
	esac
done

if [[ ! -x "$BIN" ]]; then
	echo "Native ov_bench binary not found at: $BIN" >&2
	echo "Build it first: bash $REPO_ROOT/scripts/build_cpp.sh --openvino-dir /path/to/openvino/cmake" >&2
	exit 2
fi

exec bash "$REPO_ROOT/tools/ov_bench_all.sh" \
	--binary "$BIN" \
	--suites dinov3,yolo-world,unet \
	--mode all \
	"$@"
