#!/usr/bin/env bash
# Main entry: build the native ov_benchmark, then run the registered-model
# specialization of the general ov_bench all-items entrypoint.
#
# Registered (specialized) test set: the built-in dinov3 / yolo-world / pred /
# unet suites. To target arbitrary models through the same four test items, use
# the general entrypoint directly, which also accepts --model-file:
#
#     bash /home/sgui/2026/ov_bench/tools/ov_bench_all.sh \
#         --binary /path/to/ov_benchmark --model-file /path/to/model.onnx ...
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$ROOT_DIR"
PROJECT_ROOT="$(dirname "$ROOT_DIR")"
LIB_DIR="$ROOT_DIR/scripts"
TOOLS_DIR="$ROOT_DIR/tools"
TEST_SCRIPT="$TOOLS_DIR/ov_bench_all.sh"
export CPP_DIR PROJECT_ROOT

BUILD_ROOT="${BUILD_ROOT:-/tmp/ovbench-$(id -u)}"
BUILD_DIR="${BUILD_DIR:-$BUILD_ROOT/all-models-cpp-build}"

OPENVINO_DIR="${OPENVINO_DIR:-}"
OPENVINO_LIB_DIR=""
OPENVINO_SO_PATH=""

DRY_RUN="${DRY_RUN:-0}"
FORWARD_ARGS=()

usage() {
	cat <<'EOF'
Usage: run_all_models_cpp_perf_stability_accuracy.sh [options_for_build] [options_for_test]

Single entry point (registered-model test set):
  1) build C++ ov_benchmark
  2) run the general all-items entrypoint (tools/ov_bench_all.sh) over the
     built-in dinov3 / yolo-world / pred / unet suites

General entrypoint for arbitrary models (--model-file) runs each of the same
four test items; see --help for the full argument list.

Build options:
  --openvino-dir DIR         CMake package dir containing OpenVINOConfig.cmake
  --openvino-libs PATH       Runtime directory or libopenvino.so; bundled TBB path is added when present
  --build-dir DIR            CMake build directory
  --dry-run                  Print resolved build/test command and exit
  -h, --help                 Show help

Examples:
  run_all_models_cpp_perf_stability_accuracy.sh \
    --openvino-dir /path/to/openvino/cmake \
    --openvino-libs /path/to/openvino/libs \
    --run-dir /home/sgui/2026/reports/full_cpp

  run_all_models_cpp_perf_stability_accuracy.sh \
    --openvino-libs /path/to/libopenvino.so \
    --mode multi --threads 12 --taskset-bind 0 \
    --instances 24,32,36,38 --hyper-threading 2 --duration 60 --warmup 3
EOF
}

while (($#)); do
	case "$1" in
		--openvino-dir|--openvino-libs|--build-dir)
			(($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
			[[ "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; } ;;
	esac
	case "$1" in
		--openvino-dir)
			OPENVINO_DIR="$2"; shift 2 ;;
		--openvino-libs)
			if [[ -d "$2" ]]; then
				OPENVINO_LIB_DIR="$2"; OPENVINO_SO_PATH=""
			else
				OPENVINO_SO_PATH="$2"; OPENVINO_LIB_DIR=""
			fi
			shift 2 ;;
		--build-dir)
			BUILD_DIR="$2"; shift 2 ;;
		--binary|--test-script|--openvino-lib-dir|--openvino-so-path)
			echo "Removed option: $1; use --build-dir or --openvino-libs (see --help)" >&2
			exit 2 ;;
		--dry-run)
			DRY_RUN=1; shift ;;
		-h|--help)
			usage
			bash "$TEST_SCRIPT" --help
			exit 0 ;;
		*)
			FORWARD_ARGS+=("$1"); shift ;;
	esac
done

BINARY="$BUILD_DIR/ov_benchmark"

# Validate the forwarded test options before the (expensive) build so bad
# arguments fail fast. Paths that do not exist yet are tolerated here; each item
# entrypoint re-validates against the built binary.
(
	OVB_STAGE_ACTIVE="all"
	set -- --binary "$BINARY" "${FORWARD_ARGS[@]}"
	source "$LIB_DIR/ovbench_lib.sh"
	ovb_validate
)

source "$LIB_DIR/build_cpp.sh"
build_cpp
if (( DRY_RUN == 1 )); then exit 0; fi

bash "$TEST_SCRIPT" --binary "$BINARY" "${FORWARD_ARGS[@]}"
