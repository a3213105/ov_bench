#!/usr/bin/env bash

build_cpp() {
if [[ -n "$OPENVINO_SO_PATH" ]]; then
  if [[ ! -f "$OPENVINO_SO_PATH" ]]; then
    echo "OpenVINO so path not found: $OPENVINO_SO_PATH" >&2
    exit 1
  fi
  OPENVINO_LIB_DIR="$(cd "$(dirname "$OPENVINO_SO_PATH")" && pwd)"
fi

if [[ -n "$OPENVINO_LIB_DIR" && ! -d "$OPENVINO_LIB_DIR" ]]; then
  echo "OpenVINO lib dir not found: $OPENVINO_LIB_DIR" >&2
  exit 1
fi

# Derive OpenVINO_DIR from lib dir if only so/lib path is provided.
if [[ -z "$OPENVINO_DIR" && -n "$OPENVINO_LIB_DIR" ]]; then
  for candidate in \
    "$OPENVINO_LIB_DIR/cmake/openvino"* \
    "$OPENVINO_LIB_DIR/cmake" \
    "$OPENVINO_LIB_DIR/../lib/cmake/openvino"* \
    "$OPENVINO_LIB_DIR/../cmake" \
    "$OPENVINO_LIB_DIR/../../cmake" \
    "$OPENVINO_LIB_DIR/../runtime/cmake" \
    "$OPENVINO_LIB_DIR/../../runtime/cmake"; do
    if [[ -f "$candidate/OpenVINOConfig.cmake" ]]; then
      OPENVINO_DIR="$(cd "$candidate" && pwd)"
      break
    fi
  done
  if [[ -z "$OPENVINO_DIR" ]]; then
    found_config="$(find "$OPENVINO_LIB_DIR" "$(dirname "$OPENVINO_LIB_DIR")" -maxdepth 5 -name OpenVINOConfig.cmake 2>/dev/null | head -n1 || true)"
    if [[ -n "$found_config" ]]; then
      OPENVINO_DIR="$(cd "$(dirname "$found_config")" && pwd)"
    fi
  fi
fi

if [[ -z "$OPENVINO_DIR" ]]; then
  echo "--openvino-dir is required (or provide --openvino-libs with a discoverable cmake package)" >&2
  exit 2
fi
if [[ ! -f "$OPENVINO_DIR/OpenVINOConfig.cmake" ]]; then
  echo "OpenVINOConfig.cmake not found in: $OPENVINO_DIR" >&2
  exit 1
fi
if [[ ! -x "$TEST_SCRIPT" ]]; then
  echo "Test script is not executable: $TEST_SCRIPT" >&2
  exit 1
fi

if (( DRY_RUN == 1 )); then
  echo "[DRY-RUN] BUILD_DIR=$BUILD_DIR"
  echo "[DRY-RUN] BINARY=$BINARY"
  echo "[DRY-RUN] OPENVINO_DIR=$OPENVINO_DIR"
  echo "[DRY-RUN] OPENVINO_LIB_DIR=${OPENVINO_LIB_DIR:-<unchanged>}"
  echo "[DRY-RUN] TEST_SCRIPT=$TEST_SCRIPT"
  printf '[DRY-RUN] FORWARDED_ARGS='; printf '%q ' "${FORWARD_ARGS[@]}"; echo
  return 0
fi

mkdir -p "$BUILD_DIR"
cmake -S "$CPP_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR="$OPENVINO_DIR"
cmake --build "$BUILD_DIR" --target ov_bench -j "${BUILD_JOBS:-8}"

if [[ ! -x "$BINARY" ]]; then
  echo "Built binary not executable: $BINARY" >&2
  exit 1
fi

if [[ -n "$OPENVINO_LIB_DIR" ]]; then
  runtime_library_path="$OPENVINO_LIB_DIR"
  tbb_library_dir="$OPENVINO_LIB_DIR/../../3rdparty/tbb/lib"
  if [[ -f "$tbb_library_dir/libtbb.so.12" ]]; then
    runtime_library_path+=":$tbb_library_dir"
  fi
  export LD_LIBRARY_PATH="$runtime_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

}
