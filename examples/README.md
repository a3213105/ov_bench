# examples/ — reference implementations for specific models

This directory contains test scripts and tools tied to
the concrete models originally benchmarked in this repository
(DinoV3, YOLO-World and UNet). They are
**reference implementations**: they show how to drive the generic
`ov_bench` pipeline against a particular model set, but they are not part of
the generic entry points under `tools/` and `scripts/` in the project root.

To benchmark **your own model**, do not start from this directory. Use the
generic, model-agnostic entry points instead; see the root
[docs/README.md](../docs/README.md) for the full option reference:

```bash
# single registered model file (IR/ONNX), any of the test items
tools/ov_bench_all.sh   --binary "$BIN" --model dinov3-vits16-pretrain-lvd1689m --mode all ...
tools/ov_bench_all.sh   --binary "$BIN" --model-file /path/to/my_model.xml \
                        [--input-shape NAME=DIMS ...] --mode all ...

# or run individual items
tools/ov_bench_single.sh      --binary "$BIN" --model-file MY_MODEL.xml ...
tools/ov_bench_corescaling.sh --binary "$BIN" --model-file MY_MODEL.xml ...
tools/ov_bench_multi.sh       --binary "$BIN" --model-file MY_MODEL.xml ...
tools/ov_bench_summary.sh     --run-dir "$RUN_DIR" ...
```

## Layout

```
examples/
├── scripts/     # model-specific pipeline scripts and reference utilities
├── tools/       # model-specific Python tools (shared helpers + memory diagnostics)
└── tests/       # unit/integration tests for the tools above
```

## Contents

| File | Reference implementation for |
|---|---|
| `scripts/benchmark_image_models.sh` | Full native test pipeline for DINOv3, YOLO-World and UNet, delegated to the repository build/all-items entrypoint |
| `tools/benchmark_common.py` | Shared model-name/manifest helpers used only by the other files here |
| `tools/check_precision_memory.py` | Python-IR RSS/PSS memory diagnostics for the image models (DinoV3/YOLO) |
| `tools/python_ir_benchmark.py` | Python-IR performance worker behind the memory/comparison checks |
| `tests/test_precision_memory.py` | Tests for `check_precision_memory.py` |
| `tests/validate_models.py` | Historical preprocessing validation against `dinov3/` and `yolo-world/` Python scripts |

### Full Image-Model Run

The example selects the three documented image-model suites and delegates test-item
dispatch to the `tools/ov_bench_all.sh` orchestrator. `--mode all` runs
single-process performance, accuracy and stability, the default core-scaling
sweep, multi-instance performance, and the combined summary. Build the native
binary first (`scripts/build_cpp.sh`), then:

```bash
bash scripts/build_cpp.sh --openvino-dir /path/to/openvino/cmake
bash examples/scripts/benchmark_image_models.sh \
    --ir-root /path/to/models_ov \
    --images-dir /path/to/test_images \
    --run-dir /path/to/reports/image_models
```

The wrapper forwards additional test options (e.g. `--threads`, `--precisions`,
`--taskset-bind`, `--core-scaling`, `--duration`) to `tools/ov_bench_all.sh`.
It does not duplicate the implementation in `scripts/` or `tools/`.

Scripts here that need the project use it relative to their own location; some
older one-off scripts still carry frozen absolute paths and are kept as-is for
reproducibility only.
