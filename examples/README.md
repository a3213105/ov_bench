# examples/ — reference implementations for specific models

本目录（This directory）contains the test scripts and tools that are tied to
the concrete models originally benchmarked in this repository
(DinoV3, YOLO-World, UNet and the DiDi prediction / pred models). They are
**reference implementations** (参考实现): they show how to drive the generic
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
├── scripts/     # model-specific pipeline scripts (pred model batch, random-
│                # accuracy reference matrices for DinoV3 + pred models)
├── tools/       # model-specific Python tools (image-model datasets, pred
│                # model MEMORY/throughput tools, bound image-model matrix)
└── tests/       # unit/integration tests for the tools above
```

## Contents

| File | Reference implementation for |
|---|---|
| `scripts/benchmark_pred_models.sh` | The three DiDi prediction models end-to-end (prepare inputs, build, accuracy + stability) |
| `scripts/run_random_accuracy_background.sh` | Random-accuracy matrix for the fixed DinoV3 + pred model list |
| `scripts/run_random_accuracy_check.sh` | One-off Gather/BF16 accuracy rerun on the pred models (frozen paths) |
| `scripts/run_random_accuracy_versions.sh` | A/B driver comparing one historical 2023.3 build against the patched 2025.4.1 plugin on that matrix |
| `tools/benchmark_common.py` | Shared model-name/manifest helpers used only by the other files here |
| `tools/check_precision_memory.py` | Python-IR RSS/PSS memory diagnostics for the image models (DinoV3/YOLO) and pred models |
| `tools/python_ir_benchmark.py` | Python-IR performance worker behind the memory/comparison checks |
| `tools/run_multi_instance.py` | Bound 15-process/6-core image-model performance + stability matrix |
| `tools/run_pred_memory.py` | Native per-stage compile/memory diagnostics for the pred models |
| `tools/summarize_pred_throughput.py` | Throughput summarizer for the pred-model matrix |
| `tests/test_memory_summary.py` | Tests for `summarize_pred_throughput.py` |
| `tests/test_precision_memory.py` | Tests for `check_precision_memory.py` |
| `tests/test_pred_inputs.py` | Tests for input generation in `tools/prepare_pred_models.py` (kept in the root `tools/` because it is model-agnostic) |
| `tests/validate_models.py` | Historical preprocessing validation against `dinov3/` and `yolo-world/` Python scripts |

Scripts here that need the project use it relative to their own location
(e.g. `benchmark_pred_models.sh` resolves the repo root two levels up); some
older one-off scripts still carry frozen absolute paths and are kept as-is for
reproducibility only.
