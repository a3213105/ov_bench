# OpenVINO C++ model tests

All standard native inference, accuracy, performance, stability and regression modes
use [src/main.cpp](../src/main.cpp). CMake builds one executable:
`ov_benchmark`. Model-specific schemas, input preparation and graph policies
are in [src/application.cpp](../src/application.cpp).

The only file at the `ov_bench` root is the entry script
`run_all_models_cpp_perf_stability_accuracy.sh` (plus `CMakeLists.txt`).
Layout:

| Directory | Content |
|---|---|
| `src/` | C++ sources (`main.cpp`, `core.cpp`, ...) |
| `include/ovbench/` | C++ headers |
| `tools/` | Python benchmark/diagnostic/reference tools |
| `tests/` | Unit tests (`test_*.py`, `random_accuracy_core.cpp`) |
| `scripts/` | Pipeline and workflow shell scripts |
| `docs/` | This README and [PRED_MODELS.md](./PRED_MODELS.md) |

## Supported models

| Family | Model IDs | Artifacts |
|---|---|---|
| DINOv3 | `dinov3-convnext-tiny-pretrain-lvd1689m`, `dinov3-vits16-pretrain-lvd1689m`, `dinov3-vits16plus-pretrain-lvd1689m` | Existing XML/BIN IR, RGB normalized NCHW 224 |
| YOLO-World | `v2:640`, `v2:1280`, `v2.1:640`, `v2.1:1280` | Existing XML/BIN IR, letterboxed RGB; three raw heads |
| Prediction | `26Q2_round13_run2_e40_no_map_attn`, `ped_26Q1_history_data_gen4_800v2_issue2_e23_refine`, `vehicle_q2_release_v0_w_leading_25` | Original ONNX, named floating features and boolean masks |

These are numerical inference tests, not detection decoding/NMS or business
quality evaluation. See [prediction workflow](./PRED_MODELS.md).

## Build and regression tests

Requires CMake >= 3.16, C++17, OpenVINO Runtime SDK, OpenSSL Crypto,
libjpeg, libpng and nlohmann-json. Local framework reference tests use
conda `didi_t213`.

```bash
CPP=/home/sgui/2026/ov_bench
PYTHON=/home/sgui/miniforge3/envs/didi_t213/bin/python
mkdir -p "/tmp/ovbench-$(id -u)"
RUN="$(mktemp -d "/tmp/ovbench-$(id -u)/validation-XXXXXXXX")"
cmake -S "$CPP" -B "$RUN/build" -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINO_DIR=/home/sgui/miniforge3/envs/didi_t213/lib/cmake/openvino2025.4.1
cmake --build "$RUN/build" --target ov_benchmark -j 4
BIN="$RUN/build/ov_benchmark"
"$BIN" --mode self-test
OVBENCH_BINARY="$BIN" "$PYTHON" -m unittest discover -s "$CPP/tests" -p 'test_*.py'
```

Configure separate builds for each OpenVINO SDK and use matching runtime
libraries. Do not mix headers/runtime versions or switch a compiled binary
between SDKs by changing `LD_LIBRARY_PATH`.

Generated tensors, builds, exports and caches belong in unique temporary
directories under `/tmp/ovbench-<uid>/`. Persist required JSON/TSV reports,
logs and provenance under the project reports directory. Temporary paths are
not archives; pass prepared manifests explicitly.

## Active files

| Purpose | Entry points |
|---|---|
| Native shared functionality | [src/core.cpp](../src/core.cpp), [src/stability.cpp](../src/stability.cpp), [src/image.cpp](../src/image.cpp) |
| Random accuracy and tolerance search | [src/random_accuracy.cpp](../src/random_accuracy.cpp), [tools/random_reference_worker.py](../tools/random_reference_worker.py) |
| Reference: detached accuracy matrices (specific model list) | [examples/scripts/run_random_accuracy_background.sh](../examples/scripts/run_random_accuracy_background.sh), [examples/scripts/run_random_accuracy_versions.sh](../examples/scripts/run_random_accuracy_versions.sh) |
| Accuracy summaries | [tools/summarize_random_accuracy.py](../tools/summarize_random_accuracy.py), [tools/summarize_random_accuracy_versions.py](../tools/summarize_random_accuracy_versions.py) |
| Prepared datasets | [tools/prepare_pred_models.py](../tools/prepare_pred_models.py), [tools/export_reference.py](../tools/export_reference.py), [examples/scripts/benchmark_pred_models.sh](../examples/scripts/benchmark_pred_models.sh) (reference) |
| Reference: bound performance/stability matrix (Python, image models) | [examples/tools/run_multi_instance.py](../examples/tools/run_multi_instance.py) |
| Native C++ pipeline — build + all items (registered models) | [/home/sgui/2026/ov_bench/run_all_models_cpp_perf_stability_accuracy.sh](../run_all_models_cpp_perf_stability_accuracy.sh) |
| Native C++ all-items orchestrator (general, supports `--model-file`) | [tools/ov_bench_all.sh](../tools/ov_bench_all.sh) |
| Native C++ test items (single-process / core-scaling / multi-instance / summary) | [tools/ov_bench_single.sh](../tools/ov_bench_single.sh), [tools/ov_bench_corescaling.sh](../tools/ov_bench_corescaling.sh), [tools/ov_bench_multi.sh](../tools/ov_bench_multi.sh), [tools/ov_bench_summary.sh](../tools/ov_bench_summary.sh) |
| Reference: Python IR benchmark worker (memory diagnostics) | [examples/tools/python_ir_benchmark.py](../examples/tools/python_ir_benchmark.py) |
| Reference: shared dataset/manifest helpers | [examples/tools/benchmark_common.py](../examples/tools/benchmark_common.py) |
| Reference: memory diagnostics | [examples/tools/check_precision_memory.py](../examples/tools/check_precision_memory.py) (Python IR), [examples/tools/run_pred_memory.py](../examples/tools/run_pred_memory.py) / [src/memory.cpp](../src/memory.cpp) (native prediction stages, optional CMake target; see [prediction memory workflow](./PRED_MODELS.md#native-stage-memory-diagnostic)) |
| CPU plugin deployment with guard/rollback | [tools/deploy_cpu_plugin.py](../tools/deploy_cpu_plugin.py) |
| Reference: real-model preprocessing validation | [examples/tests/validate_models.py](../examples/tests/validate_models.py) |

The files under [examples/](../examples/README.md) are reference implementations
for the specific models originally benchmarked here (DinoV3 / YOLO-World /
UNet / pred). To test your own model use the generic entry points above
(`--model-file` / `--input-shape`); they do not depend on anything in
`examples/`.

Historical fixed-source/pip comparison, one-off didi deployment and fixed
38/40-instance supplement scripts are no longer maintained here. Their reports
and deployment snapshots are not modified. The current matrix supports 40
instances directly.

The core-scaling sweep, `--core-scaling N[,N...]`, is **part of the default
single-process test set** (default list `1,2,4,8`): in `--mode single` or
`all` it runs as an additional item **after** the regular single-process
accuracy/performance/stability run, covering every selected suite, model,
precision and physical-core count. Use `--no-core-scaling` to run only the
regular single-process item. Bound points select cores from the last available
NUMA node (or `--single-numa-node N`), starting at the highest-numbered
physical core, and bind all SMT siblings with `taskset`. OpenVINO threads are
set to each requested core count. Results are saved to
`core_scaling_summary.tsv` and `core_scaling/`. With `--mode all`, the
multi-instance stage still runs after core scaling.

`--batch 0|N[,0|N...]` selects batch variants in the given order: `0` keeps
dynamic batch, while a positive integer reshapes to that static batch size.
For example, `--batch 0,1,2` tests dynamic, batch 1 and batch 2 in single,
multi-instance and core-scaling performance runs; summaries include a `batch`
column for every result. In multi-instance mode a comma-separated batch list
runs one independent wave per batch value under every `--sync-start` mode
(0, 1 and 2), each wave into its own `batch_0|N` subtree; `--sync-start 1` synchronizes
workers independently for each batch variant, while `2` runs independent and
synchronized waves. A single `--batch 0` runs only dynamic batch.

Single mode defaults to 12 explicit OpenVINO threads; `--threads N` changes the
count. Core-scaling binds N physical cores and uses N OpenVINO threads.
`throughput_fps`
uses measured wall time; `inference_throughput_fps` is calculated from mean
single-inference latency.

For generated prediction inputs, `--input-count N` controls the random tensor
pool (default 10). A count of 1 binds one input set and reuses it; a larger
count cycles through pre-generated tensors and rebinds each request. In multi
mode, every worker process generates its own in-memory pool and writes to its
own result directory; this works with bound or unbound taskset modes. It cannot
be combined with `--pred-manifest`.

In single mode, `--taskset-bind 1` binds execution to available CPUs on the
selected NUMA node, `0` leaves the process unbound, and `2` runs both variants
in that order. `--single-numa-node N` chooses a node explicitly; `last` uses
the highest-numbered available node by default. Result trees are separated
under `single/bind_1` and `single/bind_0`.
Core-scaling runs follow the same binding choices and record `taskset_bind` in
their summary.

All files ending in `.tsv` use tab characters between columns. Commas within
values, such as a `cpu_list`, remain part of that field.

## Generic Model Files

Test a new ONNX or XML/BIN model directly, without registering a model ID,
preparing images, generating manifests, or invoking Python:

```bash
"$BIN" --model-file /path/to/new_model.onnx \
  --mode performance --precisions FP32,BF16,FP16 \
  --samples 10 --seed 43 --threads 6 --warmup 3 --iterations 200 \
  --cpu-pinning NO --perf-count NO --no-performance-output-files \
  --output /path/to/NEW_REPORT_DIR
```

The model's input names, element types and partial shapes are printed before
input generation, and recorded in `results.json` and
`<model>/input_analysis.json`. Unnamed inputs receive unique `input_N` names.
Static dimensions are read directly. Dynamic leading batch defaults to 1 if
compatible; other dynamic dimensions/ranks must be specified explicitly:

```bash
"$BIN" --model-file /path/to/new_model.xml \
  --input-shape pixel_values=1,3,224,224 \
  --mode all --precisions FP32,BF16,FP16 --samples 10 --seed 43 \
  --threads 6 --warmup 3 --iterations 200 --rounds 10 \
  --cpu-pinning NO --perf-count NO --no-performance-output-files \
  --output /path/to/NEW_REPORT_DIR
```

Repeat `--input-shape NAME=DIM[,DIM...]` for multiple inputs. `NAME=` specifies
a scalar. Shapes must match the model's declared constraints. XML models use
the corresponding sibling BIN file. ONNX external-data files must be available
where referenced by the model.

Floating inputs (`f16`, `bf16`, `f32`, `f64`) use normal(0, 0.1); integer inputs
(`i8/i16/i32/i64/u8/u16/u32/u64`) use uniform values in {0, 1}; booleans use
Bernoulli(0.75). Other dtypes are rejected explicitly. The seed is salted with
the model ID derived from its filename. Generated sample dtype, shape, MD5 and
finite-value metadata are retained for reproducibility. `--samples` controls
the cached random dataset size; `--iterations` remains the exact timed inference
count, cycling through that dataset across warmup, timing and cooldown.

Supported modes are `performance`, `stability`, `accuracy`, `all` and
`preprocess`. Accuracy compares against the same model/runtime's FP32 output,
not an independent framework reference. Integer/boolean outputs require exact
agreement. Stability reuses identical random inputs across rounds and checks
output MD5. Generic loading does not apply family-specific graph rewrites.

Do not combine `--model-file` with `--model-root`, `--model`, images or prepared
manifests. Generic random testing does not infer business input semantics:
index bounds, token ranges, shape-tensor contents and relationships between
inputs may require real prepared inputs or a model-specific adapter. Random
numeric tests are not business-accuracy certification. This direct-file entry
belongs to `ov_benchmark`; the suite pipeline continues to select registered
model families.

## Random multi-precision accuracy

```bash
"$BIN" --mode random-accuracy --model dinov3-convnext-tiny-pretrain-lvd1689m \
  --model-root /home/sgui/2026/models_ov \
  --reference-root /home/sgui/2026/models --reference-python "$PYTHON" \
  --precisions FP32,BF16,FP16 --iterations 100 --seed 43 \
  --threads 32 --cpu-pinning NO --perf-count NO \
  --output "$RUN/random"
```

For each model all selected compiled models and InferRequests coexist.
Each iteration generates fresh shared inputs and runs precisions sequentially.
Floating inputs use `normal(0,0.1)`; masks use Bernoulli(0.75) with first/last
entries valid/invalid when possible. This assumes True means valid.
Model-salted seeds and input hashes allow reproducible comparisons.
`--seed N` is configurable and defaults to 43 in the native runner,
prediction input preparation and both random-accuracy orchestration scripts.
For example, append `--seed 123` after the report path when calling either
orchestration script. Historical reports retain their original recorded seeds.

Only dynamic leading batch resolves automatically to 1. Other dynamic
dimensions require `--random-shapes FILE`, keyed by model ID and input name.
Prediction graphs keep their original dynamic shapes in all precisions.
The removed static BF16 reshape workaround has no fallback.

Reference priority is ONNX Runtime CPU FP32, recognized PyTorch CPU FP32,
then native OpenVINO FP32 when external assets are absent. An available but
broken framework reference is an error, not a reason to fall back.
`--reference-config FILE` explicitly selects reference backends and mappings.
For example:

```json
{
  "vehicle_q2_release_v0_w_leading_25": {"backend": "openvino"}
}
```

Use `"backend": "onnx"` for an explicit ONNX Runtime FP32 reference.
Specify every model to force one baseline for the entire selection.
The worker's [configuration schema](../tools/random_reference_worker.py) also supports
PyTorch/custom adapters and tensor-name mappings.

OpenVINO reference runs use the same runtime's FP32 InferRequest, creating it
even if FP32 is not selected. Its FP32 row is self-comparison, not independent
correctness or repeatability validation. This baseline measures precision
drift, not framework correctness. The reference Python process handles
framework execution only; all tested OpenVINO inference remains in C++.

### Criteria and metrics

Each floating output must have cosine **strictly greater than 0.99** and
every element must satisfy:

```text
abs(actual[i] - reference[i]) <= atol + rtol * abs(reference[i])
```

Integer/boolean outputs require exact equality. Nonfinite values fail.
Zero-norm comparisons require exact equality for the cosine condition.

| Precision | atol | rtol |
|---|---:|---:|
| FP32 | 1e-5 | 1e-4 |
| BF16 | 1e-3 | 1e-2 |
| FP16 | 1e-4 | 1e-3 |

Override with precision-keyed maps, for example
`--atol FP32=1e-5,BF16=1e-3 --rtol FP16=1e-3`.
Omitted precisions retain defaults; duplicate, negative and nonfinite entries
are rejected.

Every input/output record retains name, shape, dtype, cosine, NRMSE,
`mean_abs`, `max_abs`, `abs_error_variance`, exact-equality flags, nonfinite
and out-of-tolerance counts, and input/output hashes.
For `e=abs(actual-reference)`, variance is `sum((e-mean(e))^2)/N`:
population variance of absolute errors, not signed-error variance or MSE.
Single-element variance is zero; nonfinite comparisons report null metrics.

Outputs are `results.json`, `summary.tsv`, per-case JSON, `outputs.tsv`, and
`precision_metrics.tsv`. The latter records one row per iteration and output
with `cosine`, `max_abs`, `normalized_rmse`, and `mismatched_elements`; undefined
floating-point metrics are written as `null`. The JSON retains the same metrics
per output (`cosine_similarity` is the JSON field corresponding to TSV `cosine`).
The TSV pools variance for each named output over all iterations, including
between-iteration mean differences; it is not the average of individual
variances. Unavailable pooled variance is blank. Variance does not affect
pass/fail. Framework diagnostics are in per-model `reference_worker.log`;
temporary raw transport files are removed after use.

### Optional shared-input rejection sampling

Add `--resample-common-cosine-failures` to `--mode random-accuracy`
with both BF16 and FP16 selected (FP32 remains optional). The default is off.
For each shared random candidate, evaluate every selected precision against
the configured reference using `--threshold` (default 0.99). Accept the input
if **all output cosine checks pass in BF16 OR all pass in FP16**. If both
precisions have a failing output, save the rejected input and generate a
fresh candidate. The failing outputs need not have the same name.
Zero-norm and nonfinite cosine checks retain the existing rules.
Element tolerances are not used to select inputs and are not relaxed.

`--iterations N` then means **N accepted inputs per model**, not N attempts.
Retries are unlimited, as requested; a model whose low-precision cosine
checks always fail will not finish without external cancellation. All
precisions share the accepted subset. Accepted records retain the original
0-based candidate `iteration` (which can have gaps), plus a contiguous
0-based `accepted_index`. Metrics and pass/fail summaries include accepted
inputs only; accepting an input does not imply joint accuracy passes.

`results.json` includes a `sampling` policy, selection-bias warning and
per-model attempted/accepted/rejected counters, updated after each candidate.
Matrix JSON summaries retain the sampling policy and TSV summaries include
attempted/rejected counts when this option is enabled.
Rejected candidates are flushed individually to
`MODEL/rejected_inputs.jsonl`, recording seed, candidate iteration,
threshold, reference metadata, input hashes and all selected precisions'
per-output metrics. Exact raw input tensors are retained under
`MODEL/rejected_inputs/random_ITERATION/`; descriptor paths are relative to
the per-model directory. Retaining rejected tensors can require substantial
disk space. No rejected records are mixed into normal accuracy aggregates.

This is a **conditional random sample**, not an unfiltered accuracy result
or evidence that rejected inputs are invalid. Keep an unfiltered run for
comparison. Fixed seed/model reproduces the candidate stream; selected
subsets can differ across reference backends, ISA paths or runtime versions,
so compare retained input hashes before treating results as paired.

The orchestration scripts also accept the flag after the new report path:

```bash
bash "$CPP/examples/scripts/run_random_accuracy_background.sh" "$RUN/filtered" \
  --resample-common-cosine-failures
```

The same flag is forwarded by `run_random_accuracy_versions.sh`. Its
cross-version validator still reports ERROR if accepted input hashes differ,
rather than presenting unpaired subsets as a matched accuracy comparison.

The wrapper's existing per-model two-hour timeout still applies; an interrupted
run is not complete unless the accepted count reaches the requested count.

### Background and dual-version matrices

```bash
bash "$CPP/examples/scripts/run_random_accuracy_background.sh" /path/to/NEW_REPORT_DIR
bash "$CPP/examples/scripts/run_random_accuracy_versions.sh" /path/to/NEW_DUAL_REPORT_DIR
```

Use an external detached process runner when disconnect survival is required.
The scripts do not detach themselves. The local matrix assigns model queues
to NUMA nodes 1/2/3 using each node's physical-core count, with 100 input sets.
Numeric failures do not stop other models. `PASS`/`FAIL` mean completed numeric
evaluation; `ERROR` means execution failure or incomplete coverage.

The version driver snapshots source and verifies the patched CPU plugin hash,
runs 2023.3 then patched 2025.4.1, and verifies cross-version input hashes.
Its SDK paths are local-host settings. Inner runner overrides:
`OVBENCH_SOURCE_DIR`, `OVBENCH_OPENVINO_DIR`, `OVBENCH_REFERENCE_PYTHON`.
Both accept `OVBENCH_REFERENCE_CONFIG`; the version driver snapshots it.
`OVBENCH_REFERENCE_PROJECT_ROOT` locates framework assets from relocated
snapshots. No runtime is installed or replaced by accuracy runners.

Unpatched older runtimes may fail on prediction dynamic BF16. Such failures
are reported, not bypassed by static reshape. Use the fixed plugin for current
prediction validation; see [Gather deployment details](./PRED_MODELS.md).

### FP32 tolerance search

`--mode tolerance-search --precisions FP32` searches a common multiplier
`s>=1` along `atol=s*1e-5, rtol=s*1e-4`. Cosine, nonfinite and integer failures
stop the model without tolerance relaxation. Expansion/bisection use the
maximum per-element required scale; a fresh complete inference pass then
verifies the selected tolerances and input hashes. Relative scale resolution
is `1e-8`; bounds are rounded outward.

`calibration_accuracy` retains default results, `accuracy` retains final
verification, and `tolerance_search` records trials/bounds/status.
These are empirical minima along a fixed ratio, not independent global
minima or business acceptance criteria. The version driver accepts
`OVBENCH_TEST_MODE=tolerance-search`.

## Performance and stability

Native image-model example:

```bash
"$BIN" --mode performance --model dinov3-convnext-tiny-pretrain-lvd1689m \
  --model-root /home/sgui/2026/models_ov --images /home/sgui/2026/test_images \
  --precisions FP32,BF16,FP16 --threads 6 --warmup 1 --iterations 100 \
  --cpu-pinning NO --perf-count NO --output "$RUN/performance"
```

Prepared inputs use `--prepared-inputs --manifest FILE` instead of `--images`.
Performance measures each input batch individually, excludes preprocessing,
binding, compilation, warmup, cooldown and disk output, and records raw latencies.
Warmup and performance iterations count individual inference calls, not dataset
rounds. Each call consumes the next input, wrapping to the first at the end of the
dataset. The input cursor continues across warmup, performance and cooldown.
Cooldown immediately follows measured inference and executes the same number of
calls as warmup. Only `--iterations` calls are timed; warmup, cooldown and later
output checks are excluded. The performance JSON records `iterations`,
`warmup_iterations` and `cooldown_iterations`. `sample_indices.json` records the
0-based dataset indices used for timed inference.
Alternatively use `--duration SECONDS` instead of `--iterations`. Explicitly
specifying both is rejected. Duration uses monotonic elapsed wall time starting
after warmup/the start barrier, completes at least one inference, and stops
starting new calls when the target is reached. The last call may finish after
the target. Cooldown and later output validation remain outside the measurement.
In duration mode, throughput is completed timed calls divided by actual elapsed
seconds; inference-only throughput is additionally recorded as
`inference_throughput_fps`. JSON records the requested duration, elapsed seconds,
and actual completed count; the TSV adds `timed_inferences`/`elapsed_seconds`.
This option applies to native `performance` and `all` modes, and is also accepted
by both pipeline entry points. Precision/stability coverage is unchanged.
ViT-S/16 and ViT-S/16+ FP16 default to exactly 12 BF16 SDPA rewrites.
`--no-sdpa-bf16` is the diagnostic opt-out; the redundant enable alias was
removed. Other models/precisions are not rewritten.

For stability, run the native mode and compare saved hashes across processes:

```bash
for process in a b; do
  "$BIN" --mode stability --model-root /path/to/prepared \
    --manifest /path/to/manifest.json --prepared-inputs --model MODEL_ID \
    --precisions FP32,BF16,FP16 --rounds 10 --threads 6 \
    --cpu-pinning NO --perf-count NO --output "$RUN/stability_$process"
done
"$BIN" --mode compare-stability --stability-dir "$RUN/stability_a" \
  --stability-dir "$RUN/stability_b" \
  --output "$RUN/stability_comparison.json"
```

Cross-process comparison requires at least two distinct process identities.
Repeat `--stability-dir` for multiple runs. Current `stb2` filenames encode model,
input, precision, process, round and output port, preventing collisions.
Comparison verifies hashes, output identities and process/round coverage;
stability is not an accuracy check. Legacy `stb1` filenames are explicitly
rejected rather than silently skipped; historical result files are unchanged.

The topology-specific [bound matrix runner](../examples/tools/run_multi_instance.py) retains
its 15-process, 6-physical-core-plus-HT policy for image-model performance and
cross-process stability. It requires the matching topology, 10 images,
and sufficient disk for stability binaries. The comparator runs through
`ov_benchmark --mode compare-stability`, never a separate executable.

[Memory diagnostics](../examples/tools/check_precision_memory.py) measure the Python native
runtime's RSS/PSS in fresh processes, not the C++ application's memory.

## Prepared references and diagnostic modes

[export_reference.py](../tools/export_reference.py) exports image-model input/reference
tensors; [prepare_pred_models.py](../tools/prepare_pred_models.py) creates named
prediction inputs and ONNX Runtime references. Pass the resulting manifest
explicitly. `--mode accuracy` retains dataset-level aggregate cosine behavior
for existing prepared references; it is not the per-output joint tolerance
criterion of `random-accuracy`.

`--mode all` runs performance, prepared accuracy and stability. `preprocess`
exports native image preprocessing tensors. `single` executes one prepared
sample/precision and logs stages with a configurable timeout, exiting 124 and
writing `timeout.json` on timeout. `gather-guard [--cpu-plugin FILE]` checks
Gather BF16 output and memory bounds. All are dispatched by the same main.

For a complete real-model preprocessing/inference regression:

```bash
"$PYTHON" "$CPP/examples/tests/validate_models.py" --binary "$BIN" \
  --manifest /path/to/reference/manifest.json --output "$RUN/model_validation"
```

This checks all seven image-model specifications and three precisions.
Unit tests also cover prediction preparation, random references, tolerance
search, exact integer handling, variance, stability and default graph policies.
