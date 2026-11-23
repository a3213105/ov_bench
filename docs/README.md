# OpenVINO C++/Python model tests

All standard native inference, accuracy, performance, stability and regression modes
use [src/main.cpp](../src/main.cpp). CMake builds one executable:
`ov_benchmark`. Model-specific schemas, input preparation and graph policies
are in [src/application.cpp](../src/application.cpp).

The same user-visible surface (CLI flags, `results.json` / `summary.tsv` layout,
all modes) is also provided by a Python backend in
[python/ov_bench/](../python/ov_bench/). The root
[`ov_bench`](../ov_bench) entry script runs either engine with the same
command line (see [Backend selection](#backend-selection-c-or-python)):

```bash
./ov_bench --backend python --mode self-test
./ov_bench --backend cpp    --mode self-test
./ov_bench --mode self-test          # prefer built C++ binary, else Python
```

Root files: `ov_bench` (backend-selector entry script),
`run_all_models_cpp_perf_stability_accuracy.sh` (native all-items pipeline) and
`CMakeLists.txt`.
Layout:

| Directory | Content |
|---|---|
| `src/` | C++ sources (`main.cpp`, `core.cpp`, ...) |
| `include/ovbench/` | C++ headers |
| `python/` | Python backend package (`python -m ov_bench`) |
| `tools/` | Python benchmark/diagnostic/reference tools |
| `tests/` | Unit tests (`test_*.py`, `random_accuracy_core.cpp`) |
| `scripts/` | Pipeline and workflow shell scripts (incl. `build_cpp.sh`) |
| `docs/` | This README |

## Supported models

| Family | Model IDs | Artifacts |
|---|---|---|
| DINOv3 | `dinov3-convnext-tiny-pretrain-lvd1689m`, `dinov3-vits16-pretrain-lvd1689m`, `dinov3-vits16plus-pretrain-lvd1689m` | Existing XML/BIN IR, RGB normalized NCHW 224 |
| YOLO-World | `v2:640`, `v2:1280`, `v2.1:640`, `v2.1:1280` | Existing XML/BIN IR, letterboxed RGB; three raw heads |
| UNet | `unet_carvana_scale0.5_epoch2`, `unet_carvana_scale1.0_epoch2` | Existing XML/BIN IR; cubic resize to the deployment scale (0.5 / 1.0 of source), unnormalized 0..1 RGB, no crop |
These are numerical inference tests, not detection decoding/NMS or business
quality evaluation.

## Build and regression tests

Requires CMake >= 3.16, C++17, OpenVINO Runtime SDK, OpenSSL Crypto,
libjpeg, libpng and nlohmann-json. Local framework reference tests use a
configured Python environment.

```bash
CPP=$PWD
PYTHON=${PYTHON:-python3}
OPENVINO_CMAKE_DIR=/path/to/openvino/lib/cmake/openvino
mkdir -p "/tmp/ovbench-$(id -u)"
RUN="$(mktemp -d "/tmp/ovbench-$(id -u)/validation-XXXXXXXX")"
cmake -S "$CPP" -B "$RUN/build" -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINO_DIR="$OPENVINO_CMAKE_DIR"
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

## Backend selection (C++ or Python)

Both engines expose the same CLI surface, modes and report layout:

| Backend | Entry point | Implementation |
|---|---|---|
| `cpp` | `ov_benchmark` (built from `src/`) | Native OpenVINO C++ API |
| `python` | `python -m ov_bench` (package [python/ov_bench/](../python/ov_bench/)) | OpenVINO Python API engine, same options, `results.json` / `summary.tsv` layout and modes |

The root [`ov_bench`](../ov_bench) script runs either engine with the same
command line:

```bash
./ov_bench --backend cpp    --mode self-test
./ov_bench --backend python --mode self-test
./ov_bench --mode self-test        # prefer the built C++ binary, else Python
```

`--backend` / `OVBENCH_BACKEND` selects the engine; `--binary` /
`OVBENCH_CPP_BIN` points at a built `ov_benchmark`; `OVBENCH_PYTHON` chooses
the Python interpreter (needs an OpenVINO + numpy environment). Without an
explicit selection a runnable `build/ov_benchmark` (or `OVBENCH_CPP_BIN`)
selects `cpp`, otherwise the Python backend is used.

Covered modes on both backends: `performance`, `accuracy`, `stability`, `all`,
`preprocess`, `single`, `random-accuracy`, `tolerance-search`, `self-test`,
`gather-guard` and `compare-stability`. Exit-code contract: `0` all checks
passed, `1` a result-level check did not pass (e.g. a FAILED accuracy case),
`2` usage error, identically on both engines (exception classes
`OVBError`/`std::invalid_argument` vs. `OVRuntimeError`). The single-mode stage
timeout is the one exception to this table: it bypasses normal error handling
in both engines and exits `124` after writing `timeout.json`. Remaining documented
differences between the backends:

- Generated random input *streams* are not bit-identical between engines
  (C++ `std::mt19937_64` + `std::normal_distribution<float>` vs. seeded numpy
  `Generator`/PCG64 + ziggurat normal). The generation algorithms differ at
  both the bitstream and the transform level, so bit-exact unification would
  require reimplementing one engine's full RNG stack inside the other
  (fragile, maintenance-sensitive); the supported way to obtain identical
  inputs across engines is the prepared-inputs/`--manifest` path, which the
  accuracy and single-item suites use. Reproducibility holds per backend for
  a fixed `--seed`; for cross-engine `random-accuracy` comparisons, compare
  against recorded input hashes, not assumed stream positions.
- BF16 outputs materialize through the Python runtime as a bit-identical
  float16 value view, so byte-level md5 comparisons match the native raw-byte
  handling.

## Active files

| Purpose | Entry points |
|---|---|
| Backend selector (C++/Python, same CLI) | [ov_bench](../ov_bench) (root script), [scripts/build_cpp.sh](../scripts/build_cpp.sh) |
| Python backend (all modes, same report layout) | [python/ov_bench/engine.py](../python/ov_bench/engine.py) via `python -m ov_bench` |
| Native shared functionality | [src/core.cpp](../src/core.cpp), [src/stability.cpp](../src/stability.cpp), [src/image.cpp](../src/image.cpp) |
| Random accuracy and tolerance search | [src/random_accuracy.cpp](../src/random_accuracy.cpp), [tools/random_reference_worker.py](../tools/random_reference_worker.py) |
| Accuracy summaries | [tools/summarize_random_accuracy.py](../tools/summarize_random_accuracy.py), [tools/summarize_random_accuracy_versions.py](../tools/summarize_random_accuracy_versions.py) |
| Prepared datasets | [tools/export_reference.py](../tools/export_reference.py) |
| Reference: bound performance/stability matrix (Python, image models) | [examples/tools/run_multi_instance.py](../examples/tools/run_multi_instance.py) |
| Native C++ pipeline — build + all items (registered models) | [/home/sgui/2026/ov_bench/run_all_models_cpp_perf_stability_accuracy.sh](../run_all_models_cpp_perf_stability_accuracy.sh) |
| Native C++ all-items orchestrator (general, supports `--model-file`) | [tools/ov_bench_all.sh](../tools/ov_bench_all.sh) |
| Native C++ test items (single-process / core-scaling / multi-instance / summary) | [tools/ov_bench_single.sh](../tools/ov_bench_single.sh), [tools/ov_bench_corescaling.sh](../tools/ov_bench_corescaling.sh), [tools/ov_bench_multi.sh](../tools/ov_bench_multi.sh), [tools/ov_bench_summary.sh](../tools/ov_bench_summary.sh) |
| Reference: Python IR benchmark worker (memory diagnostics) | [examples/tools/python_ir_benchmark.py](../examples/tools/python_ir_benchmark.py) |
| Reference: shared dataset/manifest helpers | [examples/tools/benchmark_common.py](../examples/tools/benchmark_common.py) |
| Reference: memory diagnostics | [examples/tools/check_precision_memory.py](../examples/tools/check_precision_memory.py) (Python IR) |
| CPU plugin deployment with guard/rollback | [tools/deploy_cpu_plugin.py](../tools/deploy_cpu_plugin.py) |
| Reference: real-model preprocessing validation | [examples/tests/validate_models.py](../examples/tests/validate_models.py) |

The files under [examples/](../examples/README.md) are reference implementations
for the specific models originally benchmarked here (DinoV3 / YOLO-World /
UNet). To test your own model use the generic entry points above
(`--model-file` / `--input-shape`); they do not depend on anything in
`examples/`.

Historical fixed-source/pip comparison, one-off deployment and fixed
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

For generated inputs, `--input-count N` controls the random tensor
pool (default 10). A count of 1 binds one input set and reuses it; a larger
count cycles through pre-generated tensors and rebinds each request. In multi
mode, every worker process generates its own in-memory pool and writes to its
own result directory; this works with bound or unbound taskset modes.

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
`--seed N` is configurable and defaults to 43 in the native runner. Pass the
same seed to repeated runs when reproducing candidate inputs; historical
reports retain their recorded seeds.

Only dynamic leading batch resolves automatically to 1. Other dynamic
dimensions require `--random-shapes FILE`, keyed by model ID and input name.

Reference priority is ONNX Runtime CPU FP32, recognized PyTorch CPU FP32,
then native OpenVINO FP32 when external assets are absent. An available but
broken framework reference is an error, not a reason to fall back.
`--reference-config FILE` explicitly selects reference backends and mappings.
For example:

```json
{
  "MODEL_ID": {"backend": "openvino"}
}
```

Use `"backend": "onnx"` for an explicit ONNX Runtime FP32 reference.
Specify every model to force one baseline for the entire selection.
The worker's [configuration schema](../tools/random_reference_worker.py) also supports
PyTorch/custom adapters and tensor-name mappings.

OpenVINO reference runs use the same runtime's FP32 InferRequest, creating it
even if FP32 is not selected. Its FP32 row is self-comparison, not independent
correctness or repeatability validation. This baseline measures precision
drift, not framework correctness. Each reference Python process handles
framework execution only; all tested OpenVINO inference stays in the engine
process that owns the benchmark (C++ binary or Python backend).

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

Both engines start the reference worker with an explicit
`TBB_MAX_THREAD_NUM`/`OMP_NUM_THREADS` cap of `max(4, 2*threads)` (the
inference session itself still runs with `--threads`), so the worker's READY
handshake cannot be starved on large hosts by oversized global thread arenas.

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

Runtime failures are reported rather than bypassed with model-specific shape
workarounds.

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
minima or business acceptance criteria.

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

[export_reference.py](../tools/export_reference.py) exports image-model
input/reference tensors. Pass the resulting manifest explicitly. `--mode accuracy`
retains dataset-level aggregate cosine behavior
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
Unit tests also cover random references, tolerance search, exact integer
handling, variance, stability and default graph policies.
