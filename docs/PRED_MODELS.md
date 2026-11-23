# Prediction model tests

The three models share [src/main.cpp](./src/main.cpp) with DINOv3 and
YOLO-World. Model-specific loading, named inputs/outputs and policies are in
[src/application.cpp](./src/application.cpp).

| Short name | Model ID | Outputs |
|---|---|---:|
| attention | `26Q2_round13_run2_e40_no_map_attn` | 11 |
| ped | `ped_26Q1_history_data_gen4_800v2_issue2_e23_refine` | 5 |
| vehicle | `vehicle_q2_release_v0_w_leading_25` | 15 |

Original ONNX files are loaded directly. Prediction inputs remain dynamic in
every precision; random/prepared synthetic samples use batch 1. Synthetic
masks assume True means valid. These are numerical tests, not business
accuracy certification. See [build and shared CLI](./README.md).

## Current accuracy workflow

```bash
CPP=/home/sgui/2026/ov_bench
PYTHON=/home/sgui/miniforge3/envs/didi_t213/bin/python
MODELS=26Q2_round13_run2_e40_no_map_attn,ped_26Q1_history_data_gen4_800v2_issue2_e23_refine,vehicle_q2_release_v0_w_leading_25
"$BIN" --mode random-accuracy --model "$MODELS" \
  --model-root /home/sgui/2026/models --reference-root /home/sgui/2026/models \
  --reference-python "$PYTHON" --reference-config /path/to/reference.json \
  --precisions FP32,BF16,FP16 --iterations 100 --seed 43 \
  --threads 32 --cpu-pinning NO --perf-count NO --output "$RUN/accuracy"
```

For each model, explicitly select `"backend": "onnx"` or
`"backend": "openvino"` in the reference JSON. Run both baselines in separate
new output directories using the same seed. Input hashes must match across
precision and reference runs. ONNX means ONNX Runtime CPU FP32; OpenVINO means
the same runtime/model's native FP32 output. Its FP32 row is self-comparison,
not independent correctness or repeated-run stability verification.

Every input/output JSON records cosine, maximum/mean absolute error,
absolute-error population variance, tolerance exceedances and hashes.
Integers require exact equality. attention and vehicle expose int32
`modals_index`; ped has no integer output. `outputs.tsv` summarizes each
output over inputs, including pooled absolute-error variance. The variance
is not signed-error variance or MSE.

The joint criterion requires each floating output's cosine > 0.99, finite
values and all elements within `atol + rtol*abs(reference)`. Precision-specific
defaults/overrides and zero-norm behavior are documented in
[accuracy criteria](./README.md#criteria-and-metrics).

To collect 100 accepted inputs instead of 100 raw candidates, add
`--resample-common-cosine-failures` to the command above. Each candidate is
shared by all precisions; it is rejected only if both BF16 and FP16 fail
their all-output cosine checks. Fresh candidates are generated until 100
are accepted, without a retry limit. Rejected inputs, hashes and output
metrics are recorded separately and their raw tensors retained. Joint
accuracy thresholds remain unchanged. This introduces selection bias;
see [sampling semantics and records](./README.md#optional-shared-input-rejection-sampling).
With this option, separate reference runs can retain different subsets:
compare input hashes rather than assuming a common seed guarantees a
paired accepted dataset.

## Prepared inputs, performance and stability

```bash
"$PYTHON" "$CPP/tools/prepare_pred_models.py" \
  --model-root /home/sgui/2026/models/pred_model \
  --output "$RUN/prepared" --samples 10
COMMON=(
  --model-root "$RUN/prepared" --manifest "$RUN/prepared/manifest.json"
  --prepared-inputs --model "$MODELS" --precisions FP32,BF16,FP16
  --threads 6 --cpu-pinning NO --perf-count NO
)
"$BIN" "${COMMON[@]}" --mode performance --warmup 1 --iterations 100 \
  --output "$RUN/performance"
for process in a b; do
  "$BIN" "${COMMON[@]}" --mode stability --rounds 10 --output "$RUN/stability_$process"
done
"$BIN" --mode compare-stability --stability-dir "$RUN/stability_a" \
  --stability-dir "$RUN/stability_b" \
  --output "$RUN/stability_comparison.json"
```

The preparer stages original ONNX files unchanged, generates named synthetic
inputs and saves ONNX Runtime FP32 references. Prepared `accuracy` uses the
existing aggregate criterion; use `random-accuracy` for per-output joint
tolerance checks. [examples/scripts/benchmark_pred_models.sh](../examples/scripts/benchmark_pred_models.sh)
provides a prepared accuracy/stability convenience wrapper, accepting
`PRED_PYTHON`, `PRED_MODEL_ROOT`, `PRED_MODELS`, `PRED_PRECISIONS`, `PRED_THREADS`,
`PRED_ROUNDS`, `PRED_SAMPLES`, `PRED_REPORT_DIR` and `PRED_BINARY`.

The [shared documentation](./README.md#performance-and-stability) defines FPS,
P99, warmup and sample coverage. Historical didi results used static BF16
reshape; they are not results for current no-reshape code. The old one-off
deployment and 40-vs-38 supplement scripts were removed; reports and remote
deployment snapshots remain untouched.

Use `--mode single` with one prepared model/precision to diagnose one inference
with stage logs and `--step-timeout-seconds`; all native modes share main.

## Native stage memory diagnostic

The local CPU plugin releases its complete transformed-model references at
the end of `compile_model`, after verifying execution graphs are ready,
including sub compiled models. The transformation pipeline also leaves scope
before this release. `C7_source_model_released` records the resulting memory
snapshot. CPU Constants, shape-inference nodes and weights retain their own
shared ownership; no shared node or buffer is forcibly invalidated. Surviving
node references may retain upstream graph structure, so model-reference release
does not imply that all transformed graph allocations are freed.
Export (including automatic compiled-model cache writes) and later graph
reconstruction explicitly fail after release. This local build is intended
for inference without export/cache serialization. Import behavior is unchanged.
The caller's original model is not modified or released by the plugin.

[src/memory.cpp](./src/memory.cpp) reuses the current prediction loading,
random input generator and graph policies. It measures process RSS/PSS/USS at
baseline, input loading, OV plugin/model loading, compilation, request creation,
first inference and every 10 inferences. The application's source OV model is
released immediately after the `model_compiled` snapshot, before request
creation. `source_model_released` records the post-release snapshot so the
two stages separate plugin compilation from application-model retention.
ONNX Runtime CPU FP32 runs in a separate fresh process; its Session creation
combines loading, graph optimization and initialization, so it has no separately
measurable OV-equivalent compilation stage.

USS is `Private_Clean + Private_Dirty` from `smaps_rollup`, excluding
explicit `Private_Hugetlb`.

The local `/home/sgui/ov_build.sh` builds and installs the modified OpenVINO,
then compares OV FP32/FP16/BF16 against ORT FP32 for all three models by default. Set `MODEL` to
`attention`, `ped`, `vehicle`, or one of the model IDs above to test one model.
`MODEL_ROOT` and `INPUTS` override the model directory and input manifest.
The manifest must provide 10 batch-1 samples for each selected model, with
consistent input names and order. The script runs three independent processes
for ORT FP32 and per OV precision (10 inferences each), plus a separate
instrumented OV run and a one-request batch 1/2/3 run per precision.
Set `PRECISIONS=FP32` for the previous FP32-only coverage, or use a
comma-separated subset of `FP32,FP16,BF16`. Duplicate/unknown values fail.
ORT baseline directories are shared under each model; all OV reports are
under `<model>/<precision>/`. The combined `memory_comparison_all_models.tsv`
includes `precision` and `ort_precision` columns. OV hashes are compared only
within the same precision; ORT outputs are checked for finite values, not
cross-backend or cross-precision numerical accuracy. Effective inference hints
are checked and actual `runtime_precision_counts` are recorded in each
precision's summary. A hint does not guarantee every node uses that precision;
inspect `trace/runtime_graph.json` for actual implementations and fallbacks.
With `GRAPH_DISCONNECT_EXPERIMENT=1`, the comparison TSVs also include
`ov_optimized`, `optimized_ov_minus_ov` and `optimized_ov_minus_ort` columns.
These are measurements from the graph-disconnect candidate's isolated
`release_fixed` process, not the three-process median used by the ordinary
`ov` column. The candidate runs with compile tracing enabled; treat small
differences as measurement noise and consult `graph_disconnect/summary.json`
for validation status and stage data.

```bash
MODEL=all PRECISIONS=FP32,FP16,BF16 MLAS_RELEASE_EXPERIMENT=1 bash /home/sgui/ov_build.sh
```

The instrumented run also writes `trace.jsonl.primitive_allocations.jsonl`:
per-node `create_primitive`, `execute_constant`, `constant_cache_lookup` and
`constant_cache_validate` net glibc allocation changes, separated into arena
live, mmap live and arena free bytes. Records are buffered and written after
the graph's primitive loop, not inside measured calls. Reporting produces
`primitive_allocations.tsv`, `primitive_allocations_by_type.tsv`,
`primitive_top_nodes.tsv` (top 50), `primitive_allocation_intervals.tsv` and
`primitive_allocation_summary.json`. Each graph's measured sum is reconciled
with its loop total and the C5a-to-C5 interval; the latter includes probe
storage/logging. These process-wide deltas are not exclusive object ownership
or allocation peaks and may exclude direct mmap/JIT storage. Concurrent work
can affect them; nested graphs have inclusive call intervals, so do not sum
across graph IDs. RSS/PSS/USS remain phase-level measurements, not per-node
allocation attribution.

`trace.jsonl.primitive_details.jsonl` further splits FullyConnected memory
binding, tensor-parallel setup/splitting, factory creation and base-node
creation. Within the oneDNN FC path it measures attribute construction,
primitive descriptor creation and packed-weight preparation. Subgraph details
separate memory/shape/attribute initialization, IR optimization and base-node
creation; nested IR steps include data-flow transformations, shape inference
and control-flow transformations. These probes are active only inside the
instrumented graph primitive loop on its current thread. Output is buffered
until loop completion. `depth=0` steps can be compared with the outer node
measurement; deeper steps are already included in their parents and must not
be added again. Worker-thread allocations still affect process-wide counters
but do not inherit thread-local probe context.
The report writes `primitive_details.tsv`, `primitive_details_by_stage.tsv`,
`primitive_detail_intervals.tsv` and `primitive_detail_summary.json`. Residuals
are reported explicitly, including unprobed work and local destruction.
Packed-weight deltas indicate preparation cost, not proven duplicate storage.

MLAS additionally records source/packed data addresses, logical byte sizes,
N/K, transpose mode, cache identity and whether the creation callback ran.
`mlas_packed_weights.tsv`, `mlas_weights_by_source.tsv` and
`mlas_weight_summary.json` summarize observed source and packed address spans,
cache hits/misses, and multiple packed buffers for the same source/layout/cache.
The raw file is `trace.jsonl.mlas_weights.jsonl`. Pointer-based grouping does
not detect equal contents at different addresses or establish buffer lifetime;
buffer sizes are not RSS or allocator overhead. Allocation and packing are
also separate nested `fc.mlas.*` steps.

The control-flow diagnostic passes an optional observer into Snippets rather
than introducing a CPU dependency in Snippets. It measures LinearIR building,
shape-inference IR cloning, pipeline registration, and each enabled lowered
pass in the main, validation and pre-generation pipelines. The observer is
absent outside the trace context. `subgraph_control_flow_by_stage.tsv`,
`subgraph_control_flow_intervals.tsv` and `subgraph_control_flow_summary.json`
reconcile these substeps with each node's control-flow parent. Repeated pass
names are aggregated; nested pipelines not passed the observer remain inside
their enclosing pass. Negative residuals can reflect local object destruction,
so positive substep sums must not be interpreted as retained object sizes.

### Destructive OV graph disconnection

```bash
MODEL=all PRECISIONS=FP16,BF16 GRAPH_DISCONNECT_EXPERIMENT=1 bash /home/sgui/ov_build.sh
```

This opt-in experiment runs independent baseline/release cases under
`<model>/<precision>/graph_disconnect`, without enabling FC artifact or MLAS
source deletion. The native flag `OV_CPU_EXPERIMENTAL_DISCONNECT_OV_GRAPH=1`
requires compile tracing. Disconnection now runs immediately after `Graph::Init`
(replication, descriptors and CPU graph optimizations), before `Graph::Activate`
(buffer allocation, primitive creation and constant execution). It enumerates all transformed OV nodes
including nested MultiSubGraphOp bodies, removes their input arguments and
control dependencies. Ordinary complete-model references are released later,
at the end of compilation as before.
It does not explicitly invalidate Constant buffers, clear node attributes,
or delete CPU execution graph nodes. Shared shape-inference OV node aliases
are deliberately mutated: missing inputs can cause exceptions, wrong shapes,
or native crashes even though weights are preserved.

`trace.jsonl.graph_disconnect.jsonl` inventories node/input/control-edge counts.
Candidate logs write `begin` before disconnection and `complete` after
disconnection, before activation, with disconnected-edge counts, weak-reference expired/
surviving node counts and glibc live delta. Baseline logs write `inventory`.
Disconnection is not equivalent to physical deletion: independently held nodes,
attributes and body models can remain alive. Model containers remain until the
normal end-of-compilation release; the measured interval no longer includes
that final release. `release_point=after_graph_init_before_activate` distinguishes
these logs from older compile-end experiments. The checkpoint
`C4a_before_activation_after_graph_disconnect` precedes activation, allowing
later allocation reuse to be distinguished from immediate RSS reduction.
The destructive candidate rejects multiple execution graphs or sub streams
before graph construction to avoid disconnecting a source still needed by
another graph. Baseline inventory is taken at the same earlier boundary.
Moving the experiment earlier can now fail during compilation itself if
activation still requires original OV inputs; isolated error capture remains
enabled. No compile/inference safety is implied by prior compile-end PASS.
Only the transformed model is mutated; application source-model connections are
not touched (its reference is released by the benchmark after compilation).

The controller collects fixed and dynamic cases even after candidate failure,
records signals/exit status and errors, checks full edge disconnection against
baseline inventory, and compares same-precision hashes/shapes. It disables
core dumps in child processes. A model's failed experiment stops the outer
build script after its four cases have been collected; test other models
individually if needed. No compile/test command is executed just by editing.

### Pre-first-inference FC compile artifacts

`FC_ARTIFACT_EXPERIMENT=1` enables separate same-precision baseline/release
processes under `<model>/<precision>/fc_compile_artifacts`. For example:

```bash
MODEL=all PRECISIONS=FP16,BF16 FC_ARTIFACT_EXPERIMENT=1 bash /home/sgui/ov_build.sh
```

The opt-in CPU flag `OV_CPU_EXPERIMENTAL_RELEASE_FC_COMPILE_ARTIFACTS=1`
requires compile tracing. Immediately after FC primitive creation, it releases
only the factory and the FC node's own post-op vector, and only for FP16/BF16
with the exact default oneDNN FC executor type. That executor stores attributes
by value and does not use either object for dynamic parameter updates. The
executor's own attributes, shape-agnostic data, weights, memory bindings and
caches remain intact. Other executor types are inventory-only: notably MLAS
holds a reference to factory attributes, so resetting that factory would create
a dangling reference. Variable/fallback executors are conservatively excluded.

`trace.jsonl.fc_compile_artifacts.jsonl` inventories each factory and post-op
vector even without release. Each successful erasure records `complete` and
the immediate glibc live delta, including deallocated nested payload; reported
inline capacities alone do not measure payload or RSS. The controller checks
that every eligible inventoried artifact was released, records compile-stage
RSS/PSS/USS and allocator deltas, and verifies later fixed/dynamic inference
hashes against baseline. No eligible objects yields `NOT_APPLICABLE`, not
release success. This is a bounded experiment, not evidence that all retained
compile-time memory is disposable. Ordinary summaries remain baseline-only.

### Runtime memory attribution

The local build script additionally runs `runtime` (10 fixed-batch calls) and
`runtime_dynamic` (batch 1/2/3) for every precision. These independent processes
set `OV_CPU_RUNTIME_MEMORY_STATS_PATH` and do not release MLAS source buffers.
Only these diagnostic processes serialize dynamic shape/parameter updates to
avoid overlapping allocation attribution; ordinary timing/memory processes
retain their original scheduling. Fixed and dynamic diagnostic output hashes
are checked against the corresponding normal runs at the same precision.
Diagnostic timing/RSS is not a replacement for the three-process medians.

The buffered JSONL records request input binding/copy, graph memory allocation,
per-node shape/output-buffer updates, parameter preparation, execution and
output handling. oneDNN executor details split primitive creation (including
FC cache-miss construction), memory binding, weight allocation/reorder,
private weight-cache hits/results and scratchpad creation/NUMA rebinding.
Scratchpad events include both logical descriptor sizes and the shared backing
block's capacity; oneDNN scratchpads already share storage in this path.
Node records use process-wide net glibc live/mmap/free bytes; request/graph
phases additionally record RSS. There is no per-node RSS/peak attribution.
Thread-local context does not attribute allocations made outside these call
intervals; nested graphs are inclusive.

[summarize_runtime_memory.py](../tools/summarize_runtime_memory.py) generates
`runtime.jsonl.by_stage.tsv`, `.top_nodes.tsv`, `.intervals.tsv`,
`.buffers.tsv` and `.summary.json`, with corresponding `runtime_dynamic.jsonl`
outputs. Requests have ordered iterations; group only equal depth and never
sum children with their parents. Residuals reconcile request/graph totals with
their immediate children. Buffer inventories deduplicate identical address/size
pairs, not overlapping spans or allocations across time. They are observations,
not proof that all objects are still retained or exclusively owned.

Optimization candidates require subsequent validation: repeated weight creation
or multiple cached layouts may justify cache reuse/eviction; scratchpad logical
sizes must be checked for shared storage before reducing them; shape-update
growth may be essential outputs/intermediates; first-only execution growth may
be JIT/cache initialization or page touching. Do not delete these allocations
solely because they are large. Compare first vs subsequent calls and batch
1/2/3 before changing retention policies.

### Destructive MLAS source release experiment

The local build script accepts `MLAS_RELEASE_EXPERIMENT=1`, off by default.
For example, `MODEL=ped MLAS_RELEASE_EXPERIMENT=1 bash /home/sgui/ov_build.sh`
runs the normal comparison first, then invokes
[check_mlas_source_release.py](../tools/check_mlas_source_release.py) in a separate
report subdirectory `<model>/<precision>/mlas_source_release` for each selected
precision. The controller also accepts `--precision FP32|FP16|BF16`
(standalone default: FP32).

The experimental CPU flag `OV_CPU_EXPERIMENTAL_RELEASE_MLAS_SOURCE=1` requires
`OV_CPU_COMPILE_MEMORY_STATS_PATH`. After all primitives and constant execution
in the graph loop, it matches observed MLAS sources to CPU Input Constants'
owning AlignedBuffers and move-assigns empty buffers into those owners. This
actually deletes the original owned allocations and clears owner pointers to
avoid double deletion at teardown. It deliberately invalidates aliases in the
application model, transformed graph and other consumers; CPU Memory views
can become dangling. It is not a supported optimization or export-safe model.
Non-owning/shared AlignedBuffer subtypes, missing source owners and overlapping
packed buffers cause explicit errors before deletion. Larger containing owner
buffers can include other data: this experiment does not guarantee exclusive
MLAS use.

The controller runs baseline/release cases in separate processes for both
10 fixed-batch calls and batch 1/2/3 using one request. It captures exit codes,
signals, native errors, memory stages and release-event logs even if a candidate
crashes, disables child core dumps, and compares outputs against the same-build
baseline at the same precision, not FP32. If a successful baseline's MLAS
trace is present but empty, its release case is explicitly `NOT_APPLICABLE`
and not run: FP16/BF16 can select other implementations. Both fixed and dynamic
baselines still run; missing traces, inference errors and mismatched effective
precision hints remain failures. No-MLAS coverage is not reported as successful
weight deletion. A failure is reported after collecting the applicable cases. Release
completion and positive released byte counts are required for PASS; merely
removing a reference is not considered a release. Logs use
`*.mlas_source_release.jsonl`; controller results are in `summary.json`.
Even PASS does not establish safety for other shapes, streams, future graph
creation, export or other models. These instrumented single-process-pair memory
deltas are diagnostics, not statistically verified optimization savings.

Build the optional `pred_memory` target with a version-matched ONNX Runtime
C++ SDK and library; the normal build is unchanged:

```bash
cmake -S "$CPP" -B "$RUN/build" -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINO_DIR=/home/sgui/miniforge3/envs/didi_t213/lib/cmake/openvino2025.4.1 \
  -DOVBENCH_BUILD_MEMORY=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/ort/include \
  -DONNXRUNTIME_LIBRARY=/path/to/libonnxruntime.so
cmake --build "$RUN/build" --target pred_memory -j 4
LD_LIBRARY_PATH=/path/to/ort/lib "$RUN/build/pred_memory" --prepare \
  /home/sgui/2026/models "$RUN/memory_inputs"
"$PYTHON" "$CPP/examples/tools/run_pred_memory.py" --binary "$RUN/build/pred_memory" \
  --model-root /home/sgui/2026/models --prepared "$RUN/memory_inputs" \
  --ort-library /path/to/libonnxruntime.so \
  --cpu-plugin /path/to/libopenvino_intel_cpu_plugin.so \
  --threads 6 --iterations 1000 --output "$RUN/memory_report"
```

The preparer generates 10 batch-1 samples per model with the original C++ seed43
generator. All four cases rotate the same bounded pool and check every output
for finite values on every inference. Exactly 1000 total calls include the first
inference, not 1000 calls after warmup. The controller runs all 12 cases serially,
archives inputs, source, executable, library/model hashes, native logs, stage
TSV and a Chinese report. RSS deltas after the first and tenth calls distinguish
initial lazy allocation from subsequent growth. No allocator trimming is done.
This is not a changing-batch, repeated-compilation or multi-request test.

The pip ORT package may provide a versioned library without its SONAME
`libonnxruntime.so.1` link. Supply a local link for the preparer if needed; the
controller creates that link only inside its report directory and does not
modify installed runtimes. The binary links both framework shared libraries;
baseline-subtracted increments are also reported to reduce this baseline bias.

### Compile-only snapshots

For the current investigation, use [check_compile_memory.py](../tools/check_compile_memory.py)
with the complete instrumented wheel installed in conda `ov`. This path uses
only OpenVINO, no ORT, prepared inputs, InferRequest or inference. It retains
the source and compiled models through the compile-return snapshot and performs
release diagnostics afterwards, without malloc_trim or reshape.

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH \
  /home/sgui/miniforge3/envs/ov/bin/python \
  /home/sgui/2026/ov_bench/tools/check_compile_memory.py \
  --model-root /home/sgui/2026/models/pred_model \
  --precisions FP32 --threads 6 --repeats 1 \
  --output /home/sgui/2026/reports/compile_only_snapshots_NEW
```

Rebuild the full wheel with `make -C /home/sgui/openvino_sgui/build -j4 ie_wheel`
and force-reinstall it using the `ov` interpreter's pip. An existing wheel may
not regenerate after CPU source changes: back up/move the old wheel before
building, then verify snapshot markers in the newly packaged plugin.
Do not inject source SOs into the installed wheel. The worker verifies loaded
OpenVINO/TBB paths and fails if the wheel lacks internal snapshots.
`snapshots.tsv` and `summary.json` retain RSS/PSS/Anonymous/Private_Dirty and
glibc live/free/mmap allocation values. These localize stages, not ownership
or proof that retained allocations are unused during later inference.

### CPU compile ledger

The source build at `/home/sgui/openvino_sgui` supports opt-in Release-safe
`OV_CPU_COMPILE_MEMORY_STATS_PATH` checkpoints. Use matching source SDK,
core, CPU plugin and TBB, not a source plugin mixed with a pip runtime.
No ownership or cleanup policy changes. This first instrumentation records
process memory, glibc `mallinfo2`, model/node/edge counts and Constant logical
byte spans deduplicated by overlapping addresses. These are not allocator
capacities or a complete object ledger; the 80% accounting gate is unverified.

```bash
CPP=/home/sgui/2026/ov_bench
OV=/home/sgui/openvino_sgui
PYTHON=/home/sgui/miniforge3/envs/ov/bin/python
SDK=/tmp/ovbench-1003/compile-ledger-sdk/onnxruntime-linux-x64-1.23.2
BUILD=/tmp/ovbench-1003/compile-ledger-source-build
cmake --build "$OV/build" --target openvino_intel_cpu_plugin -j 4
cmake -S "$CPP" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINO_DIR="$OV/build" -DOVBENCH_BUILD_MEMORY=ON \
  -DONNXRUNTIME_INCLUDE_DIR="$SDK/include" \
  -DONNXRUNTIME_LIBRARY="$SDK/lib/libonnxruntime.so"
cmake --build "$BUILD" --target pred_memory -j 4
export LD_LIBRARY_PATH="$OV/bin/intel64/Release:$OV/temp/Linux_x86_64/tbb/lib:$SDK/lib"
INPUTS=/tmp/ovbench-1003/compile-ledger-inputs-new
"$BUILD/pred_memory" --prepare /home/sgui/2026/models "$INPUTS"
"$PYTHON" "$CPP/examples/tools/run_pred_memory.py" --binary "$BUILD/pred_memory" \
  --model-root /home/sgui/2026/models --prepared "$INPUTS" \
  --ort-library "$SDK/lib/libonnxruntime.so" \
  --cpu-plugin "$OV/bin/intel64/Release/libopenvino_intel_cpu_plugin.so" \
  --threads 6 --iterations 1000 --compile-memory-stats --verify-stability \
  --output /home/sgui/2026/reports/compile-ledger-new
```

SDK/build/input paths are scratch paths; prepare an ORT C++ SDK if it is no
longer present. Output and prepared input directories must be new.
`--cpu-plugin` selects the plugin through an explicit XML and validates its
loaded mapping, rather than only recording a provenance hash.
`--compile-memory-stats` writes `compile_memory_stats.jsonl` per OV case and
fails report validation when no complete graph cycle is recorded.
`compile_memory.json` retains raw records and chronological process deltas.
The actual order is generally C0, C1, C2, C4, C5, C6, C3: CompiledModel initializes
and activates graphs inside its constructor. Nested graphs can interleave;
graph IDs identify their records.

`--verify-stability` checks every OV output against its first same-input hash.
It requires at least 20 iterations for the 10-input pool and retains hashes,
not accumulated output tensors. Hashing is outside the timed infer call but
still affects allocations/cache residency. Disable logging and hashing for
performance admission tests. `compile_ms` excludes the test runner's
post-compile memory snapshot but includes enabled plugin instrumentation.
`uordblks` excludes mmap allocations, separately recorded as `hblkhd`.

Reports contain actual runtime versions, compile memory deltas, compile/first
inference times and stability-check flags. They do not certify numerical ORT
agreement, changing batch, export/import, model cache, multiple requests,
sanitizer coverage or performance thresholds. The independent accuracy and
compatibility gates remain necessary before accepting an optimization.

## Required runtime fix

**Use the Gather-patched 2025.4.1 CPU plugin or a verified fixed release.**
The BF16 static-reshape workaround, CLI flag and report fields were removed.
There is no silent fallback to static shapes. Stable opset11 TopK ordering
remains enabled for ties; older TopK opsets do not expose this attribute.

The diagnosed dynamic-batch BF16 crash is caused by Gather JIT incorrectly
handling FP32 input/BF16 output and writing outside its logical output.
Pip versions tested from 2025.4.1 through 2026.2.1 reproduced the crash;
2026.3.0 is the first fixed release in this range. The full-model source
bisect identified [PR #36004](https://github.com/openvinotoolkit/openvino/pull/36004),
commit `7ba34b7ae0efc5d390970054230decd0a6ca1349`, directly after
`12a2726fe3e1c6f7af737286ec1f74e548d43617`. Three models, 1/16 threads and
three processes per combination gave 18/18 crashing parent runs and 18/18
normal fixed runs in the complete repeat. An earlier parent repeat included
a normal exit: lack of a crash alone does not establish memory correctness.
Gather guard tests verified bad writes before and zero bad writes after.
See [retained bisect evidence](../reports/dynamic_batch_source_bisect_20261103/summary.json).

The existing local patched 2025.4.1 plugin also completed the three models
without static reshape. Removing the crash does not imply low-precision
accuracy passes, nor does it cover all changing-batch scenarios.

[deploy_cpu_plugin.py](../tools/deploy_cpu_plugin.py) is the retained generic CPU-only
deployment helper: test candidate with the native Gather guard, back up,
atomically replace, verify loaded plugin and roll back on failure. Candidate
ABI/glibc must match the target host; do not copy the local plugin blindly to
didi. Package reinstall can overwrite a backport without changing version
metadata, so verify plugin hashes and provenance.

The local `didi_t213` patched plugin SHA256 is
`a9888e0f50855db414405f9da1a8771f06f8419d575436a07589ff21d719b5c0`.
Core, frontends and Python bindings are unchanged by this CPU-only backport.
