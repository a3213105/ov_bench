# C++ Test Pipeline

`scripts/build_cpp.sh` builds the `ov_bench` executable; `tools/ov_bench_all.sh`
then runs every test item over the **registered** model families (select
`--suites dinov3,yolo-world,pred,unet` or a subset) or over an **arbitrary
model** via `--model-file` (and `--input-shape` for dynamically-shaped IR):

```bash
# 1) Build the native binary (once per OpenVINO SDK):
bash /home/sgui/2026/ov_bench/scripts/build_cpp.sh \
  --openvino-libs /path/to/libopenvino.so
source /home/sgui/2026/ov_bench/build/ov_bench_runtime.env

BINARY=/home/sgui/2026/ov_bench/build/ov_bench

# 2) Registered-model full-items run:
bash /home/sgui/2026/ov_bench/tools/ov_bench_all.sh \
  --binary "$BINARY" \
  --suites dinov3,yolo-world,unet \
  --mode multi --threads 12 --taskset-bind 0 \
  --instances 24,32,36,38 --warmup 3 --iterations 200 --batch 0,1,2

# Arbitrary-model run (same items; pass the model files directly):
bash /home/sgui/2026/ov_bench/tools/ov_bench_all.sh \
  --binary "$BINARY" \
  --model-file /path/to/model.onnx --suites generic \
  --input-shape <input_name>=N,C,H,W   # only if the model has dynamic shape
```

The orchestrator (`tools/ov_bench_all.sh`) invokes exactly one item entrypoint
per test item, sharing a single `--run-dir`; it also `--dry-run`s downward so a
top-level `--dry-run` previews every item. Dispatch rules: single-process runs
in `single`/`all` modes; the core-scaling sweep (default `--core-scaling 1,2,4,8`,
disable with `--no-core-scaling`) runs as an additional item after the
single-process item and is ignored (with a notice) in pure `multi` mode;
multi-instance runs in `multi`/`all` modes; the summary item runs whenever the
single-process or multi-instance item ran.

`--batch 0|N[,0|N...]` runs the listed variants in order; `0` means dynamic
batch, and each positive integer selects that static batch size. A single
`--batch 0` runs dynamic only. With `--sync-start 1`, each batch variant gets a
separate worker barrier; `2` runs both independent and synchronized waves. The
batch value is included in summaries.

## Unified Results

The final table is `<run-dir>/summary_all.tsv`, generated automatically after
the selected stages finish. It now includes both single-process and
multi-instance results; it is no longer a single-process-only merge. Each row
represents one actual configuration. Open it as a tab-separated table to filter
or compare combinations without combining unlike experiments.

| Columns | Meaning |
|---|---|
| `test_type`, `suite`, `model`, `precision` | Single/multi stage and model identity |
| `taskset_bind`, `cpu_pinning`, `hyper_threading`, `sync_start` | Requested binding, pinning, HT and synchronization configuration |
| `instances`, `instances_completed` | Requested/automatic worker count and completed count |
| `base_threads_per_instance`, `threads_per_instance` | Base threads and requested worker threads after HT doubling |
| `warmup`, `cooldown`, `perf_count`, `stability_rounds` | Additional test settings |
| `measurement_mode`, `iterations`, `duration_seconds` | Count/time budget; inactive budget is blank |
| `input_samples_per_instance`, `total_timed_inferences` | Dataset size and actual completed timed calls |
| `mean_ms`, `max_instance_p99_ms`, `max_instance_elapsed_seconds` | Count-weighted mean inference latency, largest worker P99 and largest worker elapsed time |
| `total_throughput_fps`, `throughput_per_instance_fps`, `throughput_basis` | Sum of worker FPS, average worker FPS, and wall-clock/inference-only definition |
| `accuracy_passed`, `md5_consistent`, `stable_outputs_finite`, `status` | Original validation data/status, when that test was actually performed |
| `source_summary`, `result_dir` | Original table and detailed result location |

For single-process rows, instance count is 1, external binding/synchronization
are 0, and HT is `DEFAULT` because that stage does not explicitly configure HT.
For bound multi-instance rows, HT `DEFAULT` means OpenVINO-default behavior, not
an assertion that HT is enabled or disabled. Thread columns record requested
values; worker JSON retains the effective runtime properties.

Multi-instance performance does not run accuracy or MD5 repeatability checks,
so those cells read `N/A` rather than reported as passing. P99 is **not** pooled
across instances: the table retains the maximum worker P99. Mean latency is
weighted by each worker's completed inference count. Different configurations
are never averaged into one row. Inactive/missing values (e.g. the non-active
measurement budget, accuracy columns on performance-only rows) are written as
`N/A`, never left blank; older intermediate reports may lack the newly added
metric/path columns.

Per-suite summaries, `multi_instance_summary.tsv`, per-wave aggregate rows and
per-worker logs/JSON remain available as detailed intermediate artifacts.

## Layout

Top-level entrypoints (invoke these directly; each accepts `--help`):

| Entrypoint | Responsibility |
|---|---|
| `scripts/build_cpp.sh` | Resolve OpenVINO SDK/library paths and build the `ov_bench` binary (standalone CLI) |
| `tools/ov_bench_all.sh` | General orchestrator: runs every applicable test item, sharing one `--run-dir` |
| `tools/ov_bench_single.sh` | Single-process item: perf + accuracy + MD5 stability (`--mode all`) |
| `tools/ov_bench_corescaling.sh` | Core-scaling item: single-process perf swept by physical-core count |
| `tools/ov_bench_multi.sh` | Multi-instance item: worker waves with synchronized starts and HT sweeps |
| `tools/ov_bench_summary.sh` | Summary item: merges per-item summaries into `summary_all.tsv` |

Shared libraries in `scripts/` (sourced by the item entrypoints, never run
directly — except `build_cpp.sh`, which is a standalone CLI):

| Library | Responsibility |
|---|---|
| `parameters.sh` | Test defaults, help and command-line parsing (incl. `--model-file` / `--input-shape` / generic suite) |
| `validate.sh` | Parameter, suite and required-path validation, dispatched by `OVB_STAGE_ACTIVE` |
| `build_cpp.sh` | Standalone build entrypoint: resolves OpenVINO SDK/library paths, builds `ov_bench` (Release), writes `<build-dir>/ov_bench_runtime.env` |
| `ovbench_lib.sh` | Common bootstrap: sets roots, sources the other libraries, defines `set_model_args` / `set_input_args` / `suite_models` / `ovb_init` / `log` |
| `numa.sh` | Generate NUMA-local, core-scaling and per-worker CPU assignments |
| `summary.sh` | Aggregate per-instance metrics and merge suite summaries |

Each item entrypoint sets `OVB_STAGE_ACTIVE` before sourcing `ovbench_lib.sh`,
which selects the right validation path. Items share the parsed configuration
without reparsing arguments. Scripts in `scripts/` are model-agnostic; scripts
tied to specific benchmarked models (DinoV3/YOLO-World reference matrices) live
under `../examples/` and are outside this pipeline.

`--mode single` runs ordinary performance, precision and MD5 stability tests.
`--mode multi` runs performance waves only; `--mode all` runs both stages.
Multithreaded performance waves never overlap different models. Disabling
`--taskset-bind` also disables OpenVINO CPU pinning for multi-instance workers.
`--taskset-bind 0` runs unbound: `--instances` selects the count sweep.
`--taskset-bind 1` runs bound: instance counts are automatic and `--instances`
is ignored. `--taskset-bind 2` runs both: for each model, finish all bound
precision tests first, then run the unbound instance-count sweep. Modes 0 and 2
require `--instances`. Each bound worker binds
`--threads` physical cores and all their logical siblings within a NUMA node;
remaining cores that cannot form a complete group are idle.
Binding uses `taskset --cpu-list`: CPU assignments remain NUMA-local, but memory
allocation is not restricted to that NUMA node (there is no `--membind`).
`--numactl-bind` remains a compatibility alias for `--taskset-bind`; both accept
0, 1 and 2. The canonical option and aggregate field are `taskset-bind` and
`taskset_bind` respectively.
Outputs are separated under `multi_instance/<suite>/<model>/bind_1/` and
`bind_0/`. The aggregate TSV includes `taskset_bind` (1 or 0) for each wave.

```bash
bash /home/sgui/2026/ov_bench/tools/ov_bench_all.sh \
  --binary /path/to/ov_bench \
  --mode multi --threads 12 --taskset-bind 2 \
  --instances 24,32,36,38 --warmup 3 --iterations 200
```
Total FPS is the sum of worker FPS; matrix P99 is the largest worker P99.

## Unbound Hyper-Threading

`tools/ov_bench_all.sh` accepts `--hyper-threading 0|1|2` for unbound multi-instance waves:

| Value | ENABLE_HYPER_THREADING | Requested INFERENCE_NUM_THREADS |
|---|---|---|
| `0` | NO | `--threads` |
| `1` | YES | `--threads * 2` |
| `2` (default) | YES then NO, in separate waves | Doubled, then baseline |

CPU pinning remains NO in every unbound wave. Binding mode is unaffected: its
core assignments, thread count and OpenVINO-default HT behavior remain unchanged.
With `--taskset-bind 2`, each model completes its bound tests first, then all
HT-ON unbound instance counts/precisions, then all HT-OFF unbound cases, before
the next model starts. This option does not change ordinary single-process tests.

Unbound reports live under `bind_0/ht_YES/` and `bind_0/ht_NO/` to prevent
overwrites. Summary columns include `hyper_threading` (YES/NO/DEFAULT),
`base_threads_per_instance`, and `threads_per_instance` (the requested worker
thread count after doubling). Each worker JSON retains the effective OpenVINO
properties; inspect those for the actual runtime configuration.

```bash
bash /home/sgui/2026/ov_bench/tools/ov_bench_all.sh \
  --binary /path/to/ov_bench \
  --mode multi --taskset-bind 0 --threads 12 \
  --instances 24,32,36,38 --hyper-threading 2 \
  --warmup 3 --iterations 200
```

The native C++ executable uses `--hyper-threading YES|NO` and an already
calculated `--threads` count; only the pipeline performs the automatic doubling.

`--sync-start 0|1|2` controls the multi-instance warmup barrier and defaults to 0.
By default workers receive no `--start-signal` and start timing independently
after their own warmup. With `--sync-start 1`, the pipeline waits for all workers
to be ready and creates their shared start signal; `--start-timeout` applies only
in this mode. This option is forwarded by the compile-first entry point.
Both modes wait for every worker to finish before proceeding to another model.
With `--sync-start 2`, each binding/HT/instance-count/precision configuration is
tested first with independent starts (0), then with synchronized starts (1).
The two waves never overlap. Results are stored under
`<configuration>/sync_0/` and `<configuration>/sync_1/` for every synchronization
selection; the multi-instance TSV appends `sync_start` to distinguish the rows.
The setting is forwarded by the main compile-first entry, for example:

```bash
--mode multi --taskset-bind 0 --threads 12 --instances 32 --sync-start 2 --duration 60
```
Multi-instance performance workers use `--no-performance-output-files`: output
checks retain MD5, shape, dtype, byte size and finite-value metadata in JSON,
without writing per-input `check_*.bin` tensors. This avoids multiplying large
output files by the number of workers and configurations. Stability-mode output
files are unchanged. Existing reports are never automatically deleted.
Choose one performance budget: `--iterations N` or `--duration SECONDS`.
`--iterations` is the exact number of timed inference calls per worker.
If neither is supplied the pipeline defaults to 100 iterations (native C++: 200).
Supplying `--duration` replaces that default; explicitly supplying both options
is rejected, regardless of their order. Duration accepts positive finite seconds,
including fractions such as `0.5`.

Duration is measured independently for each worker/model/precision/HT/binding
case. Its monotonic wall-clock timer begins after warmup and any synchronization
wait. Inputs continue cycling until the duration expires. An in-flight inference
is allowed to finish, so elapsed time can exceed the target; at least one
inference is completed. Compilation, warmup, cooldown, output checks and report
writing are not included in this duration. It is not a timeout for a stuck infer.

Duration-mode `throughput_fps` is completed timed calls divided by actual
measurement wall time; latency percentiles still use per-call inference times.
`inference_throughput_fps` separately records throughput from summed inference
latencies. Iteration-mode `throughput_fps` retains its existing inference-only
definition. Worker JSON records `measurement_mode`, `requested_duration_seconds`,
`elapsed_seconds` and actual `samples`/`iterations`; requested iterations are null
in duration mode. The per-worker TSV adds `timed_inferences` and `elapsed_seconds`.
The multi-instance TSV adds `duration_seconds` and `total_timed_inferences`;
its requested `iterations` column is blank in duration mode. Aggregate FPS remains
the sum of worker FPS, not throughput across the entire pipeline runtime.

```bash
bash /home/sgui/2026/ov_bench/tools/ov_bench_all.sh \
  --binary /path/to/ov_bench \
  --mode multi --taskset-bind 0 --threads 12 --instances 32 \
  --duration 60 --warmup 3
```

`--warmup` is the number of untimed warmup calls and also the cooldown call count.
Input samples advance sequentially and wrap around; the cursor continues across
all three stages.
Native FP32 references measure precision drift, not independent framework
correctness. Multi-instance-only mode measures performance, not accuracy or MD5
repeatability; use `--mode all` to retain the ordinary validation tests as well.

## Parameter Migration

Removed parameter names are rejected rather than silently accepted.

| Previous options | Current option |
|---|---|
| `--threads`, `--instance-threads` | `--threads` for every process; default 12 |
| `--model-root-images`, `--model-root-unet` | `--ir-root` |
| `--multi-instance-counts` | `--instances` |
| `--multi-instance-only` | `--mode multi`; use `--mode all` for both stages |
| `--rounds` | `--stability-rounds` |
| `--multi-start-timeout` | `--start-timeout` |
| `--cpu-pinning` | Derived from `--taskset-bind` for workers; single-process tests use NO |
| `--openvino-lib-dir`, `--openvino-so-path` | `--openvino-libs` accepts a directory or library file |
| Compile entry `--binary`, `--test-script` | Removed; use `--build-dir`, fixed test pipeline |

`--openvino-dir` is retained because compilation needs an SDK with headers and
CMake targets, not just a shared library. Optional model manifests may use
distinct input/reference formats. Per-family model selectors and profiling are
retained because they control independent behavior.