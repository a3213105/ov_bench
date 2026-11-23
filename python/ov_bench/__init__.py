"""ov_bench Python backend: functional port of the native C++ ov_benchmark engine.

Covers the same user-visible surface (CLI flags, results.json / summary.tsv
layout, modes performance|accuracy|stability|all|preprocess|single|
random-accuracy|tolerance-search|self-test|compare-stability) so the shell
entrypoints under tools/ can drive it in place of the C++ binary
(selected with --backend python).

Porting notes (differences from the C++ engine, documented on purpose):
- Random inputs are generated with a numpy ``Generator`` whose seed is derived
  from (seed, model id). The distribution parameters match the C++ code
  (normal(0,0.1) float, uniform{0,1} integer, Bernoulli(0.75) boolean) but the
  streams are NOT bit-identical to the C++ mt19937_64 sequence. Reproducibility
  holds *within* the Python backend for a fixed --seed.
- BF16 outputs materialise through OpenVINO Python as a float16 *value* view
  whose raw bits equal the true bf16 bits (SDK 2025.4 behaviour). Byte- and
  bit-level consumers (md5, gather-guard) therefore compare the bit patterns
  directly, identical in meaning to the native raw-byte handling.
- The FP16 DINOv3 ViT SDPA->BF16 graph rewrite is native-only; the Python
  backend reports 0 rewritten nodes (so un-rewritten FP16 attention may show
  lower cosine similarity on those models).

Selecting the backend:
    ./ov_bench --backend python --mode <mode> ...   # Python backend (this package)
    ./ov_bench --backend cpp    --mode <mode> ...   # native C++ binary (if built)
Environment overrides: OVBENCH_BACKEND (cpp|python), OVBENCH_PYTHON
(interpreter with openvino+numpy), OVBENCH_CPP_BIN (path to the C++ binary).
"""

__version__ = "1.0.0"
