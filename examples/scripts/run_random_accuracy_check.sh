OV=/home/sgui/miniforge3/envs/ov
CPP=/home/sgui/2026/ov_bench
BIN=/tmp/ovbench-1003/ov-env-gather-20261106/build/ov_benchmark
OVLIB=$OV/lib/python3.11/site-packages/openvino/libs
MODELS=26Q2_round13_run2_e40_no_map_attn,ped_26Q1_history_data_gen4_800v2_issue2_e23_refine,vehicle_q2_release_v0_w_leading_25
R=/home/sgui/2026/reports/gather_fix_ov_accuracy_stability_rerun

export LD_LIBRARY_PATH="$OVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$R"

# 准备相同输入及 ONNX Runtime FP32 参考输出
"$OV/bin/python" "$CPP/tools/prepare_pred_models.py" \
  --model-root /home/sgui/2026/models/pred_model \
  --output "$R/prepared" --samples 10 --seed 43

# FP32/BF16 精度对照
"$BIN" --model-root /home/sgui/2026/models/pred_model \
  --manifest "$R/prepared/manifest.json" --prepared-inputs \
  --model "$MODELS" --precisions FP32,BF16 \
  --threads 6 --cpu-pinning NO --perf-count NO \
  --mode accuracy --output "$R/accuracy"

# 两个独立进程各跑 10 轮 BF16 稳定性测试
for process in a b; do
  "$BIN" --model-root /home/sgui/2026/models/pred_model \
    --manifest "$R/prepared/manifest.json" --prepared-inputs \
    --model "$MODELS" --precisions BF16 \
    --threads 6 --cpu-pinning NO --perf-count NO \
    --mode stability --rounds 10 --output "$R/stability_$process"
done

# 比较两个进程的 BF16 输出
"$BIN" --mode compare-stability \
  --stability-dir "$R/stability_a" \
  --stability-dir "$R/stability_b" \
  --output "$R/stability_comparison.json"
