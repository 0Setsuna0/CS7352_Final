#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-THUDM/CogVideoX-2b}"
OUT_ROOT="${OUT_ROOT:-results/full_benchmark}"
PROMPTS="${PROMPTS:-configs/prompts_benchmark.txt}"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --prompt_file "$PROMPTS"
  --seed 42
  --num_inference_steps 50
  --num_frames 49
  --height 480
  --width 720
  --guidance_scale 6.0
  --dtype fp16
  --log_latency
  --log_memory
  --save_video
  --benchmark_csv "$OUT_ROOT/benchmark.csv"
)

"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel none --output_dir "$OUT_ROOT/none"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel naive_tome --merge_ratio 0.1 --output_dir "$OUT_ROOT/naive_tome_r01"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel naive_tome --merge_ratio 0.2 --output_dir "$OUT_ROOT/naive_tome_r02"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel naive_tome --merge_ratio 0.3 --output_dir "$OUT_ROOT/naive_tome_r03"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel kv_rnr --kv_reduce_ratio 0.1 --output_dir "$OUT_ROOT/kv_rnr_conservative"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel qv_rnr --q_reduce_ratio 0.2 --kv_reduce_ratio 0.1 --output_dir "$OUT_ROOT/qv_rnr_conservative"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --schedule_config configs/rnr_cogvideox2b_default.yaml --output_dir "$OUT_ROOT/rnr_tome_default"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --q_reduce_ratio 0.5 --kv_reduce_ratio 0.3 --matching_cache_steps 5 --output_dir "$OUT_ROOT/rnr_tome_fast"
