#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-THUDM/CogVideoX-2b}"
OUT_ROOT="${OUT_ROOT:-results/ablation}"
PROMPTS="${PROMPTS:-configs/prompts_benchmark.txt}"
MAX_PROMPTS="${MAX_PROMPTS:-4}"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --prompt_file "$PROMPTS"
  --max_prompts "$MAX_PROMPTS"
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

"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel naive_tome --merge_ratio 0.1 --output_dir "$OUT_ROOT/naive_r01"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel naive_tome --merge_ratio 0.2 --output_dir "$OUT_ROOT/naive_r02"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel naive_tome --merge_ratio 0.3 --output_dir "$OUT_ROOT/naive_r03"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --similarity_type cosine --output_dir "$OUT_ROOT/rnr_cosine"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --similarity_type euclidean --output_dir "$OUT_ROOT/rnr_euclidean"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --reduce_mode mean --output_dir "$OUT_ROOT/rnr_mean"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --reduce_mode replace --output_dir "$OUT_ROOT/rnr_replace"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --matching_cache_steps 1 --output_dir "$OUT_ROOT/rnr_cache1"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --matching_cache_steps 5 --output_dir "$OUT_ROOT/rnr_cache5"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel kv_rnr --kv_reduce_ratio 0.2 --output_dir "$OUT_ROOT/kv_only"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel qv_rnr --q_reduce_ratio 0.2 --kv_reduce_ratio 0.1 --output_dir "$OUT_ROOT/qv"
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --disable_schedule --output_dir "$OUT_ROOT/schedule_off"
