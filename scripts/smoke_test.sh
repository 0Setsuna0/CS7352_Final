#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-THUDM/CogVideoX-2b}"
OUT_DIR="${OUT_DIR:-results/smoke}"
COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --prompt "A paper boat floats across a sunlit pond, soft ripples, realistic motion."
  --output_dir "$OUT_DIR"
  --seed 42
  --num_inference_steps 2
  --num_frames 9
  --height 256
  --width 384
  --guidance_scale 6.0
  --dtype fp16
  --enable_cpu_offload
  --log_latency
  --log_memory
  --save_video
  --benchmark_csv "$OUT_DIR/benchmark.csv"
)

"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel none
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel naive_tome --merge_ratio 0.1
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel kv_rnr --kv_reduce_ratio 0.1
"$PYTHON_BIN" scripts/run_cogvideox_accel.py "${COMMON_ARGS[@]}" --accel rnr_tome --q_reduce_ratio 0.2 --kv_reduce_ratio 0.1
