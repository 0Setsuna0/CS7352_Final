# Schedule-aware Asymmetric Token Merging for Efficient CogVideoX Inference

This repository is a CS7352 course project on training-free inference acceleration for video generation. It targets CogVideoX-2B and studies the quality-efficiency trade-off of spatial and spatiotemporal Token Merging.

The project keeps the inherited naive frame-wise hidden-state ToMe implementation as a baseline, then adds an AsymRnR-style attention-level method: visual Q and/or K/V tokens are reduced inside attention, Q outputs are restored to the original visual length, text tokens are never reduced, and block input/output shapes stay unchanged.

For the current implementation status, successful configs, known pitfalls, and agent handoff notes, start with [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md).

## Relation to the Proposal

The original proposal promised training-free spatial Token Merging for video generation using bipartite matching over visual tokens. Early code proved that hidden-state/full-block merging can speed up CogVideoX, but the generated videos showed blur, grid texture, flicker, and structure distortion. The current main method therefore moves from naive block-level ToMe to schedule-aware asymmetric reduction/restoration while staying within the Token Merging/Token Reduction theme.

## Methods

- `none`: original CogVideoX-2B baseline.
- `naive_tome`: inherited block-level visual hidden-state merge + scatter/restore, useful as a failure baseline.
- `kv_rnr`: reduce visual K/V only; Q stays full length, so output length is naturally preserved.
- `qv_rnr`: reduce visual Q and restore Q output; K/V are reduced conservatively.
- `rnr_tome`: schedule-aware asymmetric RnR with feature ratios, block/timestep gates, Euclidean matching, replace/mean reduction modes, and matching-cache reuse.

The new RnR implementation is under `src/tokmerge/rnr/`. Existing ToMe code remains under `src/tokmerge/`.

## Install

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
python .\scripts\check_env.py
```

`check_env.py` should report a `diffusers path` inside `cog_diffuser\diffusers`; the vendored diffusers checkout is installed editable.

## Model Weights

Scripts default to `THUDM/CogVideoX-2b` for compatibility with the existing project. You can switch to `zai-org/CogVideoX-2b`:

```powershell
python .\scripts\run_cogvideox_accel.py --model_path zai-org/CogVideoX-2b --accel none
```

Use `--local_files_only` after the model is cached.

## Smoke Test

Small smoke settings are only for code/shape validation, not formal results.

```powershell
python .\scripts\run_cogvideox_accel.py `
  --accel rnr_tome `
  --prompt "A paper boat floats across a sunlit pond, soft ripples, realistic motion." `
  --output_dir results/smoke `
  --seed 42 `
  --num_inference_steps 2 `
  --num_frames 9 `
  --height 256 `
  --width 384 `
  --enable_cpu_offload `
  --log_latency `
  --log_memory `
  --save_video `
  --benchmark_csv results/smoke/benchmark.csv
```

Bash helper:

```bash
bash scripts/smoke_test.sh
```

## Single-switch Quality Test

For a high-quality baseline/RnR pair, use:

```powershell
python .\scripts\run_quality_rnr_test.py
```

Edit these variables inside the script:

```python
ENABLE_TOKEN_MERGE = True   # True enables RnR, False runs baseline
PROMPT_INDEX = 7            # e.g. lantern_boat_river
RNR_CONFIG_INDEX = 6        # official_base, recommended
```

`RNR_CONFIG_INDEX = 7` selects `official_fast`, which is faster but riskier for quality.

## Benchmark

Formal benchmark prompts live in `configs/prompts_benchmark.txt`.

```powershell
python .\scripts\run_cogvideox_accel.py `
  --accel none `
  --model_path THUDM/CogVideoX-2b `
  --prompt_file configs/prompts_benchmark.txt `
  --output_dir results/baseline `
  --seed 42 `
  --num_inference_steps 50 `
  --num_frames 49 `
  --height 480 `
  --width 720 `
  --log_latency `
  --log_memory `
  --save_video `
  --benchmark_csv results/benchmark.csv
```

Run the final method:

```powershell
python .\scripts\run_cogvideox_accel.py `
  --accel rnr_tome `
  --model_path THUDM/CogVideoX-2b `
  --prompt_file configs/prompts_benchmark.txt `
  --output_dir results/rnr_tome_default `
  --seed 42 `
  --num_inference_steps 50 `
  --num_frames 49 `
  --height 480 `
  --width 720 `
  --schedule_config configs/rnr/rnr_official_base.yaml `
  --log_latency `
  --log_memory `
  --save_video `
  --benchmark_csv results/benchmark.csv
```

Bash helper for the full matrix:

```bash
bash scripts/benchmark_cogvideox_token_merge.sh
```

## Ablations

```bash
bash scripts/run_ablation.sh
```

The ablation script covers cosine vs Euclidean matching, mean vs replace reduction, cache reuse 1 vs 5, Q/KV variants, naive ToMe ratios, and schedule on/off.

## Quality Evaluation

Create a pair CSV with columns `baseline_video,candidate_video,prompt`, then run:

```powershell
python .\scripts\evaluate_quality.py `
  --pairs_csv results/pairs_to_review.csv `
  --output_dir results/quality
```

Outputs:

- `quality_metrics.csv`
- `pairwise_review.md`
- `qualitative_grid.html`

The lightweight evaluator computes frame MSE/PSNR and temporal difference proxies. VBench/FVD are not fabricated; add them only after running a proper evaluator.

## Code Structure

- `src/tokmerge/merging.py`: inherited pure-tensor ToMe merge/unmerge library.
- `src/tokmerge/runtime.py`: inherited JSON config attach/detach helpers.
- `src/tokmerge/rnr/`: new SA-RnR-ToMe implementation.
- `scripts/run_cogvideox_accel.py`: unified inference/benchmark CLI.
- `scripts/evaluate_quality.py`: lightweight quality comparison.
- `configs/merge/`: old naive/ToMA-like JSON configs.
- `configs/rnr/`: RnR presets, including official-base and official-fast schedules.
- `docs/AGENT_HANDOFF.md`: current handoff for teammates and future agents.
- `docs/source_audit.md`: external source audit and remaining deviations.
- `report/main.tex`: NeurIPS-style report skeleton.
- `results/`: reproducible result templates and generated benchmark outputs.

## Add Prompts

Edit `configs/prompts_benchmark.txt`. Blank lines and lines beginning with `#` are ignored.

## Modify Config

For RnR, prefer `configs/rnr/rnr_official_base.yaml` or `configs/rnr/rnr_official_fast.yaml`. Fixed-ratio fallback configs are also under `configs/rnr/`.

```powershell
python .\scripts\run_cogvideox_accel.py --accel rnr_tome --schedule_config configs/rnr/rnr_official_base.yaml
```

For inherited naive ToMe, old JSON configs remain in `configs/merge/`.

## Known Issues

- Full-resolution CogVideoX-2B can exceed 12 GiB VRAM. Use CPU offload for smoke tests or a larger GPU for formal benchmarks.
- Naive full-block ToMe can create blur, pixelation, flicker, structure distortion, and motion instability.
- Prefer official-schedule RnR over fixed-ratio RnR for complex scenes.
- Proportional attention bias can disable faster SDPA kernels; the default RnR config keeps `prop_attn=false`.

## Results Policy

Do not write unrun numbers into the report. Current verified numbers are summarized in `docs/AGENT_HANDOFF.md`; future results should be copied from script output or per-run JSON metadata.

## References

This project references AsymRnR, Diffusers, CogVideoX, ToMe, and related Token Merging work. See `docs/source_audit.md` and `report/references.bib`.

The project-owned implementation is the code under `src/tokmerge/`, the vendored CogVideoX integration edits under `cog_diffuser/diffusers`, and the scripts/configs/docs in this repository. AsymRnR/Diffusers/CogVideoX were used as references, not copied wholesale.
