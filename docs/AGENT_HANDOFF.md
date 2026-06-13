# Agent Handoff: CogVideoX RnR Acceleration

Last updated: 2026-06-13

This is the main handoff document for future teammates or coding agents. The current useful path is the official-schedule RnR implementation, not the older full-block TokenMerge path.

## Current Status

The repository now supports three practical modes:

- `none`: original CogVideoX-2B baseline.
- `naive_tome`: inherited hidden-state/block TokenMerge baseline. Keep it as a failure/ablation baseline.
- `rnr_tome`: current recommended method. It reduces visual Q and V/K inside attention, restores Q output, leaves text tokens untouched, and uses the official AsymRnR redundancy schedule.

The recommended high-quality setting is:

```python
ENABLE_TOKEN_MERGE = True
RNR_CONFIG_INDEX = 6  # official_base
```

in [`scripts/run_quality_rnr_test.py`](../scripts/run_quality_rnr_test.py).

## Chosen Approach

The current chosen approach is **official-schedule Asymmetric Reduction-and-Restoration (AsymRnR) for CogVideoX-2B**.

In plain terms:

1. Keep CogVideoX text tokens full length.
2. Only reduce visual tokens inside attention.
3. Reduce visual Q tokens, run attention on the shorter Q sequence, then restore Q attention output back to the original visual-token length.
4. Reduce visual K/V tokens without restoring them, because K/V only define the attention memory.
5. Use the official AsymRnR redundancy schedule to decide which block/timestep/feature should be reduced and by how much.

This replaced the earlier fixed-ratio RnR and full-block TokenMerge routes because those were faster on some prompts but much less stable visually.

Implementation flow:

```text
run_quality_rnr_test.py
  -> loads configs/rnr/rnr_official_base.yaml
  -> apply_rnr_to_cogvideox(pipe.transformer, cfg)
  -> CogVideoXTransformer3DModel.forward observes current latent video layout and timestep
  -> each CogVideoXBlock sends block index + runtime to CogVideoXRnRAttnProcessor2_0
  -> runtime reads official safetensors redundancy tier for feature/block/timestep
  -> scheduler maps tier to ratio using q/v threshold tables
  -> partition selects destination/source visual tokens with random spatiotemporal chunks
  -> matching pairs source tokens to destination tokens
  -> processor reduces Q and K/V visual tokens
  -> scaled_dot_product_attention runs on reduced visual sequences plus full text tokens
  -> Q output is restored to original visual length
  -> block output shape stays identical to baseline
```

The important implementation files are:

- [`configs/rnr/rnr_official_base.yaml`](../configs/rnr/rnr_official_base.yaml): recommended config and official schedule thresholds.
- [`src/tokmerge/rnr/runtime.py`](../src/tokmerge/rnr/runtime.py): loads the official safetensors schedule and records current timestep/layout.
- [`src/tokmerge/rnr/scheduler.py`](../src/tokmerge/rnr/scheduler.py): maps redundancy tiers to reduction ratios.
- [`src/tokmerge/rnr/partition.py`](../src/tokmerge/rnr/partition.py): implements official-style random spatiotemporal chunk partitioning.
- [`src/tokmerge/rnr/reduce_restore.py`](../src/tokmerge/rnr/reduce_restore.py): builds reduction plans, reduces tokens, restores Q output.
- [`src/tokmerge/rnr/cogvideox_processor.py`](../src/tokmerge/rnr/cogvideox_processor.py): applies the RnR attention path.
- [`cogvideox_transformer_3d.py`](../cog_diffuser/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py): passes layout/timestep/block context into RnR.

## What Was Implemented

Main RnR package:

- [`src/tokmerge/rnr/rnr_config.py`](../src/tokmerge/rnr/rnr_config.py): config loading, including `schedule_file`, `schedule_url`, `partition_mode`, and ratio schedules.
- [`src/tokmerge/rnr/runtime.py`](../src/tokmerge/rnr/runtime.py): runtime state, safetensors schedule loading, matching cache, statistics.
- [`src/tokmerge/rnr/scheduler.py`](../src/tokmerge/rnr/scheduler.py): official threshold-to-ratio logic.
- [`src/tokmerge/rnr/partition.py`](../src/tokmerge/rnr/partition.py): official-style random spatiotemporal chunk destination selection.
- [`src/tokmerge/rnr/matching.py`](../src/tokmerge/rnr/matching.py): pure PyTorch matching with memory-safe Euclidean distance.
- [`src/tokmerge/rnr/reduce_restore.py`](../src/tokmerge/rnr/reduce_restore.py): reduction/restoration plan and token restore.
- [`src/tokmerge/rnr/cogvideox_processor.py`](../src/tokmerge/rnr/cogvideox_processor.py): CogVideoX attention processor with Q restore and K/V reduction.
- [`src/tokmerge/rnr/apply.py`](../src/tokmerge/rnr/apply.py): attach/detach helpers.

Vendored diffusers integration:

- [`cogvideox_transformer_3d.py`](../cog_diffuser/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py) attaches `_rnr_runtime` and `_rnr_block_index` to CogVideoX blocks and forwards layout information to the RnR runtime.

Scripts:

- [`scripts/run_quality_rnr_test.py`](../scripts/run_quality_rnr_test.py): easiest quality script. Toggle baseline/RnR and choose prompt/config by index.
- [`scripts/run_cogvideox_accel.py`](../scripts/run_cogvideox_accel.py): unified CLI for baseline, naive ToMe, and RnR.
- [`scripts/run_quality_tokenmerge_test.py`](../scripts/run_quality_tokenmerge_test.py): old TokenMerge quality toggle.
- [`scripts/evaluate_quality.py`](../scripts/evaluate_quality.py): lightweight pairwise video metrics and review HTML.

## Configs

RnR configs live in [`configs/rnr`](../configs/rnr).

Quality-script index mapping:

| Index | Name | Use |
|---:|---|---|
| 0 | `quality_safe` | Minimal compression, quality debugging |
| 1 | `conservative` | Fixed-ratio quality fallback |
| 2 | `balanced` | Fixed-ratio speed/quality fallback |
| 3 | `current_default` | Historical fixed-ratio config |
| 4 | `aggressive` | Speed test, higher quality risk |
| 5 | `max_speed` | Stress/speed only |
| 6 | `official_base` | Recommended official-style config |
| 7 | `official_fast` | Faster official-style config, more quality risk |

`official_base` uses:

- `schedule_file`: `configs/rnr/schedulers/euclidean_cogvideox-2b_cache5.safetensors`
- `schedule_url`: official AsymRnR GitHub raw URL, auto-downloaded if missing
- `schedule`: `q: {6: 0.4, 7: 0.8}`, `v: {8: 0.3}`
- `partition_mode`: `random_chunk`
- `matching_cache_steps`: `5`
- `prop_attn`: `false`

Important: the printed `q_ratio=0.4, kv_ratio=0.2` are config candidates/fallbacks. With official schedules, the actual average reduction is dynamic and appears in metadata as `q_reduction_ratio` and `kv_reduction_ratio`.

## Prompts

Both quality scripts now share prompt indices:

| Index | Name |
|---:|---|
| 0 | `panda_studio_guitar` |
| 1 | `robot_wave_studio` |
| 2 | `ceramic_cup_steam` |
| 3 | `red_ball_roll` |
| 4 | `fox_turntable` |
| 5 | `paper_airplane` |
| 6 | `rainy_cafe_window` |
| 7 | `lantern_boat_river` |
| 8 | `bookstore_cat_walk` |
| 9 | `plaza_street_musician` |
| 10 | `greenhouse_butterfly` |

The negative-prompt path was removed from the quality scripts because it interfered with simple-scene prompts and could erase the subject.

## How To Run

Recommended baseline/RnR quality pair:

```powershell
# In scripts/run_quality_rnr_test.py:
# ENABLE_TOKEN_MERGE = False
# PROMPT_INDEX = 7
# RNR_CONFIG_INDEX = 6
.\.venv\Scripts\python.exe .\scripts\run_quality_rnr_test.py

# Then switch only:
# ENABLE_TOKEN_MERGE = True
.\.venv\Scripts\python.exe .\scripts\run_quality_rnr_test.py
```

Recommended CLI smoke:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_cogvideox_accel.py `
  --accel rnr_tome `
  --schedule_config configs/rnr/rnr_official_base.yaml `
  --prompt "A white ceramic cup on a plain table, steam rises slowly, clean studio background." `
  --num_inference_steps 2 `
  --num_frames 9 `
  --height 256 `
  --width 384 `
  --guidance_scale 1.0 `
  --enable_cpu_offload `
  --save_video `
  --output_dir results/smoke_official_rnr `
  --benchmark_csv results/smoke_official_rnr/benchmark.csv `
  --local_files_only `
  --log_latency `
  --log_memory
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Verified Results

Validation:

- `pytest tests -q`: `34 passed`.
- `py_compile` passed for RnR modules and quality/CLI scripts.
- Tiny real CogVideoX official-base smoke passed with `schedule_loaded=true`.

High-quality result: `lantern_boat_river`, 49 frames, 720x480, 40 steps, seed 123, model CPU offload, `THUDM/CogVideoX-2b`.

| Method | E2E | Transformer | Avg Step | Speedup |
|---|---:|---:|---:|---:|
| baseline | 289.345 s | 264.8315 s | 6.4737 s | 1.00x |
| official_base RnR | 253.842 s | 229.5358 s | 5.6059 s | 1.14x E2E / 1.15x transformer |

Official-base RnR dynamic reduction stats for that run:

- `q_reduction_ratio`: `0.1756764957`
- `kv_reduction_ratio`: `0.08975`
- `rnr_cache_hit_rate`: `0.7790868925`

Historical fixed-ratio RnR result, simpler prompt:

- baseline: `303.825 s` E2E, `271.7829 s` transformer
- fixed-ratio RnR: `248.946 s` E2E, `226.8976 s` transformer
- speedup: `1.22x` E2E / `1.20x` transformer

The official-base run was visually better and more stable than earlier fixed-ratio configs in complex scenes.

## Failed Or Deprecated Routes

These routes were explored earlier and should not be treated as the main path anymore.

| Route | What was tried | What went wrong | Current status |
|---|---|---|---|
| Full-block hidden-state TokenMerge | Merge visual hidden states before attention/FFN, then scatter/restore after the block | Produced blur, grid texture, structure distortion, flicker, and motion instability. Higher ratios could speed up inference but damaged quality too much | Keep only as `naive_tome` failure/ablation baseline |
| Pre-attention restore bridge | Merge before attention, restore immediately after attention | Useful for debugging shape correctness, but it gives little real speedup because much of the block still runs close to full length | Deprecated as final method |
| Fixed-ratio RnR | Use constant `q_reduce_ratio=0.40`, `kv_reduce_ratio=0.20`, `matching_cache_steps=5` across active blocks/steps | Fast on simple prompts, but blurred fine details and destabilized steam, reflections, small objects, and complex backgrounds | Replaced by official schedule configs; keep as historical fallback |
| Deterministic strided destination tokens | Pick destination tokens from a fixed `(t,h,w)` lattice | Stable and simple, but can miss subject edges and creates sampling bias; worse than official random spatiotemporal chunk split | Replaced by `partition_mode="random_chunk"` |
| Broadcast Euclidean distance | Compute pairwise distance with `dst.unsqueeze(2) - src.unsqueeze(1)` | Tried to allocate hundreds of GiB at 720x480 CogVideoX token counts | Replaced by matrix identity `||a-b||^2 = ||a||^2 + ||b||^2 - 2ab` |
| Large negative prompt in quality scripts | Add generic negative prompt against blur, static subject, clutter, etc. | Interfered with simple prompt design; `static subject`/`no movement` could erase valid static objects such as cups | Removed completely from quality scripts |
| Over-simple prompts | Plain studio backgrounds with weak subject contrast | Some prompts collapsed into nearly uniform color/background, especially with white objects on gray walls | Added clearer and more dynamic prompts; use indices `6-10` for harder tests |
| Over-complex prompts | Dense forests, many tiny details, too much background motion | CogVideoX-2B and fixed-ratio merging both tended to smear details; hard to isolate acceleration artifacts | Avoid for primary benchmark; use moderate-complexity prompts instead |
| Sequential CPU offload for speed comparison | Run with sequential offload to fit memory | Transfer overhead dominates; token reduction savings become hard to interpret | Quality scripts use model offload; formal speed claims should note offload mode |
| `prop_attn=true` in attention | Add size log bias for merged K/V tokens | Can disable faster SDPA kernels and reduce speed; official CogVideoX configs here do not need it | Keep off by default |
| Hidden-state `h` reduction | Implement official-style optional hidden-state reduce/restore before QKV | It is structurally supported, but official CogVideoX-2B config only uses `q` and `v`; enabling `h` risks extra RoPE/restore instability | Disabled by default; experiment only |
| Fully exact upstream matching | Use official compiled `square_dist` and per-head full-dimensional matching | Better fidelity to upstream but not friendly to this Windows/12 GiB setup; extension dependency and memory cost are risky | Current code uses pure PyTorch and `_MATCH_DIM=128` feature subsampling |

## Pitfalls And Fixes

- Use this repository only: `C:\WorkSpace\AI\CS7352_Final`.
- Make sure `diffusers_path` printed by scripts is inside `cog_diffuser/diffusers/src/diffusers`.
- Full-resolution 49-frame CogVideoX-2B is tight on 12 GiB GPUs. Use `_OFFLOAD_MODE = "model"` in quality scripts.
- Old fixed-ratio RnR (`q=0.4`, `kv=0.2`, cache 5 everywhere) can blur or destabilize fine details. Prefer `RNR_CONFIG_INDEX = 6`.
- The official schedule file is 50-step. Local 40-step quality runs map denoising progress onto that schedule. This works, but it is not bit-identical to the official 50-step setup.
- The first Euclidean implementation used broadcast subtraction and tried to allocate hundreds of GiB. Current matching uses the matrix identity for squared distance.
- Matching still subsamples features to `_MATCH_DIM = 128` for memory stability. This is a known deviation from upstream.
- Per-head matching is not fully upstream-equivalent; heads are flattened into the feature dimension.
- `prop_attn=true` can disable faster SDPA kernels. Official CogVideoX configs here keep it off.
- Hidden-state `h` reduction plumbing exists but is disabled by default. Official CogVideoX-2B config only enables `q` and `v`.
- Do not reintroduce negative prompts in quality scripts unless testing that change explicitly.
- Outputs are intentionally named with prompt/config names to avoid overwriting prior runs.

## Docs Policy

The old development plans and corrupted historical summary were removed because they were either obsolete or misleading. Keep this handoff file current. Keep [`docs/source_audit.md`](source_audit.md) for external-source comparison and remaining deviations from upstream AsymRnR.

Do not invent benchmark numbers. Only record metrics copied from script output or per-run JSON metadata.
