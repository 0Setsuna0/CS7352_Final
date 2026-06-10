# Spatiotemporal Token Merging for CogVideoX-2b — Experiment Results

## Hardware
- **GPU**: NVIDIA GeForce RTX 4090 (24 GiB VRAM)
- **Offload mode**: None (full model on GPU)

## Baseline Configuration
- **Model**: THUDM/CogVideoX-2b (fp16)
- **Quality preset**: 480×720, 49 frames, 40 denoising steps, CFG 5.5
- **Video tokens per step**: 17,550 (13 latent frames × 30 × 45 grid)
- **Text tokens**: 226

## Key Results

### Performance (Quality Preset, 40 steps)

| Configuration | transformer_sec | avg_step (s) | Speedup |
|--------------|----------------|--------------|---------|
| Baseline (no merge) | 88.42 | 2.21 | 1.00× |
| Block + adaptive r=0.2 | 77.87 | 1.94 | **1.14×** |
| Block + fixed r=0.2 | 78.26 | 1.95 | **1.13×** |
| Block + fixed r=0.4 | 64.10 | 1.60 | **1.38×** |

### Impact of CPU Offload

| Offload Mode | Baseline (s) | Merge r=0.2 (s) | Speedup |
|-------------|-------------|-----------------|---------|
| Sequential CPU offload | 138.14 | 155.22 | 0.89× (slower!) |
| No offload (GPU-only) | 88.42 | 78.26 | **1.13×** |

**Finding**: Token merging only speeds up when computation is the bottleneck. With sequential CPU offload, the CPU↔GPU weight transfer dominates, and merge overhead actually slows things down.

## Token Merging Design

### Architecture
- **Matching**: Bipartite soft matching on L2-normalized hidden states
  - Content-adaptive (`hidden_norm`): cosine similarity on 128-dim subsample
  - Fixed positional (`fixed`): grid-distance nearest-neighbor, fully cached
- **Merge**: Size-weighted average (vectorized scatter_add)
- **Unmerge**: Scatter dst values back to absorbed positions
- **Scope**: `block` — merge before attention, run attention + FF on shorter sequence, unmerge after FF
- **RoPE handling**: `pre_rope` — gather RoPE embeddings at kept token positions
- **First-frame protection**: Frame-0 latent tokens never absorbed

### Active Layers
- Strategy: `middle_wide` — blocks 6–23 out of 30 (60% of blocks)
- Blocks 0–5 and 24–29 run without merging

### Checkerboard Partition
- Even (h+w) positions → src set (absorbable)
- Odd (h+w) positions → dst set (always kept)
- ~50/50 split; merge ratio controls how many src tokens are actually absorbed

## Optimizations Applied
1. **Partition caching**: Checkerboard pattern computed once, reused across all steps
2. **Index caching**: src/dst index arrays cached by grid shape + device
3. **Feature dim reduction**: Matching uses 128-dim subsample of 1920-dim features (15× bmm speedup)
4. **Vectorized merge/unmerge**: scatter_add + searchsorted, no Python loops
5. **Fixed matching mode**: Pre-computed grid-distance pairing, zero per-step matching cost
6. **Top-level imports**: Eliminated per-call import overhead

## Phase Progression

| Phase | Description | Status |
|-------|------------|--------|
| 0 | Baseline lock + CUDA event timing | ✅ |
| 1 | Pure-tensor merge library (25/25 tests) | ✅ |
| 2 | Config plumbing into vendored diffusers | ✅ |
| 3 | pre_attn_restore correctness bridge | ✅ |
| 4 | Attention on reduced sequence with RoPE | ✅ |
| 5 | scope=block (main deliverable) | ✅ |
| 6 | Spatiotemporal variant | ✅ |

## Files
- `src/tokmerge/merging.py` — Core merge/unmerge/matching library
- `src/tokmerge/runtime.py` — Config loading and transformer attachment
- `tests/test_merging.py` — 25 acceptance tests
- `configs/merge/` — JSON merge configurations
- `scripts/run_baseline_smoke_test.py` — Inference script with timing + merge support
- `cog_diffuser/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py` — Modified block forward
- `cog_diffuser/diffusers/src/diffusers/models/attention_processor.py` — Modified attention processor (RoPE + prop_attn)

## Output Videos
- `outputs/baseline_quality_nooffload.mp4` — Clean baseline (no merge)
- `outputs/merge_quality_r20_nooffload.mp4` — Fixed merge r=0.2 (1.13× speedup)
- `outputs/merge_quality_adaptive_r20.mp4` — Content-adaptive merge r=0.2 (1.14× speedup)
- `outputs/merge_quality_fixed_r40.mp4` — Fixed merge r=0.4 (1.38× speedup)
