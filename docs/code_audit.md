# Code Audit for dev_plan_v5

Date: 2026-06-12  
Repository: `C:\WorkSpace\AI\CS7352_Final`

## Commands Run

- `git status --short`
  - Output before implementation: `?? docs/dev_plan_v5.md`
- `Get-ChildItem -Depth 2`
  - Top-level directories: `.github`, `.venv`, `cog_diffuser`, `configs`, `docs`, `logs`, `outputs`, `report`, `scripts`, `src`, `tests`
- `scripts/check_env.py`
  - Python: 3.12.12
  - Platform: Windows-11-10.0.26200-SP0
  - Torch: 2.11.0+cu128
  - CUDA: available, CUDA 12.8
  - GPU: NVIDIA GeForce RTX 5070, 11.94 GiB
  - diffusers: 0.39.0.dev0
  - transformers: 4.57.6
  - accelerate: 1.13.0
  - huggingface_hub: 0.36.2
  - diffusers path: `cog_diffuser\diffusers\src\diffusers\__init__.py`
  - editable checkout active: True
- `rg -n "tome|merge|token|CogVideoX|AttnProcessor|attention|processor|hidden_states|rotary|RoPE|vae|benchmark|latency|memory" src scripts tests docs configs`

## Existing naive ToMe implementation summary

The existing implementation lives in:

- `src/tokmerge/merging.py`
- `src/tokmerge/runtime.py`
- `cog_diffuser/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py`
- `cog_diffuser/diffusers/src/diffusers/models/attention_processor.py`
- JSON configs under `configs/merge/`

It implements bipartite matching over the visual-token sequence, then merges and restores tokens with `RestoreInfo`. It supports `scope="block"`, `scope="attn_only"`, `scope="pre_attn_restore"`, and `scope="kv_only"`.

The historical baseline most relevant to the proposal is `scope="block"`: it merges visual hidden states before attention/FFN and scatters/unmerges them back to the original visual length after the block. Text tokens are split out before visual-token merging and are concatenated back for joint attention/FFN, so text tokens are not directly merged. The implementation supports merge ratio, spatial/spatiotemporal modes, first-frame protection, shifted checkerboard partition, CFG-consistent matching, per-layer strategies, and fixed-seed scripts.

Risk noted in existing docs: full-block hidden-state merging can create blur, grid artifacts, structure distortion, flicker, and motion instability because absorbed visual positions lose independent representation inside the block and are copied/restored later.

## Current CogVideoX integration summary

The runtime path uses vendored diffusers:

- `CogVideoXPipeline` from editable `cog_diffuser/diffusers`
- `CogVideoXTransformer3DModel`
- `CogVideoXBlock`
- `CogVideoXAttnProcessor2_0`

`CogVideoXTransformer3DModel.forward` receives latent video tensors shaped `[B, F, C, H, W]`, applies patch embedding, then splits text tokens first and visual tokens second. For CogVideoX-2B, default 49 output frames become 13 latent temporal tokens because `patch_size_t=4` in this vendored path: `(49 + 4 - 1) // 4 = 13`. Spatial latent token size is `height // patch_size` by `width // patch_size`; the current transformer config uses patch size metadata, not hard-coded `height // 16`.

Attention is MMDiT-style joint attention: text and visual tokens are concatenated before QKV projection, RoPE is applied to the visual query/key slice, SDPA is called, and outputs are split back into text and visual sequences.

## Existing CLI/script entry points

- `scripts/check_env.py`: environment and editable diffusers check
- `scripts/run_baseline_smoke_test.py`: baseline generation with optional old JSON merge config
- `scripts/run_quality_tokenmerge_test.py`: single high-quality generation toggle
- `scripts/run_all_tokenmerge_configs.py`: sweep over `configs/merge/*.json`
- `scripts/measure_noise_floor.py`: deterministic latent noise-floor check

The new unified entry point is `scripts/run_cogvideox_accel.py`.

## Risks in current implementation

- Full-block hidden-state merge remains a useful failure/baseline, but it is too aggressive for final quality.
- Old KV-only support can add proportional attention bias, which may disable efficient SDPA kernels.
- Old adaptive matching and restore are shape-correct but can still amplify CFG branch differences if CFG-consistent matching is disabled.
- 11.94 GiB GPU memory is tight for full 480x720/49-frame CogVideoX-2B; smoke scripts use smaller settings and/or CPU offload.
- Existing experiment results under `outputs/` are real historical outputs, but new SA-RnR-ToMe results are not run yet.

## Modification plan

- Keep existing `src/tokmerge/merging.py` and JSON configs as the `naive_tome` baseline.
- Add a new `src/tokmerge/rnr/` package for schedule-aware asymmetric attention-level reduction/restoration.
- Add a new CogVideoX RnR attention processor that reduces visual Q and/or K/V after QKV projection and RoPE, restores Q output to full length, and never reduces text tokens.
- Add a runtime matching cache that stores only matching indices/plans and resets per prompt.
- Add `scripts/run_cogvideox_accel.py` with `--accel none|naive_tome|kv_rnr|qv_rnr|rnr_tome`.
- Add benchmark, smoke, ablation, and quality-evaluation scripts under `scripts/`.
- Add README, report skeleton, source audit, and results templates without fabricated metrics.
