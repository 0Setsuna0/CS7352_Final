# Experiment Summary

Status: single-prompt quality comparison completed; full multi-prompt benchmark is still not run yet.

Smoke validation: `rnr_tome` passed a small real CogVideoX run on 2026-06-12 with 9 frames, 256x384, 2 denoising steps, seed 42, CPU offload, and local `THUDM/CogVideoX-2b` weights. This is only a code-path smoke result and is not a formal benchmark.

## Hardware

- GPU: NVIDIA GeForce RTX 5070, 11.94 GiB
- OS/platform: Windows-11-10.0.26200-SP0

## Software Versions

- Python: 3.12.12
- Torch: 2.11.0+cu128
- CUDA: 12.8
- diffusers: 0.39.0.dev0 from editable `cog_diffuser/diffusers`
- transformers: 4.57.6
- accelerate: 1.13.0

## Model

- Default: `THUDM/CogVideoX-2b`
- Alternative: `zai-org/CogVideoX-2b`

## Prompt Count

- `configs/prompts_benchmark.txt` currently contains 10 prompts across low-motion, high-motion, camera motion, fine detail, and multi-object categories.

## Methods

- none
- naive_tome ratios 0.1, 0.2, 0.3
- kv_rnr conservative
- qv_rnr conservative
- rnr_tome default
- rnr_tome fast

## Main Table

Single-prompt quality-script run, 49 frames, 480x720, 40 denoising steps, seed 123, model CPU offload:

| Method | E2E latency | Transformer latency | Avg step | Peak memory | Speedup vs baseline | Quality metric | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 303.825 s | 271.7829 s | 6.6465 s | 10.805 GiB | 1.00x | pending review | done |
| rnr_tome | 248.946 s | 226.8976 s | 5.5196 s | 10.805 GiB | 1.22x E2E / 1.20x transformer | pending review | done |

Full benchmark table over `configs/prompts_benchmark.txt`:

| Method | E2E latency | Avg step | Peak memory | Quality metric | Status |
|---|---:|---:|---:|---:|---|
| none | not run yet | not run yet | not run yet | not run yet | TODO |
| naive_tome | not run yet | not run yet | not run yet | not run yet | TODO |
| kv_rnr | not run yet | not run yet | not run yet | not run yet | TODO |
| qv_rnr | not run yet | not run yet | not run yet | not run yet | TODO |
| rnr_tome | not run yet | not run yet | not run yet | not run yet | TODO |

## Observed Artifacts

- TODO after qualitative review.

## Best Recommended Config

- Current single-prompt best: `rnr_tome` default from `configs/rnr_cogvideox2b_default.yaml`, pending visual review.
- Default RnR run used Euclidean matching, replace reduction, `dst_stride=(2,2,2)`, `matching_cache_steps=5`, Q reduction 0.4, KV reduction 0.2, cache hit rate 0.8.

## Failed Configs

- TODO after benchmark.

## Notes on Reproducibility

- Use fixed seed 42.
- Keep the benchmark CSV and per-run JSON metadata.
- Do not compare smoke-test numbers to formal 49-frame, 480x720 benchmark numbers.
