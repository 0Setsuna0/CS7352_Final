# Source Audit for SA-RnR-ToMe

This file records the external sources checked while implementing the local CogVideoX RnR acceleration path. The implementation uses these sources as references for method design and API compatibility; it does not copy the AsymRnR repository wholesale.

## AsymRnR paper

- URL: https://arxiv.org/abs/2412.11706
- URL: https://arxiv.org/html/2412.11706v2
- Checked for: the reduction/restoration framing, asymmetric treatment of Q and K/V, and the motivation that diffusion transformer redundancy can be exploited without training.
- Implementation takeaway: this project implements an AsymRnR-lite variant in `src/tokmerge/rnr/`, with Q reduction followed by output restoration, K/V reduction without restoration, feature-aware ratios, block/timestep gates, Euclidean matching, replace/mean reduction modes, and matching-cache reuse.

## AsymRnR code

- URL: https://github.com/wenhao728/AsymRnR
- CogVideoX script: https://github.com/wenhao728/AsymRnR/blob/main/scripts/cogvideox/inference.sh
- Checked for: CogVideoX experiment entry-point conventions and the fact that CogVideoX-2B is a supported target in the reference project.
- Implementation takeaway: the local code keeps the project-owned simplified runtime under `src/tokmerge/rnr/` and exposes a local CLI rather than vendoring or copying the external repository.

## Diffusers attention processor and CogVideoX source

- Docs: https://huggingface.co/docs/diffusers/en/api/attnprocessor
- Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
- Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/cogvideox_transformer_3d.py
- Checked for: attention processor API shape, `CogVideoXAttnProcessor2_0` QKV flow, RoPE placement, and text/video split.
- Local authority: because this repository vendors diffusers, the actual implementation follows `cog_diffuser/diffusers/src/diffusers/...` in this repo, not whichever API happens to be newest upstream.

## CogVideoX official sources

- Repository: https://github.com/zai-org/CogVideo
- Model card: https://huggingface.co/zai-org/CogVideoX-2b
- Existing project scripts currently default to `THUDM/CogVideoX-2b`; the new CLI keeps that default for compatibility while allowing `--model_path zai-org/CogVideoX-2b`.

## Existing local project documents

- `docs/AGENT_HANDOFF.md`: current handoff for future teammates/agents, including implementation status, run commands, successful results, and pitfalls.
- `docs/source_audit.md`: this external-source audit and remaining deviations from upstream AsymRnR.
- `docs/github_setup.md`: repository collaboration notes.

## Boundary of this implementation

- This is not TeaCache, DeepCache, TaylorCache, LCM, DMD, or pure sparse attention.
- It remains a training-free Token Merging/Token Reduction project.
- Report/result documents should only include numbers copied from script output or per-run JSON metadata. Do not invent benchmark or quality numbers.

## Comparison with upstream AsymRnR code

After inspecting the upstream repository, the closest files are:

- `arnr/match.py`: computes dot/cosine/euclidean/random matching. Euclidean uses the compiled `square_dist` extension, with `torch.cdist` as the conceptual fallback.
- `arnr/reduce.py`: reduces source tokens into destination tokens, where `replace` keeps destination tokens and unreduced source tokens while represented/reduced source tokens are restored from their matched destination.
- `arnr/operators.py`: caches matching outputs per feature (`h`, `q`, `v`) and exposes reduction/restoration functions for hidden input, Q output, and K/V output.
- `arnr/cogvideox/attention.py`: applies hidden-state reduction before QKV when configured, applies RoPE, reduces Q and K/V inside attention, restores Q output, then optionally restores hidden-state output.
- `arnr/cogvideox/apply.py`: loads a safetensors redundancy schedule and YAML ratio thresholds to decide feature/block/timestep reduction.

The first local implementation was intentionally an AsymRnR-lite adaptation rather than an exact clone:

- It implements Q reduction/restoration and K/V reduction, which is the core plan requirement.
- It does not load the upstream safetensors redundancy schedule; it uses fixed CLI/config ratios plus block/timestep gates.
- It uses one shared matching plan across attention heads by flattening heads into the feature dimension; upstream can match per head.
- It uses deterministic strided destinations; upstream randomly selects one destination token per spatiotemporal chunk.
- It uses a pure-torch matrix identity for squared Euclidean distance instead of the upstream compiled `square_dist` extension. This preserves nearest-neighbor ranking for full features while avoiding a hard extension dependency.
- For memory safety on 480x720 CogVideoX-2B, matching subsamples high-dimensional features to about 128 dimensions. This may change exact matched pairs compared with upstream full-dimensional matching, but it prevents the pairwise matching step from allocating hundreds of GiB on 12 GiB GPUs.

## Follow-up alignment implemented on 2026-06-13

The local RnR path was updated to align more closely with the official CogVideoX AsymRnR code:

- Added `schedule_file` / `schedule_url` support and safetensors loading for official redundancy schedules.
- Added official-style threshold-to-ratio schedules, e.g. CogVideoX-2B base `q: {6: 0.4, 7: 0.8}` and `v: {8: 0.3}`.
- Added official-style random spatiotemporal chunk destination selection via `partition_mode: "random_chunk"`.
- Changed token-count calculation to use the official `num_unreduce_src` formulation.
- Added optional hidden-state (`h`) reduction/restoration plumbing, disabled by default for CogVideoX-2B quality configs because the official CogVideoX config only enables `q` and `v`.
- Added `configs/rnr/rnr_official_base.yaml` and `configs/rnr/rnr_official_fast.yaml`.
- Verified an actual tiny CogVideoX run with `configs/rnr/rnr_official_base.yaml`; metadata showed `schedule_loaded=true` and dynamic average reduction ratios rather than fixed global ratios.

## Deprecated local routes

Earlier local experiments produced useful negative results:

- Full-block hidden-state TokenMerge is useful as a `naive_tome` baseline, but not as the final method because it creates blur, grid texture, structure distortion, flicker, and motion instability.
- Fixed-ratio RnR was faster on simple prompts, but less stable on fine details and complex scenes than the official redundancy schedule.
- Deterministic strided destinations were replaced by official-style random spatiotemporal chunk destinations.
- Broadcast Euclidean distance was removed after causing extreme memory allocation; the local implementation now uses a matrix-distance formulation.
- Generic negative prompts were removed from quality scripts because they interfered with subject preservation and valid static-object prompts.
- `prop_attn=true` remains off by default because it can disable faster SDPA kernels in this environment.

Remaining known deviations from upstream:

- The local matcher still uses pure PyTorch with feature subsampling (`_MATCH_DIM = 128`) instead of the official compiled `square_dist` extension.
- The local implementation maps arbitrary inference step counts onto the 50-step official CogVideoX schedule using denoising progress. This makes 40-step local quality runs usable, but it is not identical to running the original 50-step schedule.
- Per-head matching remains flattened across heads for memory stability.
