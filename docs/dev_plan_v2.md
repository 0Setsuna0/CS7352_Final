# Spatiotemporal Token Merging for CogVideoX-2B — Development Plan (v2)

> Target repo: `F:\git\hw\CS7352_Final`
> Baseline: `THUDM/CogVideoX-2b` via vendored `cog_diffuser/diffusers` (editable install)
> Team: 3 graduate students. Course: CS7352.
>
> This document is the single source of truth for implementation. Every Phase has
> concrete file paths, interfaces, commands, and acceptance criteria. An agent
> should be able to execute Phase by Phase without further design decisions.

## 0. Scope, Claims, and Honesty Contract

This project implements **training-free spatiotemporal token merging** for a video
Diffusion Transformer (CogVideoX-2B) and characterizes the **quality vs. efficiency
Pareto front**.

Three things this plan fixes relative to the proposal / old plan, and they are mandatory:

1. **Real speedup, not fake speedup.** Token merging MUST reduce the sequence length
   that self-attention actually computes on. A "merge then immediately restore before
   attention" design is forbidden as the final deliverable because it does not reduce
   attention FLOPs (this was Risk 1 in the old plan). Merge happens before attention on
   the reduced sequence; unmerge happens after attention.
2. **Spatiotemporal, matching the proposal title.** We deliver BOTH a spatial-only
   variant AND a spatiotemporal variant, and we quantify the difference (especially
   temporal flicker). The proposal says "spatiotemporal"; the deliverable must contain a
   temporal axis.
3. **Honest hardware reporting.** The repo README targets a 12 GB GPU with sequential
   CPU offload; the proposal claims RTX 4090 numbers. CPU offload can mask GPU compute
   speedup. We MUST report a `transformer-only` timing in addition to end-to-end, and
   clearly state which offload mode each number was measured under.

Out of scope (do NOT do): retraining, distillation (DMD/LCM), VAE changes, text-encoder
changes, writing a video model from scratch.

## 1. Repository Facts (already true, do not rebuild)

- Entry script: `scripts/run_baseline_smoke_test.py`. Presets `smoke` (256x384, 9 frames,
  2 steps) and `quality` (480x720, 49 frames, 40 steps). It already logs `load_seconds`,
  `inference_seconds`, `peak_gpu_memory_gib` and writes a sibling `.json`.
- Scheduler for 2b: `CogVideoXDDIMScheduler`, `use_dynamic_cfg=False`.
- Frame constraint for 2b: `num_frames` must be `8N+1` and `<= 49`. `height`/`width`
  divisible by 8.
- Editable diffusers source tree to edit:
  - `cog_diffuser/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py`
    (`CogVideoXTransformer3DModel`, `CogVideoXBlock`)
  - `cog_diffuser/diffusers/src/diffusers/models/attention_processor.py`
    (`CogVideoXAttnProcessor2_0`)
  - `cog_diffuser/diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py`
- Env verify: `.\.venv\Scripts\python.exe .\scripts\check_env.py` must show
  `diffusers editable checkout active: True`.

## 2. Architecture Decision (read before coding)

CogVideoX uses an MMDiT-style block: text tokens and video tokens are **concatenated**
before QKV, attention is applied jointly (with RoPE applied to the video portion), then
split back. This is why naive token merging is harder than in plain ViT.

Decision: we implement merging **inside the attention computation**, operating only on
the video-token portion, in three placements selectable by config:

- `placement = "pre_attn_restore"` — DEBUG / CORRECTNESS ONLY. Merge before attn, restore
  before attn output is added. Used to validate matching/restore math against baseline.
  Not a speed deliverable.
- `placement = "attn_seq_reduce"` — PRIMARY DELIVERABLE. The video tokens fed into
  scaled-dot-product-attention are the merged (shorter) set; text tokens stay full; RoPE
  is applied to merged video positions; output video tokens are unmerged after attention.
  This actually shortens the attention sequence. This is the variant that must produce
  the speed numbers.

We support two merge metrics:

- `mode = "spatial"` — bipartite soft matching within each frame (ToMe-style). The
  conservative variant.
- `mode = "spatiotemporal"` — matching across a local temporal window (frame `t` with
  `t-1..t+1`) so temporally redundant tokens merge; this is the proposal's headline and
  the flicker-sensitive variant.

## 3. Module Design

### 3.1 New file: `src/tokmerge/merging.py`

Pure-tensor, framework-light, unit-testable. No diffusers import.

```python
# src/tokmerge/merging.py
from dataclasses import dataclass
import torch

@dataclass
class MergeConfig:
    enabled: bool = False
    ratio: float = 0.0                 # fraction of video tokens removed per merge call
    mode: str = "spatial"              # "spatial" | "spatiotemporal"
    metric: str = "cosine"             # only "cosine" in v1
    placement: str = "attn_seq_reduce" # "attn_seq_reduce" | "pre_attn_restore"
    layers: tuple[int, ...] = ()       # transformer block indices where merging is active
    temporal_window: int = 1           # used only for spatiotemporal; t-w..t+w
    protect_first_frame: bool = True   # never merge tokens of frame 0 (anchors structure)

@dataclass
class RestoreInfo:
    # everything needed to scatter merged tokens back to full layout
    src_idx: torch.Tensor   # which tokens were absorbed
    dst_idx: torch.Tensor   # into which kept tokens
    keep_idx: torch.Tensor  # indices of kept (unmerged-position) tokens
    sizes: torch.Tensor     # weights per merged token
    num_video_tokens: int
    grid: tuple[int, int, int]  # (frames, gh, gw)

def bipartite_soft_match(metric: torch.Tensor, r: int,
                         frames: int, gh: int, gw: int,
                         mode: str, temporal_window: int,
                         protect_first_frame: bool):
    """Return (src_idx, dst_idx, keep_idx) for r merges.
    metric: [B, N, C] (L2-normalized upstream)."""
    ...

def merge_tokens(x: torch.Tensor, src_idx, dst_idx, keep_idx, sizes):
    """Weighted-average merge. Returns (merged_x [B, N-r, C], new_sizes)."""
    ...

def unmerge_tokens(merged_x: torch.Tensor, info: RestoreInfo) -> torch.Tensor:
    """Scatter back to [B, num_video_tokens, C].
    Absorbed tokens copy their destination value."""
    ...
```

Hard requirements:

- `ratio = 0` => identity (merged == input, restore == input) exactly (`allclose`).
- Works for arbitrary `B, frames, gh, gw, C`.
- Operates on the video portion only; caller is responsible for splitting text/video.
- Deterministic given a fixed input.

### 3.2 New file: `src/tokmerge/runtime.py`

Glue between config and the diffusers model. No heavy logic.

```python
# src/tokmerge/runtime.py
def attach_merge_config(transformer, cfg: "MergeConfig") -> None:
    """Set transformer._merge_cfg and per-block block_index; idempotent;
    safe to call each run."""

def detach_merge_config(transformer) -> None:
    """Remove merge config -> exact baseline behavior."""

def load_merge_config(path: str) -> "MergeConfig":
    """Load a JSON config from configs/."""
```

### 3.3 Edits to vendored diffusers (minimal, guarded)

`cogvideox_transformer_3d.py`:

- `CogVideoXTransformer3DModel.__init__`: add `self._merge_cfg = None`.
- `CogVideoXTransformer3DModel.forward`: compute and pass `(frames, gh, gw)` grid metadata
  down to blocks (derivable from latent shape + `patch_size`). Assign `block_index` to
  each block once.
- `CogVideoXBlock.forward`: if `self._merge_cfg` enabled AND `block_index in cfg.layers`,
  route attention through the merge path; otherwise unchanged.

`attention_processor.py` (`CogVideoXAttnProcessor2_0`):

- Only touched for `placement == "attn_seq_reduce"`. Accept optional merge directives via
  `attention_kwargs`. When present: split text/video, merge video Q/K/V positions
  consistently, apply RoPE to merged video positions, run SDPA on the reduced sequence,
  then unmerge the video output. When absent: behave exactly as upstream.

Guard rule: every edit is wrapped so that with no merge config the code path is
byte-for-byte the original behavior. Verified in Phase 1 / Phase 2 acceptance.

## 4. Config Surface

New files under `configs/`:

- `configs/merge/spatial_r10.json`, `..._r20.json`, `..._r30.json`
- `configs/merge/st_r20.json` (spatiotemporal)
- `configs/merge/layers_middle.json`, `layers_middle_wide.json`, `layers_late_off.json`
- `configs/prompts_eval.json` (10 prompts, fixed; include 1 high-motion, 1 low-motion,
  1 multi-object, 1 camera-move, etc.)
- `configs/experiment_matrix.json` (drives the matrix runner in Phase 6)

Example `configs/merge/spatial_r20.json`:

```json
{
  "enabled": true,
  "ratio": 0.2,
  "mode": "spatial",
  "metric": "cosine",
  "placement": "attn_seq_reduce",
  "layers": "middle_wide",
  "temporal_window": 1,
  "protect_first_frame": true
}
```

`layers` may be a named strategy resolved at runtime against `num_layers`
(CogVideoX-2b has 30 transformer blocks):

- `middle`: middle 1/3 of blocks
- `middle_wide`: skip first 20% and last 20%
- `late_off`: enable early + middle, disable last ~20% (protect fine detail)

## 5. CLI Integration

Extend `scripts/run_baseline_smoke_test.py` (additive, defaults preserve current
behavior):

- `--merge-config PATH` (default `None` => baseline, no merging)
- `--report PATH` (where to append a benchmark row; default keeps existing sibling JSON
  behavior)
- `--transformer-timing` flag: wrap the transformer forward with timing hooks and log
  `transformer_seconds` and `avg_step_seconds` separately from `inference_seconds`.

Metadata JSON gains: `merge_enabled`, `merge_ratio`, `merge_mode`, `merge_placement`,
`merge_layers_resolved`, `transformer_seconds`, `avg_step_seconds`, `offload_mode`.

## 6. Evaluation Pipeline

New file: `src/eval/run_matrix.py` — runs the full experiment matrix and writes one tidy
CSV.

New file: `src/eval/metrics.py`:

- `clip_score(video_frames, prompt)` — semantic alignment (per-frame CLIP, averaged).
- `temporal_consistency(video_frames)` — mean CLIP cosine similarity between adjacent
  frames (flicker proxy; cheap, no dataset needed).
- `warp_error(video_frames)` — optional optical-flow warp error (RAFT) as a stronger
  flicker metric; gated behind a flag because it adds deps.
- `fvd(...)` — OPTIONAL, behind `--with-fvd`; requires a reference clip set. If skipped,
  report says "FVD not run; used CLIP + temporal-consistency + VBench + human eval
  instead" honestly.

VBench: integrate the subset that does not need a private dataset — at minimum
`subject_consistency`, `background_consistency`, `motion_smoothness`,
`temporal_flickering`, `imaging_quality`. Document the exact VBench dimensions used.
(VBench is the proposal-grade benchmark; do this, don't hand-wave it.)

## 7. Experiment Matrix (the actual research output)

```text
models:   CogVideoX-2b
presets:  quality (480x720, 49 frames, 40 steps)   # smoke only for plumbing
mode:     baseline, spatial, spatiotemporal
ratio:    0.0(baseline), 0.10, 0.20, 0.30
layers:   middle_wide (primary), late_off (ablation)
prompts:  10 (configs/prompts_eval.json)
seeds:    2 fixed seeds
offload:  report under BOTH "sequential" (12GB reality) and, if a >=24GB GPU is
          available, "model"/"none" (clean GPU speedup)
```

Primary plots / tables:

1. Pareto front: x = transformer_seconds (and end-to-end), y = quality
   (VBench composite / CLIP / temporal-consistency). One curve per mode.
2. Peak VRAM vs ratio.
3. Spatial vs spatiotemporal at equal ratio: `temporal_flickering` and `motion_smoothness`
   side-by-side (this is the proposal's core claim).
4. Layer-strategy ablation (`middle_wide` vs `late_off`) at ratio 0.2.
5. Qualitative grid: baseline vs r0.2 spatial vs r0.2 spatiotemporal for 3 prompts, with
   zoomed crops on artifact regions.

## 8. Phased Execution (agent-runnable)

Each phase ends with a concrete command and a pass/fail check. Do not advance until the
check passes.

### Phase 0 — Baseline lock + instrumentation

Tasks:

- Add `--transformer-timing`, `transformer_seconds`, `avg_step_seconds`, `offload_mode`
  to the smoke script + metadata.
- Run smoke and quality once; archive metadata as the baseline reference.

Commands:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_baseline_smoke_test.py --preset smoke --local-files-only --transformer-timing
.\.venv\Scripts\python.exe .\scripts\run_baseline_smoke_test.py --preset quality --local-files-only --transformer-timing
```

Pass: both produce video + JSON; JSON contains non-null `transformer_seconds` and
`avg_step_seconds`.

### Phase 1 — Core merger + tests (no model needed)

Tasks: implement `src/tokmerge/merging.py`; add `tests/test_merging.py`.

Tests required:

- ratio 0 => exact identity (merge and unmerge).
- output shapes for varied `B, frames, gh, gw, C`.
- weighted-average correctness on a toy tensor (hand-computed expected).
- `protect_first_frame`: frame-0 tokens never appear in `src_idx`.
- spatiotemporal mode only matches within `temporal_window`.

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_merging.py -q
```

Pass: all tests green.

### Phase 2 — Wire config plumbing (no behavior change yet)

Tasks: `src/tokmerge/runtime.py`; add `self._merge_cfg=None`, `block_index`, grid metadata
in the transformer; `attach/detach_merge_config`.

Pass: with NO merge config attached, quality output is bit-identical (same seed => same
frames hash) to Phase 0 baseline.

```powershell
.\.venv\Scripts\python.exe .\scripts\run_baseline_smoke_test.py --preset smoke --local-files-only
# compare frame hash / metadata vs Phase 0 baseline -> must match
```

### Phase 3 — `pre_attn_restore` path (correctness bridge)

Tasks: implement merge -> attn -> restore inside `CogVideoXBlock` for debug placement.

Pass: at ratio 0, identical to baseline; at ratio 0.1 the smoke preset runs without
crashing and shapes survive through unpatchify / decoder.

```powershell
.\.venv\Scripts\python.exe .\scripts\run_baseline_smoke_test.py --preset smoke --local-files-only --merge-config .\configs\merge\spatial_r10.json
```

### Phase 4 — `attn_seq_reduce` path (REAL speedup)

Tasks: implement video-only merge inside `CogVideoXAttnProcessor2_0` with correct
text/video split, RoPE on merged positions, SDPA on reduced sequence, unmerge after
attention.

Pass:

- correctness: ratio 0 == baseline.
- speed: with `--transformer-timing`, `transformer_seconds` at ratio 0.2 (middle_wide) is
  measurably lower than baseline on a non-offload or model-offload run. Record the
  percentage.

```powershell
.\.venv\Scripts\python.exe .\scripts\run_baseline_smoke_test.py --preset quality --local-files-only --merge-config .\configs\merge\spatial_r20.json --transformer-timing
```

### Phase 5 — Spatiotemporal variant

Tasks: enable `mode="spatiotemporal"` end-to-end through the attn path.

Pass: runs at ratio 0.2; produces a video; temporal-consistency metric computable.

```powershell
.\.venv\Scripts\python.exe .\scripts\run_baseline_smoke_test.py --preset quality --local-files-only --merge-config .\configs\merge\st_r20.json --transformer-timing
```

### Phase 6 — Metrics + matrix runner

Tasks: `src/eval/metrics.py`, `src/eval/run_matrix.py`, VBench subset wiring.

Pass: `run_matrix.py` produces `report/results.csv` with one row per
(mode, ratio, layers, prompt, seed) and all metric columns populated (FVD optional).

```powershell
.\.venv\Scripts\python.exe .\src\eval\run_matrix.py --config .\configs\experiment_matrix.json --out .\report\results.csv
```

### Phase 7 — Analysis, plots, report, NeurIPS checklist

Tasks: generate the 5 plots/tables from section 7; write `report/` draft; fill NeurIPS
checklist; record demo video; write honest limitations (offload caveat, no-distillation,
ratio ceiling).

Pass: report contains Pareto front, spatial-vs-spatiotemporal flicker comparison, VRAM
curve, qualitative grid, and a "relation to prior work" paragraph (ToMe, ToMeSD,
DeepCache, LCM/DMD, FlashAttention).

## 9. Three-Person Split (parallelizable after Phase 2)

- **Member A — Core & Speed.** Phases 0, 1, 2; owns `merging.py` + the `attn_seq_reduce`
  attention surgery (Phase 4). Hardest engineering. Deliverable: real speedup numbers.
- **Member B — Temporal & Quality.** Phase 5 spatiotemporal variant,
  `protect_first_frame` / layer-strategy ablations, artifact analysis, qualitative grids.
  Deliverable: the "spatiotemporal beats spatial on flicker" story.
- **Member C — Eval & Report.** Phase 6 metrics (CLIP, temporal-consistency, VBench
  subset, optional FVD/warp-error), `run_matrix.py`, all plots, report writing, NeurIPS
  checklist, demo video. Deliverable: the Pareto front and the paper.

Sync points: after Phase 2 (interface freeze), after Phase 4 (speed confirmed), after
Phase 6 (data frozen for writing).

## 10. Risk Register (and what to do)

- **R1 No real speedup** -> mitigated by mandatory `attn_seq_reduce`; verify with
  `transformer_seconds`, not end-to-end.
- **R2 Offload masks GPU gains** -> always report transformer-only timing; borrow a
  >=24GB GPU for one clean run.
- **R3 Flicker at high ratio** -> start 0.1, `protect_first_frame`, `late_off` strategy;
  spatiotemporal mode tuned with `temporal_window=1` first.
- **R4 RoPE/index bugs in attn surgery** -> Phase 3 correctness bridge before Phase 4;
  ratio-0 identity test at every phase.
- **R5 VBench setup cost** -> use the dataset-free subset; document exactly which
  dimensions; FVD stays optional.

## 11. Definition of Done

- `attn_seq_reduce` shows measurable transformer-time reduction at ratio 0.2 with VBench
  composite drop within an explicitly stated tolerance.
- Both spatial and spatiotemporal variants implemented and compared on flicker.
- `report/results.csv` + 5 figures + report draft + NeurIPS checklist + demo video.
- Baseline path bit-identical when merging disabled (regression-safe).

## 12. Reference Plan

Core:

- Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023. (bipartite matching,
  size-weighted average, training-free merging)
- Bolya & Hoffman, "Token Merging for Fast Stable Diffusion", CVPRW 2023. (ToMe applied to
  diffusion / generation; closest prior art — state our delta: video + spatiotemporal +
  real attention-seq reduction in DiT)
- Yang et al., "CogVideoX", 2024. (baseline model + DiT video architecture)
- Peebles & Xie, "DiT: Scalable Diffusion Models with Transformers", ICCV 2023.

Compared / related:

- Ma et al., "DeepCache", NeurIPS 2024. (temporal feature reuse across steps — orthogonal,
  combinable)
- Luo et al., "Latent Consistency Models", 2023; DMD. (step reduction via training —
  explicitly NOT our route)
- Dao et al., "FlashAttention", NeurIPS 2022. (kernel-level attention — orthogonal to
  token-count reduction; combinable)

Evaluation:

- Huang et al., "VBench", 2023. (primary benchmark; we use a documented subset)
- Unterthiner et al., FVD, 2018. (optional distributional metric)
- Hessel et al., "CLIPScore", 2021. (semantic alignment)
