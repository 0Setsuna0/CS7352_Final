"""Pure-tensor token merging: bipartite soft matching, merge, unmerge.

No diffusers dependency. All functions operate on the video-token portion only;
the caller splits text/video before calling and re-concatenates after.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MergeConfig:
    enabled: bool = False
    ratio: float = 0.0
    mode: str = "spatial"               # "spatial" | "spatiotemporal"
    scope: str = "block"                # "block" | "kv_only" | "attn_only" | "pre_attn_restore"
    rope_mode: str = "pre_rope"         # "pre_rope" | "dst"
    prop_attn: bool = True              # log(size) bias in softmax
    match_feature: str = "hidden_norm"  # "hidden_norm" | "attn_k"
    layers: tuple[int, ...] = ()
    temporal_window: int = 1
    protect_first_frame: bool = True    # latent frame 0
    skip_early_ratio: float = 0.0       # skip merging for the first X% of timesteps (0-1)
    skip_late_ratio: float = 0.0        # skip merging for the last X% of timesteps (0-1)
    partition: str = "checkerboard"     # "checkerboard" | "checkerboard_shifted"
    reuse_interval: int = 1             # reuse matching pattern for this many block calls
    # --- Quality enhancements (v4) ---
    ratio_schedule: str = "constant"    # "constant" | "cosine" | "bell" | "linear_decay"
    layer_ratio_decay: float = 0.0      # 0=all layers same ratio; >0=later active layers use less
    protect_topk_ratio: float = 0.0     # protect top X% tokens by activation magnitude
    cfg_consistent: bool = False        # force cond/uncond branches to share merge pattern
    unmerge_mode: str = "copy"          # "copy" | "interpolate"


@dataclass
class RestoreInfo:
    src_idx: torch.Tensor    # [B, r]       — indices of absorbed tokens
    dst_idx: torch.Tensor    # [B, r]       — indices they merged into
    keep_idx: torch.Tensor   # [B, N-r]     — indices of kept tokens
    sizes: torch.Tensor      # [B, N-r]     — weight per kept token (≥1)
    num_video_tokens: int
    grid: tuple[int, int, int]  # (frames, gh, gw)


_partition_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_indices_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_fixed_match_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _build_spatial_partition(
    n_video: int,
    frames: int,
    gh: int,
    gw: int,
    partition: str = "checkerboard",
    partition_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split video tokens into two disjoint sets (src, dst) for bipartite matching.

    For spatial mode we use a checkerboard on the (h, w) grid within each frame
    so that roughly half the tokens are in each set. ``checkerboard_shifted``
    flips the source/destination color by offset, reducing fixed grid artifacts
    when different transformer layers use alternating offsets.
    """
    if partition == "checkerboard":
        offset = 0
    elif partition == "checkerboard_shifted":
        offset = partition_offset % 2
    else:
        raise ValueError(f"Unknown partition: {partition!r}")

    key = (n_video, frames, gh, gw, partition, offset)
    if key in _partition_cache:
        return _partition_cache[key]

    idx = torch.arange(n_video)
    hw = idx % (gh * gw)
    h = hw // gw
    w = hw % gw
    is_src = ((h + w + offset) % 2 == 0)
    result = (is_src, ~is_src)
    _partition_cache[key] = result
    return result


def _build_temporal_mask(
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    gh: int,
    gw: int,
    temporal_window: int,
) -> torch.Tensor:
    """Build temporal-neighborhood mask for spatiotemporal matching.

    Returns temporal_mask[i, j] which is True when src token i and dst token j
    are within temporal_window frames of each other.
    Uses the actual post-filtered src/dst indices.
    """
    src_frames = src_indices // (gh * gw)
    dst_frames = dst_indices // (gh * gw)
    return (src_frames.unsqueeze(1) - dst_frames.unsqueeze(0)).abs() <= temporal_window


def compute_effective_ratio(
    base_ratio: float,
    timestep: float,
    schedule: str,
    skip_early: float = 0.0,
    skip_late: float = 0.0,
) -> float:
    """Compute the effective merge ratio for a given timestep.

    Args:
        base_ratio: the configured max ratio.
        timestep: current timestep value in [0, 1000] (higher = noisier/earlier).
        schedule: "constant", "cosine", "bell", or "linear_decay".
        skip_early: fraction of initial (noisy) steps to skip entirely.
        skip_late: fraction of final (detail) steps to skip entirely.

    Returns:
        Effective ratio in [0, base_ratio].
    """
    if base_ratio <= 0:
        return 0.0

    progress = 1.0 - timestep / 1000.0  # 0 at start → 1 at end of denoising

    if skip_early > 0 and progress < skip_early:
        return 0.0
    if skip_late > 0 and progress > (1.0 - skip_late):
        return 0.0

    if schedule == "constant":
        return base_ratio
    elif schedule == "cosine":
        # Smooth bell-like curve peaking at midpoint of active window
        active_start = skip_early
        active_end = 1.0 - skip_late
        active_len = active_end - active_start
        if active_len <= 0:
            return 0.0
        local_progress = (progress - active_start) / active_len
        return base_ratio * math.sin(math.pi * local_progress)
    elif schedule == "bell":
        # Gaussian-like: peaks around 40-60% overall progress
        active_start = skip_early
        active_end = 1.0 - skip_late
        active_len = active_end - active_start
        if active_len <= 0:
            return 0.0
        local_progress = (progress - active_start) / active_len
        return base_ratio * math.exp(-((local_progress - 0.5) ** 2) / 0.08)
    elif schedule == "linear_decay":
        # Starts at full ratio, linearly decays to 0 at the end of active window
        active_start = skip_early
        active_end = 1.0 - skip_late
        active_len = active_end - active_start
        if active_len <= 0:
            return 0.0
        local_progress = (progress - active_start) / active_len
        return base_ratio * (1.0 - local_progress)
    else:
        return base_ratio


def compute_layer_ratio(
    base_ratio: float,
    block_index: int,
    active_layers: tuple[int, ...],
    layer_ratio_decay: float,
) -> float:
    """Apply per-layer ratio decay: earlier active layers keep full ratio,
    later active layers use progressively less.

    Args:
        base_ratio: effective ratio after timestep scheduling.
        block_index: current transformer block index.
        active_layers: sorted tuple of all active layer indices.
        layer_ratio_decay: 0 = uniform, 1 = last active layer gets 0 ratio.

    Returns:
        Layer-adjusted ratio.
    """
    if layer_ratio_decay <= 0 or len(active_layers) <= 1:
        return base_ratio

    rank = active_layers.index(block_index) if block_index in active_layers else 0
    n = len(active_layers) - 1
    decay_factor = 1.0 - layer_ratio_decay * (rank / n)
    return base_ratio * max(0.0, decay_factor)


def bipartite_soft_match(
    metric: torch.Tensor,
    r: int,
    frames: int,
    gh: int,
    gw: int,
    mode: str = "spatial",
    temporal_window: int = 1,
    protect_first_frame: bool = True,
    match_feature: str = "hidden_norm",
    partition: str = "checkerboard",
    partition_offset: int = 0,
    protect_topk_ratio: float = 0.0,
    cfg_consistent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bipartite soft matching on L2-normalised token features.

    Args:
        metric: [B, N, C] — L2-normalised hidden states (video tokens only).
        r: number of tokens to merge (remove).
        frames, gh, gw: video grid dimensions so that N = frames * gh * gw.
        mode: "spatial" or "spatiotemporal".
        temporal_window: for spatiotemporal, frame distance for cross-frame matching.
        protect_first_frame: never absorb tokens from latent frame 0.
        partition: source/destination partition strategy.
        partition_offset: parity offset for shifted partition strategies.
        protect_topk_ratio: protect top X% of tokens by activation norm from absorption.
        cfg_consistent: if True and B==2 (CFG), use cond branch pattern for both.

    Returns:
        src_idx:  [B, r]    — absorbed token indices (into original N).
        dst_idx:  [B, r]    — destination token indices they merge into.
        keep_idx: [B, N-r]  — indices of all kept tokens (sorted).
        sizes:    [B, N-r]  — initial size of each kept token (all 1.0).
    """
    B, N, C = metric.shape
    assert N == frames * gh * gw, f"N={N} != frames*gh*gw={frames*gh*gw}"

    if r == 0:
        keep_idx = torch.arange(N, device=metric.device).unsqueeze(0).expand(B, -1)
        sizes = torch.ones(B, N, device=metric.device, dtype=metric.dtype)
        src_idx = torch.empty(B, 0, dtype=torch.long, device=metric.device)
        dst_idx = torch.empty(B, 0, dtype=torch.long, device=metric.device)
        return src_idx, dst_idx, keep_idx, sizes

    _idx_key = (
        N,
        frames,
        gh,
        gw,
        mode,
        protect_first_frame,
        partition,
        partition_offset % 2,
        str(metric.device),
    )
    if _idx_key in _indices_cache:
        src_indices, dst_indices = _indices_cache[_idx_key]
    else:
        if mode in ("spatial", "spatiotemporal"):
            is_src, is_dst = _build_spatial_partition(
                N,
                frames,
                gh,
                gw,
                partition=partition,
                partition_offset=partition_offset,
            )
            is_src = is_src.to(metric.device)
            is_dst = is_dst.to(metric.device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if protect_first_frame:
            frame0_mask = torch.arange(N, device=metric.device) < (gh * gw)
            is_src = is_src & ~frame0_mask

        src_indices = is_src.nonzero(as_tuple=False).squeeze(-1)  # [n_src]
        dst_indices = is_dst.nonzero(as_tuple=False).squeeze(-1)  # [n_dst]
        _indices_cache[_idx_key] = (src_indices, dst_indices)

    n_src = src_indices.shape[0]
    n_dst = dst_indices.shape[0]
    r = min(r, n_src)

    if r == 0:
        keep_idx = torch.arange(N, device=metric.device).unsqueeze(0).expand(B, -1)
        sizes = torch.ones(B, N, device=metric.device, dtype=metric.dtype)
        src_idx = torch.empty(B, 0, dtype=torch.long, device=metric.device)
        dst_idx = torch.empty(B, 0, dtype=torch.long, device=metric.device)
        return src_idx, dst_idx, keep_idx, sizes

    if match_feature == "fixed":
        # Fixed positional merge: pair each src with its nearest dst neighbor.
        # Fully deterministic, no similarity computation, batch-independent.
        _fixed_key = (
            N,
            frames,
            gh,
            gw,
            mode,
            temporal_window,
            protect_first_frame,
            r,
            partition,
            partition_offset % 2,
            str(metric.device),
        )
        if _fixed_key in _fixed_match_cache:
            batch_src_idx, batch_dst_idx, keep_idx = _fixed_match_cache[_fixed_key]
            batch_src_idx = batch_src_idx.expand(B, -1)
            batch_dst_idx = batch_dst_idx.expand(B, -1)
            keep_idx = keep_idx.expand(B, -1)
        else:
            # For each src, find nearest dst by L1 grid distance
            src_hw = src_indices % (gh * gw)
            src_h, src_w = src_hw // gw, src_hw % gw
            src_t = src_indices // (gh * gw)

            dst_hw = dst_indices % (gh * gw)
            dst_h, dst_w = dst_hw // gw, dst_hw % gw
            dst_t = dst_indices // (gh * gw)

            # Grid distance: |h_diff| + |w_diff|; for spatial, same frame, so no t
            h_diff = (src_h.unsqueeze(1) - dst_h.unsqueeze(0)).abs()
            w_diff = (src_w.unsqueeze(1) - dst_w.unsqueeze(0)).abs()
            t_diff = (src_t.unsqueeze(1) - dst_t.unsqueeze(0)).abs()
            grid_dist = h_diff + w_diff + t_diff * (gh + gw)  # penalize cross-frame

            if mode == "spatiotemporal":
                grid_dist = grid_dist.masked_fill(t_diff > temporal_window, 99999)

            nearest_dst = grid_dist.argmin(dim=1)  # [n_src]
            # Take first r src tokens (they're all equally good in fixed mode)
            take_src = src_indices[:r]  # [r]
            take_dst = dst_indices[nearest_dst[:r]]  # [r]

            removed = torch.zeros(N, dtype=torch.bool, device=metric.device)
            removed[take_src] = True
            keep = (~removed).nonzero(as_tuple=False).squeeze(-1)  # [N-r]

            batch_src_idx = take_src.unsqueeze(0)
            batch_dst_idx = take_dst.unsqueeze(0)
            keep_idx = keep.unsqueeze(0)
            _fixed_match_cache[_fixed_key] = (batch_src_idx, batch_dst_idx, keep_idx)
            batch_src_idx = batch_src_idx.expand(B, -1)
            batch_dst_idx = batch_dst_idx.expand(B, -1)
            keep_idx = keep_idx.expand(B, -1)

        sizes = torch.ones(B, N - r, device=metric.device, dtype=metric.dtype)
        return batch_src_idx, batch_dst_idx, keep_idx, sizes

    # Content-adaptive matching (hidden_norm / attn_k)
    # Subsample feature dims for matching to reduce bmm cost (1920 → ≤128)
    _MATCH_DIM = 128
    if C > _MATCH_DIM:
        stride = C // _MATCH_DIM
        metric_sub = metric[:, :, ::stride]  # [B, N, ~128]
    else:
        metric_sub = metric

    metric_src = metric_sub[:, src_indices]  # [B, n_src, C']
    metric_dst = metric_sub[:, dst_indices]  # [B, n_dst, C']

    scores = torch.bmm(metric_src, metric_dst.transpose(1, 2))  # [B, n_src, n_dst]

    if mode == "spatiotemporal":
        temporal_mask = _build_temporal_mask(
            src_indices, dst_indices, gh, gw, temporal_window
        ).to(metric.device)
        scores = scores.masked_fill(~temporal_mask.unsqueeze(0), -torch.inf)

    # Importance-based protection: penalise scores of high-importance src tokens
    # so they are unlikely to be selected for absorption.
    if protect_topk_ratio > 0:
        importance = metric[:, src_indices].norm(dim=-1)  # [B, n_src]
        n_protect = int(n_src * protect_topk_ratio)
        if n_protect > 0:
            _, protect_local = importance.topk(n_protect, dim=1)  # [B, n_protect]
            penalty = torch.zeros_like(scores[:, :, 0])  # [B, n_src]
            penalty.scatter_(1, protect_local, -1e9)
            scores = scores + penalty.unsqueeze(2)

    best_dst_score, best_dst_local = scores.max(dim=2)  # [B, n_src]
    _, top_src_local = best_dst_score.topk(r, dim=1)  # [B, r]

    batch_src_idx = src_indices[top_src_local]  # [B, r]
    matched_dst_local = torch.gather(best_dst_local, 1, top_src_local)  # [B, r]
    batch_dst_idx = dst_indices[matched_dst_local]  # [B, r]

    # CFG-consistent: force both batch elements to use the same merge pattern
    if cfg_consistent and B == 2:
        batch_src_idx = batch_src_idx[1:2].expand(B, -1)
        batch_dst_idx = batch_dst_idx[1:2].expand(B, -1)

    removed = torch.zeros(B, N, dtype=torch.bool, device=metric.device)
    removed.scatter_(1, batch_src_idx, True)
    not_removed = (~removed).long()
    sorted_indices = torch.argsort(-not_removed, dim=1, stable=True)
    keep_idx = sorted_indices[:, :N - r]
    keep_idx = keep_idx.sort(dim=1).values

    sizes = torch.ones(B, N - r, device=metric.device, dtype=metric.dtype)

    return batch_src_idx, batch_dst_idx, keep_idx, sizes


def _find_positions_vectorized(keep_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
    """For each element in dst_idx, find its position in keep_idx.

    Assumes keep_idx is sorted along dim=1 (which it is by construction).
    Uses searchsorted for O(r log K) instead of O(r * K).
    """
    return torch.searchsorted(keep_idx.contiguous(), dst_idx.contiguous())


def merge_tokens(
    x: torch.Tensor,
    info: RestoreInfo,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Size-weighted average merge. Fully vectorized.

    Args:
        x: [B, N, C] video tokens.
        info: RestoreInfo with indices and sizes.

    Returns:
        merged: [B, N-r, C] merged video tokens.
        new_sizes: [B, N-r] updated sizes after absorbing src tokens.
    """
    B, N, C = x.shape
    r = info.src_idx.shape[1]

    if r == 0:
        return x.clone(), info.sizes.clone()

    K = info.keep_idx.shape[1]
    keep = torch.gather(x, 1, info.keep_idx.unsqueeze(-1).expand(-1, -1, C))  # [B, K, C]
    src = torch.gather(x, 1, info.src_idx.unsqueeze(-1).expand(-1, -1, C))    # [B, r, C]

    dst_in_keep = _find_positions_vectorized(info.keep_idx, info.dst_idx)  # [B, r]

    new_sizes = info.sizes.clone()  # [B, K]

    # Vectorized scatter-add: accumulate src tokens into their dst positions
    # First, count how many sources go to each dst position
    size_delta = torch.zeros(B, K, device=x.device, dtype=x.dtype)
    size_delta.scatter_add_(1, dst_in_keep, torch.ones_like(dst_in_keep, dtype=x.dtype))

    # Accumulate src values weighted by 1 into dst positions
    weighted_src = torch.zeros(B, K, C, device=x.device, dtype=x.dtype)
    weighted_src.scatter_add_(1, dst_in_keep.unsqueeze(-1).expand(-1, -1, C), src)

    # Compute new values: (old_val * old_size + accumulated_src) / (old_size + count)
    old_sizes = new_sizes.clone()
    new_sizes = old_sizes + size_delta
    keep = (keep * old_sizes.unsqueeze(-1) + weighted_src) / new_sizes.unsqueeze(-1).clamp(min=1)

    return keep, new_sizes


def unmerge_tokens(
    merged_x: torch.Tensor,
    info: RestoreInfo,
) -> torch.Tensor:
    """Scatter merged tokens back to full layout. Fully vectorized.

    Absorbed tokens copy their destination's value.

    Args:
        merged_x: [B, N-r, C]
        info: RestoreInfo.

    Returns:
        full_x: [B, N, C]
    """
    B, _, C = merged_x.shape
    N = info.num_video_tokens
    r = info.src_idx.shape[1]

    if r == 0:
        return merged_x.clone()

    full_x = torch.zeros(B, N, C, device=merged_x.device, dtype=merged_x.dtype)
    full_x.scatter_(1, info.keep_idx.unsqueeze(-1).expand(-1, -1, C), merged_x)

    dst_in_keep = _find_positions_vectorized(info.keep_idx, info.dst_idx)
    dst_values = torch.gather(
        merged_x, 1, dst_in_keep.unsqueeze(-1).expand(-1, -1, C)
    )
    full_x.scatter_(1, info.src_idx.unsqueeze(-1).expand(-1, -1, C), dst_values)

    return full_x


def unmerge_tokens_interpolated(
    merged_x: torch.Tensor,
    info: RestoreInfo,
) -> torch.Tensor:
    """Interpolated unmerge: absorbed tokens get a blend of their dst value
    and the mean of spatial neighbours in the kept set.

    This reduces the "block copy" artifact of standard unmerge where src
    positions are exact duplicates of their dst, producing grid patterns
    after 40 denoising steps.

    Args:
        merged_x: [B, N-r, C]
        info: RestoreInfo.

    Returns:
        full_x: [B, N, C]
    """
    B, _, C = merged_x.shape
    N = info.num_video_tokens
    r = info.src_idx.shape[1]

    if r == 0:
        return merged_x.clone()

    # Start with standard copy-unmerge as the base
    full_x = torch.zeros(B, N, C, device=merged_x.device, dtype=merged_x.dtype)
    full_x.scatter_(1, info.keep_idx.unsqueeze(-1).expand(-1, -1, C), merged_x)

    dst_in_keep = _find_positions_vectorized(info.keep_idx, info.dst_idx)
    dst_values = torch.gather(
        merged_x, 1, dst_in_keep.unsqueeze(-1).expand(-1, -1, C)
    )

    # Compute spatial-neighbour average for each src position.
    # For each src token, find its 4-connected neighbours among kept tokens.
    frames, gh, gw = info.grid
    src_pos = info.src_idx  # [B, r]

    src_t = src_pos // (gh * gw)
    src_hw = src_pos % (gh * gw)
    src_h = src_hw // gw
    src_w = src_hw % gw

    # Collect values from up to 4 spatial neighbours (same frame, ±1 in h or w)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbour_sum = torch.zeros(B, r, C, device=merged_x.device, dtype=merged_x.dtype)
    neighbour_count = torch.zeros(B, r, 1, device=merged_x.device, dtype=merged_x.dtype)

    for dh, dw in offsets:
        nh = src_h + dh
        nw = src_w + dw
        valid = (nh >= 0) & (nh < gh) & (nw >= 0) & (nw < gw)  # [B, r]
        neighbour_flat = src_t * (gh * gw) + nh.clamp(0, gh - 1) * gw + nw.clamp(0, gw - 1)
        # Gather from full_x (which has kept tokens placed already)
        neighbour_vals = torch.gather(
            full_x, 1, neighbour_flat.unsqueeze(-1).expand(-1, -1, C)
        )
        neighbour_sum += neighbour_vals * valid.unsqueeze(-1).float()
        neighbour_count += valid.unsqueeze(-1).float()

    # Blend: 70% dst value + 30% neighbour average (where neighbours exist)
    has_neighbours = neighbour_count.squeeze(-1) > 0  # [B, r]
    neighbour_avg = neighbour_sum / neighbour_count.clamp(min=1)

    blend_weight = 0.3
    blended = torch.where(
        has_neighbours.unsqueeze(-1),
        dst_values * (1 - blend_weight) + neighbour_avg * blend_weight,
        dst_values,
    )

    full_x.scatter_(1, info.src_idx.unsqueeze(-1).expand(-1, -1, C), blended)
    return full_x


def merge_kv_tokens(
    x: torch.Tensor,
    info: RestoreInfo,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge token axis for attention K/V tensors.

    Args:
        x: [B, H, N, D] video key/value tokens.
        info: RestoreInfo over the N video-token positions.

    Returns:
        merged: [B, H, N-r, D]
        new_sizes: [B, N-r]
    """
    B, H, N, D = x.shape
    flat = x.transpose(1, 2).reshape(B, N, H * D)
    merged_flat, new_sizes = merge_tokens(flat, info)
    merged = merged_flat.reshape(B, -1, H, D).transpose(1, 2).contiguous()
    return merged, new_sizes


def size_log_bias(
    info: RestoreInfo,
    num_text_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """Additive attention bias: log(sizes) for proportional attention.

    Returns a tensor broadcastable to [B, num_heads, Q_len, K_len] where
    K_len = num_text_tokens + (N_video - r). Text keys get zero bias.
    """
    B = info.sizes.shape[0]
    n_merged = info.sizes.shape[1]  # N_video - r

    text_bias = torch.zeros(B, num_text_tokens, device=info.sizes.device, dtype=info.sizes.dtype)
    video_bias = info.sizes.clamp(min=1).log()  # [B, n_merged]

    # [B, 1, 1, num_text + n_merged] — broadcast over heads and query positions
    bias = torch.cat([text_bias, video_bias], dim=1)  # [B, K_len]
    return bias.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, K_len]
