from __future__ import annotations

from dataclasses import dataclass

import torch

from .matching import match_features
from .partition import VisualLayout, source_destination_indices


@dataclass
class ReductionPlan:
    src_idx: torch.Tensor
    dst_idx: torch.Tensor
    keep_idx: torch.Tensor
    dst_pos: torch.Tensor
    sizes: torch.Tensor
    num_tokens: int
    layout: VisualLayout
    reduce_mode: str
    similarity_type: str

    @property
    def removed_tokens(self) -> int:
        return int(self.src_idx.shape[1])

    @property
    def kept_tokens(self) -> int:
        return int(self.keep_idx.shape[1])


def _flatten_feature(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    if x.ndim == 3:
        return x, None
    if x.ndim == 4:
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d), (h, d)
    raise AssertionError(f"expected [B,N,C] or [B,H,N,D], got {tuple(x.shape)}")


def _unflatten_feature(x: torch.Tensor, head_shape: tuple[int, int] | None) -> torch.Tensor:
    if head_shape is None:
        return x
    h, d = head_shape
    b, n, _ = x.shape
    return x.reshape(b, n, h, d).transpose(1, 2).contiguous()


def _positions_in_keep(keep_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(keep_idx.contiguous(), dst_idx.contiguous())


def make_identity_plan(
    batch_size: int,
    layout: VisualLayout,
    device: torch.device,
    dtype: torch.dtype,
    reduce_mode: str = "replace",
    similarity_type: str = "cosine",
) -> ReductionPlan:
    keep_idx = torch.arange(layout.num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
    empty = torch.empty(batch_size, 0, dtype=torch.long, device=device)
    sizes = torch.ones(batch_size, layout.num_tokens, device=device, dtype=dtype)
    return ReductionPlan(empty, empty, keep_idx, empty, sizes, layout.num_tokens, layout, reduce_mode, similarity_type)


def build_reduction_plan(
    features: torch.Tensor,
    layout: VisualLayout,
    ratio: float,
    dst_stride: tuple[int, int, int],
    similarity_type: str = "cosine",
    reduce_mode: str = "replace",
    protect_first_frame: bool = True,
    protect_topk_ratio: float = 0.0,
    cfg_consistent: bool = True,
    partition_mode: str = "random_chunk",
    generator: torch.Generator | None = None,
    partition_cache_key: object | None = None,
) -> ReductionPlan:
    """Build source/destination indices for asymmetric token reduction."""

    flat, _ = _flatten_feature(features)
    batch_size, n_tokens, _ = flat.shape
    layout.validate(n_tokens)

    if ratio <= 0:
        return make_identity_plan(batch_size, layout, flat.device, flat.dtype, reduce_mode, similarity_type)

    src_candidates, dst_candidates = source_destination_indices(
        layout,
        dst_stride,
        flat.device,
        protect_first_frame=protect_first_frame,
        partition_mode=partition_mode,
        generator=generator,
        cache_key_extra=partition_cache_key,
    )
    pool_tokens = int(src_candidates.numel() + dst_candidates.numel())
    num_unreduce_src = int(pool_tokens * (1.0 - ratio)) - int(dst_candidates.numel())
    num_unreduce_src = max(0, min(int(src_candidates.numel()), num_unreduce_src))
    r = int(src_candidates.numel()) - num_unreduce_src
    if r <= 0 or dst_candidates.numel() == 0:
        return make_identity_plan(batch_size, layout, flat.device, flat.dtype, reduce_mode, similarity_type)

    # Matching is intentionally performed on fp32-normal-safe features, but the
    # returned indices remain lightweight int64 tensors only.
    dst_feat = flat[:, dst_candidates]
    src_feat = flat[:, src_candidates]
    match = match_features(dst_feat, src_feat, similarity_type=similarity_type)
    source_scores = match.similarity_for_src.clone()

    if protect_topk_ratio > 0:
        n_protect = int(src_candidates.numel() * protect_topk_ratio)
        if n_protect > 0:
            importance = src_feat.float().norm(dim=-1)
            protected = importance.topk(n_protect, dim=1).indices
            penalty = torch.full_like(source_scores, torch.finfo(source_scores.dtype).min / 2)
            source_scores.scatter_(1, protected, penalty.gather(1, protected))

    top_src_local = source_scores.argsort(dim=1, descending=True, stable=True)[:, :r]
    src_idx = src_candidates[top_src_local]
    dst_local = torch.gather(match.dst_for_src, 1, top_src_local)
    dst_idx = dst_candidates[dst_local]

    if cfg_consistent and batch_size == 2:
        src_idx = src_idx[1:2].expand(batch_size, -1)
        dst_idx = dst_idx[1:2].expand(batch_size, -1)

    removed = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=flat.device)
    removed.scatter_(1, src_idx, True)
    keep_idx = (~removed).long().argsort(dim=1, descending=True, stable=True)[:, : n_tokens - r]
    keep_idx = keep_idx.sort(dim=1).values
    dst_pos = _positions_in_keep(keep_idx, dst_idx)

    sizes = torch.ones(batch_size, n_tokens - r, device=flat.device, dtype=flat.dtype)
    if reduce_mode == "mean":
        delta = torch.zeros_like(sizes)
        delta.scatter_add_(1, dst_pos, torch.ones_like(dst_pos, dtype=flat.dtype))
        sizes = sizes + delta
    elif reduce_mode != "replace":
        raise ValueError(f"Unknown reduce_mode: {reduce_mode!r}")

    return ReductionPlan(src_idx, dst_idx, keep_idx, dst_pos, sizes, n_tokens, layout, reduce_mode, similarity_type)


def reduce_sequence(x: torch.Tensor, plan: ReductionPlan, reduce_mode: str | None = None) -> torch.Tensor:
    """Reduce a visual-token tensor using a precomputed plan.

    Supports ``[B,N,C]`` and attention-head layout ``[B,H,N,D]``. In replace
    mode, source tokens are discarded. In mean mode, source values are averaged
    into their matched destination token.
    """

    flat, head_shape = _flatten_feature(x)
    batch_size, n_tokens, channels = flat.shape
    if n_tokens != plan.num_tokens:
        raise AssertionError(f"plan expects {plan.num_tokens} tokens, got {n_tokens}")
    if plan.removed_tokens == 0:
        return x.clone()

    mode = reduce_mode or plan.reduce_mode
    kept = torch.gather(flat, 1, plan.keep_idx.unsqueeze(-1).expand(-1, -1, channels))
    if mode == "replace":
        return _unflatten_feature(kept, head_shape)
    if mode != "mean":
        raise ValueError(f"Unknown reduce_mode: {mode!r}")

    src = torch.gather(flat, 1, plan.src_idx.unsqueeze(-1).expand(-1, -1, channels))
    accum = torch.zeros_like(kept)
    accum.scatter_add_(1, plan.dst_pos.unsqueeze(-1).expand(-1, -1, channels), src)
    reduced = (kept + accum) / plan.sizes.unsqueeze(-1).clamp(min=1)
    return _unflatten_feature(reduced, head_shape)


def restore_sequence(x_reduced: torch.Tensor, plan: ReductionPlan) -> torch.Tensor:
    """Restore reduced Q-output back to the original visual-token length."""

    flat, head_shape = _flatten_feature(x_reduced)
    batch_size, kept_tokens, channels = flat.shape
    if kept_tokens != plan.kept_tokens:
        raise AssertionError(f"plan expects {plan.kept_tokens} kept tokens, got {kept_tokens}")
    if plan.removed_tokens == 0:
        return x_reduced.clone()

    full = torch.zeros(batch_size, plan.num_tokens, channels, device=flat.device, dtype=flat.dtype)
    full.scatter_(1, plan.keep_idx.unsqueeze(-1).expand(-1, -1, channels), flat)
    dst_values = torch.gather(flat, 1, plan.dst_pos.unsqueeze(-1).expand(-1, -1, channels))
    full.scatter_(1, plan.src_idx.unsqueeze(-1).expand(-1, -1, channels), dst_values)
    return _unflatten_feature(full, head_shape)


def key_size_log_bias(plan: ReductionPlan, num_text_tokens: int, num_heads: int) -> torch.Tensor:
    """Return SDPA bias for proportional attention over reduced K/V tokens."""

    text_bias = torch.zeros(
        plan.sizes.shape[0], num_text_tokens, device=plan.sizes.device, dtype=plan.sizes.dtype
    )
    bias = torch.cat([text_bias, plan.sizes.clamp(min=1).log()], dim=1)
    return bias.unsqueeze(1).unsqueeze(1).expand(-1, num_heads, -1, -1)
