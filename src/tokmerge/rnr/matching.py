from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_MATCH_DIM = 128


@dataclass
class MatchResult:
    """Batched best-destination matching result for source tokens."""

    dst_for_src: torch.Tensor
    source_order: torch.Tensor
    similarity_for_src: torch.Tensor
    similarity: torch.Tensor


def compute_similarity(dst: torch.Tensor, src: torch.Tensor, similarity_type: str = "cosine") -> torch.Tensor:
    """Compute ``[B, N_dst, N_src]`` similarity scores.

    Higher scores always mean "more similar"; Euclidean distance is returned as
    negative squared distance so it follows the same ranking convention.
    """

    if dst.ndim != 3 or src.ndim != 3:
        raise AssertionError("dst and src must both be [B, N, C]")
    if dst.shape[0] != src.shape[0] or dst.shape[2] != src.shape[2]:
        raise AssertionError(f"incompatible shapes: dst={tuple(dst.shape)}, src={tuple(src.shape)}")

    if dst.shape[-1] > _MATCH_DIM:
        stride = max(1, dst.shape[-1] // _MATCH_DIM)
        dst = dst[..., ::stride]
        src = src[..., ::stride]

    similarity_type = similarity_type.lower()
    if similarity_type == "cosine":
        dst_n = F.normalize(dst.float(), dim=-1)
        src_n = F.normalize(src.float(), dim=-1)
        return torch.bmm(dst_n, src_n.transpose(1, 2)).to(dst.dtype)
    if similarity_type == "dot":
        return torch.bmm(dst, src.transpose(1, 2))
    if similarity_type == "euclidean":
        dst_f = dst.float()
        src_f = src.float()
        dst_norm = (dst_f * dst_f).sum(dim=-1, keepdim=True)
        src_norm = (src_f * src_f).sum(dim=-1).unsqueeze(1)
        sq_dist = (dst_norm + src_norm - 2.0 * torch.bmm(dst_f, src_f.transpose(1, 2))).clamp_min_(0.0)
        return (-sq_dist).to(dst.dtype)
    if similarity_type == "random":
        return torch.rand(dst.shape[0], dst.shape[1], src.shape[1], device=dst.device, dtype=dst.dtype)
    raise ValueError(f"Unknown similarity_type: {similarity_type!r}")


def match_features(dst: torch.Tensor, src: torch.Tensor, similarity_type: str = "cosine") -> MatchResult:
    """Match every source token to its best destination and rank sources.

    Returns local destination indices for each source, source ranking from most
    to least redundant, the best similarity value per source, and the full
    similarity matrix.
    """

    similarity = compute_similarity(dst, src, similarity_type)
    best_score, dst_for_src = similarity.max(dim=1)
    source_order = best_score.argsort(dim=1, descending=True, stable=True)
    return MatchResult(
        dst_for_src=dst_for_src,
        source_order=source_order,
        similarity_for_src=best_score,
        similarity=similarity,
    )
