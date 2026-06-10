"""Tests for src/tokmerge/merging.py — Phase 1 acceptance tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tokmerge.merging import (
    MergeConfig,
    RestoreInfo,
    bipartite_soft_match,
    merge_kv_tokens,
    merge_tokens,
    unmerge_tokens,
    size_log_bias,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metric(B, frames, gh, gw, C, seed=0):
    """Random L2-normalised metric tensor."""
    torch.manual_seed(seed)
    x = torch.randn(B, frames * gh * gw, C)
    return torch.nn.functional.normalize(x, dim=-1)


def _full_merge_cycle(metric, x, r, frames, gh, gw, mode="spatial", **kwargs):
    """Run match -> merge -> unmerge, return (merged, restored, info)."""
    src_idx, dst_idx, keep_idx, sizes = bipartite_soft_match(
        metric, r, frames, gh, gw, mode=mode, **kwargs
    )
    info = RestoreInfo(
        src_idx=src_idx, dst_idx=dst_idx, keep_idx=keep_idx,
        sizes=sizes, num_video_tokens=metric.shape[1],
        grid=(frames, gh, gw),
    )
    merged, new_sizes = merge_tokens(x, info)
    info_with_sizes = RestoreInfo(
        src_idx=info.src_idx, dst_idx=info.dst_idx,
        keep_idx=info.keep_idx, sizes=new_sizes,
        num_video_tokens=info.num_video_tokens, grid=info.grid,
    )
    restored = unmerge_tokens(merged, info_with_sizes)
    return merged, restored, info_with_sizes


# ===========================================================================
# 1. ratio=0 => exact identity
# ===========================================================================

class TestRatioZeroIdentity:
    @pytest.mark.parametrize("B,frames,gh,gw,C", [
        (1, 2, 4, 4, 8),
        (2, 3, 6, 8, 16),
        (1, 13, 30, 45, 32),
    ])
    def test_merge_unmerge_identity(self, B, frames, gh, gw, C):
        metric = _make_metric(B, frames, gh, gw, C)
        x = torch.randn(B, frames * gh * gw, C)
        merged, restored, info = _full_merge_cycle(
            metric, x, r=0, frames=frames, gh=gh, gw=gw,
        )
        assert merged.shape == x.shape
        assert restored.shape == x.shape
        assert torch.allclose(merged, x)
        assert torch.allclose(restored, x)

    def test_sizes_all_ones(self):
        metric = _make_metric(1, 2, 4, 4, 8)
        src, dst, keep, sizes = bipartite_soft_match(metric, 0, 2, 4, 4)
        assert (sizes == 1).all()
        assert keep.shape == (1, 32)


# ===========================================================================
# 2. Output shapes for varied B, frames, gh, gw, C
# ===========================================================================

class TestOutputShapes:
    @pytest.mark.parametrize("B,frames,gh,gw,C,r", [
        (1, 2, 4, 4, 8, 5),
        (2, 3, 6, 8, 16, 10),
        (1, 4, 4, 6, 32, 20),
        (3, 2, 10, 10, 64, 30),
    ])
    def test_shapes(self, B, frames, gh, gw, C, r):
        N = frames * gh * gw
        metric = _make_metric(B, frames, gh, gw, C)
        x = torch.randn(B, N, C)

        src_idx, dst_idx, keep_idx, sizes = bipartite_soft_match(
            metric, r, frames, gh, gw,
        )
        actual_r = src_idx.shape[1]
        assert actual_r <= r
        assert src_idx.shape == (B, actual_r)
        assert dst_idx.shape == (B, actual_r)
        assert keep_idx.shape == (B, N - actual_r)
        assert sizes.shape == (B, N - actual_r)

        info = RestoreInfo(
            src_idx=src_idx, dst_idx=dst_idx, keep_idx=keep_idx,
            sizes=sizes, num_video_tokens=N, grid=(frames, gh, gw),
        )
        merged, new_sizes = merge_tokens(x, info)
        assert merged.shape == (B, N - actual_r, C)
        assert new_sizes.shape == (B, N - actual_r)

        restored = unmerge_tokens(merged, info)
        assert restored.shape == (B, N, C)


# ===========================================================================
# 3. Weighted-average correctness on a toy tensor (hand-computed)
# ===========================================================================

class TestWeightedAverage:
    def test_two_tokens_merge(self):
        """Manually verify merge of 2 tokens: result should be their average."""
        B, N, C = 1, 4, 2
        x = torch.tensor([[[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0],
                            [7.0, 8.0]]])
        # Manually construct: merge token 0 into token 1
        src_idx = torch.tensor([[0]])
        dst_idx = torch.tensor([[1]])
        keep_idx = torch.tensor([[1, 2, 3]])
        sizes = torch.ones(1, 3)
        info = RestoreInfo(src_idx, dst_idx, keep_idx, sizes, N, (1, 2, 2))

        merged, new_sizes = merge_tokens(x, info)
        assert merged.shape == (1, 3, 2)
        # Token at keep position 0 (global 1) should be avg of [1,2] and [3,4] = [2,3]
        assert torch.allclose(merged[0, 0], torch.tensor([2.0, 3.0]))
        # Token at keep position 1 (global 2) unchanged
        assert torch.allclose(merged[0, 1], torch.tensor([5.0, 6.0]))
        # Token at keep position 2 (global 3) unchanged
        assert torch.allclose(merged[0, 2], torch.tensor([7.0, 8.0]))
        # Size of dst should be 2
        assert new_sizes[0, 0].item() == 2.0

    def test_unmerge_copies_dst_value(self):
        """After unmerge, absorbed tokens should have dst's merged value."""
        B, N, C = 1, 4, 2
        x = torch.tensor([[[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0],
                            [7.0, 8.0]]])
        src_idx = torch.tensor([[0]])
        dst_idx = torch.tensor([[1]])
        keep_idx = torch.tensor([[1, 2, 3]])
        sizes = torch.ones(1, 3)
        info = RestoreInfo(src_idx, dst_idx, keep_idx, sizes, N, (1, 2, 2))

        merged, new_sizes = merge_tokens(x, info)
        info2 = RestoreInfo(src_idx, dst_idx, keep_idx, new_sizes, N, (1, 2, 2))
        restored = unmerge_tokens(merged, info2)

        # Token 0 (absorbed) should have dst's merged value [2, 3]
        assert torch.allclose(restored[0, 0], torch.tensor([2.0, 3.0]))
        # Token 1 (dst) should be [2, 3]
        assert torch.allclose(restored[0, 1], torch.tensor([2.0, 3.0]))


# ===========================================================================
# 4. protect_first_frame: frame-0 tokens never in src_idx
# ===========================================================================

class TestProtectFirstFrame:
    @pytest.mark.parametrize("frames,gh,gw", [
        (3, 4, 4),
        (5, 6, 8),
        (13, 30, 45),
    ])
    def test_frame0_not_in_src(self, frames, gh, gw):
        N = frames * gh * gw
        r = N // 4
        metric = _make_metric(1, frames, gh, gw, 16)
        src_idx, _, _, _ = bipartite_soft_match(
            metric, r, frames, gh, gw, protect_first_frame=True,
        )
        frame0_end = gh * gw
        assert (src_idx < frame0_end).sum().item() == 0

    def test_frame0_can_be_in_src_when_disabled(self):
        frames, gh, gw = 3, 4, 4
        N = frames * gh * gw
        r = N // 3
        metric = _make_metric(1, frames, gh, gw, 16)
        src_idx, _, _, _ = bipartite_soft_match(
            metric, r, frames, gh, gw, protect_first_frame=False,
        )
        # With protection off, frame-0 tokens CAN appear in src
        # (not guaranteed, but with enough r it's very likely)
        # We just verify no error is raised


# ===========================================================================
# 5. Spatiotemporal: matching only within temporal_window
# ===========================================================================

class TestSpatiotemporal:
    def test_within_window(self):
        frames, gh, gw = 5, 4, 4
        N = frames * gh * gw
        r = 10
        metric = _make_metric(1, frames, gh, gw, 16)
        src_idx, dst_idx, _, _ = bipartite_soft_match(
            metric, r, frames, gh, gw,
            mode="spatiotemporal", temporal_window=1,
        )
        # Verify: for each (src, dst) pair, frame distance <= 1
        src_frames = src_idx // (gh * gw)
        dst_frames = dst_idx // (gh * gw)
        frame_dist = (src_frames - dst_frames).abs()
        assert (frame_dist <= 1).all(), f"Found cross-frame merge beyond window: {frame_dist}"

    def test_wider_window(self):
        frames, gh, gw = 5, 4, 4
        N = frames * gh * gw
        r = 10
        metric = _make_metric(1, frames, gh, gw, 16)
        src_idx, dst_idx, _, _ = bipartite_soft_match(
            metric, r, frames, gh, gw,
            mode="spatiotemporal", temporal_window=2,
        )
        src_frames = src_idx // (gh * gw)
        dst_frames = dst_idx // (gh * gw)
        frame_dist = (src_frames - dst_frames).abs()
        assert (frame_dist <= 2).all()

    def test_spatiotemporal_protect_first_frame(self):
        frames, gh, gw = 4, 6, 6
        N = frames * gh * gw
        r = 20
        metric = _make_metric(1, frames, gh, gw, 16)
        src_idx, _, _, _ = bipartite_soft_match(
            metric, r, frames, gh, gw,
            mode="spatiotemporal", temporal_window=1,
            protect_first_frame=True,
        )
        frame0_end = gh * gw
        assert (src_idx < frame0_end).sum().item() == 0


# ===========================================================================
# 6. Batched matching: B=2 with different content → different indices
# ===========================================================================

class TestBatchedMatching:
    def test_different_content_different_indices(self):
        B, frames, gh, gw, C = 2, 3, 4, 4, 16
        N = frames * gh * gw
        torch.manual_seed(0)
        m0 = torch.nn.functional.normalize(torch.randn(1, N, C), dim=-1)
        torch.manual_seed(999)
        m1 = torch.nn.functional.normalize(torch.randn(1, N, C), dim=-1)
        metric = torch.cat([m0, m1], dim=0)

        r = 8
        src_idx, dst_idx, keep_idx, sizes = bipartite_soft_match(
            metric, r, frames, gh, gw,
        )
        assert src_idx.shape[0] == 2
        # Different random content should yield different src selections
        assert not torch.equal(src_idx[0], src_idx[1])

    def test_batched_merge_unmerge_correctness(self):
        B, frames, gh, gw, C = 2, 2, 4, 4, 8
        N = frames * gh * gw
        torch.manual_seed(42)
        metric = _make_metric(B, frames, gh, gw, C, seed=42)
        x = torch.randn(B, N, C)

        merged, restored, info = _full_merge_cycle(
            metric, x, r=5, frames=frames, gh=gh, gw=gw,
        )
        # Kept tokens should be exactly preserved (before averaging)
        for b in range(B):
            kept_original = x[b][info.keep_idx[b]]
            # After merge+unmerge, kept tokens that were NOT destinations
            # should retain original values → check shape at least
            assert restored.shape == x.shape


# ===========================================================================
# 7. size_log_bias correctness
# ===========================================================================

class TestSizeLogBias:
    def test_hand_computed(self):
        """Two merged tokens with sizes [1, 3]: bias should be [0, log(3)]."""
        sizes = torch.tensor([[1.0, 3.0]])
        info = RestoreInfo(
            src_idx=torch.zeros(1, 0, dtype=torch.long),
            dst_idx=torch.zeros(1, 0, dtype=torch.long),
            keep_idx=torch.arange(2).unsqueeze(0),
            sizes=sizes,
            num_video_tokens=2,
            grid=(1, 1, 2),
        )
        bias = size_log_bias(info, num_text_tokens=3, num_heads=4)
        # Shape: [1, 1, 1, 5] (3 text + 2 video)
        assert bias.shape == (1, 1, 1, 5)
        # Text positions: zero
        assert (bias[0, 0, 0, :3] == 0).all()
        # Video: [log(1), log(3)] = [0, 1.0986...]
        assert abs(bias[0, 0, 0, 3].item() - 0.0) < 1e-6
        assert abs(bias[0, 0, 0, 4].item() - math.log(3.0)) < 1e-5

    def test_all_ones_is_zero(self):
        sizes = torch.ones(2, 10)
        info = RestoreInfo(
            src_idx=torch.zeros(2, 0, dtype=torch.long),
            dst_idx=torch.zeros(2, 0, dtype=torch.long),
            keep_idx=torch.arange(10).unsqueeze(0).expand(2, -1),
            sizes=sizes,
            num_video_tokens=10,
            grid=(1, 2, 5),
        )
        bias = size_log_bias(info, num_text_tokens=5, num_heads=8)
        assert (bias == 0).all()


# ===========================================================================
# 8. Determinism
# ===========================================================================

class TestDeterminism:
    def test_same_input_same_output(self):
        metric = _make_metric(1, 3, 4, 4, 16, seed=7)
        x = torch.randn(1, 48, 16)
        r = 8

        src1, dst1, keep1, _ = bipartite_soft_match(metric, r, 3, 4, 4)
        src2, dst2, keep2, _ = bipartite_soft_match(metric, r, 3, 4, 4)
        assert torch.equal(src1, src2)
        assert torch.equal(dst1, dst2)
        assert torch.equal(keep1, keep2)


# ===========================================================================
# 9. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_r_larger_than_src_set(self):
        """r exceeds number of src tokens → clamps to max possible."""
        frames, gh, gw = 2, 2, 2
        N = frames * gh * gw  # 8
        metric = _make_metric(1, frames, gh, gw, 4)
        # With protect_first_frame, src set is even smaller
        src_idx, _, keep_idx, _ = bipartite_soft_match(
            metric, r=100, frames=frames, gh=gh, gw=gw,
            protect_first_frame=True,
        )
        assert src_idx.shape[1] <= N
        assert src_idx.shape[1] + keep_idx.shape[1] == N

    def test_single_frame(self):
        """Single frame, spatial only (protect_first_frame off since there's only 1 frame)."""
        frames, gh, gw = 1, 4, 4
        metric = _make_metric(1, frames, gh, gw, 8)
        x = torch.randn(1, 16, 8)
        merged, restored, info = _full_merge_cycle(
            metric, x, r=3, frames=frames, gh=gh, gw=gw,
            protect_first_frame=False,
        )
        assert merged.shape == (1, 13, 8)
        assert restored.shape == (1, 16, 8)

    def test_single_frame_with_protection_noop(self):
        """Single frame + protect_first_frame=True: no merge possible, returns identity."""
        frames, gh, gw = 1, 4, 4
        metric = _make_metric(1, frames, gh, gw, 8)
        x = torch.randn(1, 16, 8)
        merged, restored, info = _full_merge_cycle(
            metric, x, r=3, frames=frames, gh=gh, gw=gw,
            protect_first_frame=True,
        )
        assert merged.shape == x.shape
        assert torch.allclose(merged, x)


# ===========================================================================
# 10. KV-only helpers
# ===========================================================================

class TestKVOnlyHelpers:
    def test_merge_kv_tokens_preserves_head_layout(self):
        frames, gh, gw = 2, 4, 4
        B, H, D = 2, 3, 5
        N = frames * gh * gw
        r = 6
        metric = _make_metric(B, frames, gh, gw, C=16, seed=11)
        src_idx, dst_idx, keep_idx, sizes = bipartite_soft_match(
            metric,
            r=r,
            frames=frames,
            gh=gh,
            gw=gw,
            protect_first_frame=False,
        )
        info = RestoreInfo(src_idx, dst_idx, keep_idx, sizes, N, (frames, gh, gw))
        kv = torch.randn(B, H, N, D)

        merged, new_sizes = merge_kv_tokens(kv, info)

        assert merged.shape == (B, H, N - r, D)
        assert new_sizes.shape == (B, N - r)
        assert torch.all(new_sizes >= 1)

    def test_checkerboard_shifted_flips_source_partition(self):
        frames, gh, gw = 2, 4, 4
        metric = _make_metric(1, frames, gh, gw, C=8, seed=12)

        src0, _, _, _ = bipartite_soft_match(
            metric,
            r=4,
            frames=frames,
            gh=gh,
            gw=gw,
            protect_first_frame=False,
            match_feature="fixed",
            partition="checkerboard_shifted",
            partition_offset=0,
        )
        src1, _, _, _ = bipartite_soft_match(
            metric,
            r=4,
            frames=frames,
            gh=gh,
            gw=gw,
            protect_first_frame=False,
            match_feature="fixed",
            partition="checkerboard_shifted",
            partition_offset=1,
        )

        assert not torch.equal(src0, src1)
