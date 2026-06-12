from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokmerge.rnr.cogvideox_processor import CogVideoXRnRAttnProcessor2_0
from tokmerge.rnr.matching import compute_similarity, match_features
from tokmerge.rnr.partition import VisualLayout, source_destination_indices
from tokmerge.rnr.reduce_restore import build_reduction_plan, reduce_sequence, restore_sequence
from tokmerge.rnr.rnr_config import RnRConfig
from tokmerge.rnr.runtime import RnRRuntime


def test_similarity_shapes_and_euclidean_order():
    dst = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    src = torch.tensor([[[0.1, 0.0], [1.8, 0.0]]])

    sim = compute_similarity(dst, src, "euclidean")
    assert sim.shape == (1, 2, 2)
    result = match_features(dst, src, "euclidean")
    assert result.dst_for_src.tolist() == [[0, 1]]


def test_partition_protects_first_frame_sources():
    layout = VisualLayout(frames=3, height=4, width=4)
    src_idx, dst_idx = source_destination_indices(layout, (2, 2, 2), torch.device("cpu"))
    assert src_idx.numel() > 0
    assert dst_idx.numel() > 0
    assert (src_idx >= 16).all()


def test_reduce_restore_replace_3d_and_4d_shapes():
    torch.manual_seed(0)
    layout = VisualLayout(frames=2, height=4, width=4)
    x = torch.randn(2, layout.num_tokens, 8)
    plan = build_reduction_plan(x, layout, ratio=0.25, dst_stride=(1, 2, 2), similarity_type="cosine")

    reduced = reduce_sequence(x, plan)
    restored = restore_sequence(reduced, plan)
    assert reduced.shape[0] == x.shape[0]
    assert reduced.shape[1] == layout.num_tokens - plan.removed_tokens
    assert restored.shape == x.shape

    heads = torch.randn(2, 3, layout.num_tokens, 5)
    reduced_heads = reduce_sequence(heads, plan)
    restored_heads = restore_sequence(reduced_heads, plan)
    assert reduced_heads.shape == (2, 3, layout.num_tokens - plan.removed_tokens, 5)
    assert restored_heads.shape == heads.shape


def test_mean_reduction_updates_sizes():
    torch.manual_seed(1)
    layout = VisualLayout(frames=2, height=4, width=4)
    x = torch.randn(1, layout.num_tokens, 4)
    plan = build_reduction_plan(
        x,
        layout,
        ratio=0.25,
        dst_stride=(1, 2, 2),
        similarity_type="dot",
        reduce_mode="mean",
    )
    assert plan.sizes.max().item() > 1.0
    reduced = reduce_sequence(x, plan, "mean")
    assert reduced.shape[1] == layout.num_tokens - plan.removed_tokens


class _FakeAttn(torch.nn.Module):
    def __init__(self, dim: int = 12, heads: int = 3):
        super().__init__()
        self.heads = heads
        self.is_cross_attention = False
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False), torch.nn.Identity()])
        self.norm_q = None
        self.norm_k = None

    def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):  # pragma: no cover
        return attention_mask


def _run_processor(method: str):
    torch.manual_seed(2)
    layout = VisualLayout(frames=2, height=4, width=4)
    cfg = RnRConfig(
        method=method,
        q_reduce_ratio=0.25,
        kv_reduce_ratio=0.25,
        dst_stride=(1, 2, 2),
        similarity_type="cosine",
        layers="all",
    )
    runtime = RnRRuntime(cfg, num_layers=1)
    runtime.observe_transformer_call(torch.tensor([500.0]), layout)

    processor = CogVideoXRnRAttnProcessor2_0()
    processor._rnr_runtime = runtime
    processor._rnr_block_index = 0
    attn = _FakeAttn()
    hidden = torch.randn(2, layout.num_tokens, 12)
    text = torch.randn(2, 5, 12)
    out_hidden, out_text = processor(attn, hidden, text)
    return runtime, out_hidden, out_text, hidden, text


def test_processor_kv_rnr_preserves_shapes():
    runtime, out_hidden, out_text, hidden, text = _run_processor("kv_rnr")
    assert out_hidden.shape == hidden.shape
    assert out_text.shape == text.shape
    assert runtime.stats.kv_tokens_after < runtime.stats.kv_tokens_before
    assert runtime.stats.q_tokens_after == runtime.stats.q_tokens_before


def test_processor_qv_rnr_restores_query_output_shape():
    runtime, out_hidden, out_text, hidden, text = _run_processor("qv_rnr")
    assert out_hidden.shape == hidden.shape
    assert out_text.shape == text.shape
    assert runtime.stats.q_tokens_after < runtime.stats.q_tokens_before


def test_processor_rnr_tome_cache_hits():
    runtime, out_hidden, out_text, hidden, text = _run_processor("rnr_tome")
    assert out_hidden.shape == hidden.shape
    assert out_text.shape == text.shape

    processor = CogVideoXRnRAttnProcessor2_0()
    processor._rnr_runtime = runtime
    processor._rnr_block_index = 0
    attn = _FakeAttn()
    processor(attn, hidden, text)
    assert runtime.stats.cache_hits > 0
