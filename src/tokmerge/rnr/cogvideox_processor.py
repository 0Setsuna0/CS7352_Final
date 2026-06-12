from __future__ import annotations

import torch
import torch.nn.functional as F

from .reduce_restore import key_size_log_bias, reduce_sequence, restore_sequence


def _select_rotary_emb(image_rotary_emb, keep_idx: torch.Tensor):
    if image_rotary_emb is None:
        return None
    if keep_idx.ndim != 2:
        raise ValueError(f"Expected keep_idx [B,N], got {tuple(keep_idx.shape)}")
    if keep_idx.shape[0] > 1 and not (keep_idx == keep_idx[:1]).all():
        raise ValueError("Hidden-state RnR with per-batch RoPE indices is not supported yet.")

    idx = keep_idx[0].to("cpu")
    if isinstance(image_rotary_emb, tuple):
        selected = []
        for rope in image_rotary_emb:
            rope_idx = idx.to(rope.device)
            if rope.ndim == 2:
                selected.append(rope.index_select(0, rope_idx))
            elif rope.ndim == 3:
                selected.append(rope.index_select(1, rope_idx))
            else:
                raise ValueError(f"Unsupported rotary embedding shape: {tuple(rope.shape)}")
        return tuple(selected)
    rope_idx = idx.to(image_rotary_emb.device)
    if image_rotary_emb.ndim == 2:
        return image_rotary_emb.index_select(0, rope_idx)
    if image_rotary_emb.ndim == 3:
        return image_rotary_emb.index_select(1, rope_idx)
    raise ValueError(f"Unsupported rotary embedding shape: {tuple(image_rotary_emb.shape)}")


class CogVideoXRnRAttnProcessor2_0:
    """CogVideoX attention processor with asymmetric RnR token reduction."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXRnRAttnProcessor2_0 requires PyTorch 2.0 or newer.")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        runtime = getattr(self, "_rnr_runtime", None)
        block_index = getattr(self, "_rnr_block_index", None)

        h_plan = None
        if runtime is not None and block_index is not None and runtime.block_enabled(block_index):
            h_ratio = runtime.ratio_for("h", block_index)
            if h_ratio > 0:
                h_plan = runtime.reduction_plan("h", block_index, hidden_states, h_ratio)
                hidden_states = reduce_sequence(hidden_states, h_plan, runtime.config.reduce_mode)
            runtime.record_tokens("h", h_plan.num_tokens if h_plan is not None else hidden_states.shape[1], hidden_states.shape[1])

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            video_rotary_emb = _select_rotary_emb(image_rotary_emb, h_plan.keep_idx) if h_plan is not None else image_rotary_emb
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], video_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], video_rotary_emb)

        q_plan = None
        kv_plan = None
        if runtime is not None and block_index is not None and runtime.block_enabled(block_index):
            if attention_mask is not None:
                raise ValueError("RnR attention reduction does not support attention_mask yet.")

            q_text, q_video = query[:, :, :text_seq_length], query[:, :, text_seq_length:]
            k_text, k_video = key[:, :, :text_seq_length], key[:, :, text_seq_length:]
            v_text, v_video = value[:, :, :text_seq_length], value[:, :, text_seq_length:]

            q_ratio = runtime.ratio_for("q", block_index)
            if q_ratio > 0:
                q_plan = runtime.reduction_plan("q", block_index, q_video, q_ratio)
                q_video = reduce_sequence(q_video, q_plan, runtime.config.reduce_mode)
            runtime.record_tokens("q", query.shape[2] - text_seq_length, q_video.shape[2])

            kv_ratio = runtime.ratio_for("kv", block_index)
            if kv_ratio > 0:
                kv_plan = runtime.reduction_plan("kv", block_index, v_video, kv_ratio)
                k_video = reduce_sequence(k_video, kv_plan, runtime.config.reduce_mode)
                v_video = reduce_sequence(v_video, kv_plan, runtime.config.reduce_mode)
            runtime.record_tokens("kv", key.shape[2] - text_seq_length, k_video.shape[2])

            query = torch.cat([q_text, q_video], dim=2)
            key = torch.cat([k_text, k_video], dim=2)
            value = torch.cat([v_text, v_video], dim=2)

        attn_bias = attention_mask
        if kv_plan is not None and runtime is not None and runtime.config.prop_attn:
            bias = key_size_log_bias(kv_plan, text_seq_length, attn.heads).to(query.device, query.dtype)
            attn_bias = bias if attn_bias is None else attn_bias + bias

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )

        if q_plan is not None:
            text_output, video_output = attn_output[:, :, :text_seq_length], attn_output[:, :, text_seq_length:]
            video_output = restore_sequence(video_output, q_plan)
            attn_output = torch.cat([text_output, video_output], dim=2)

        hidden_states = attn_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        if h_plan is not None:
            hidden_states = restore_sequence(hidden_states, h_plan)
        return hidden_states, encoder_hidden_states
