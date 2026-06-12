from __future__ import annotations

from .cogvideox_processor import CogVideoXRnRAttnProcessor2_0
from .rnr_config import RnRConfig
from .runtime import RnRRuntime


def apply_rnr_to_cogvideox(transformer, config: RnRConfig) -> RnRRuntime:
    """Attach RnR processors and runtime state to a CogVideoX transformer."""

    runtime = RnRRuntime(config=config, num_layers=len(transformer.transformer_blocks))
    transformer._merge_cfg = None
    transformer._rnr_runtime = runtime
    for idx, block in enumerate(transformer.transformer_blocks):
        block._merge_cfg = None
        block._rnr_runtime = runtime
        block._rnr_block_index = idx
        if not hasattr(block, "_rnr_original_processor"):
            block._rnr_original_processor = block.attn1.processor
        block.attn1.set_processor(CogVideoXRnRAttnProcessor2_0())
    return runtime


def detach_rnr_from_cogvideox(transformer) -> None:
    transformer._rnr_runtime = None
    for block in transformer.transformer_blocks:
        block._rnr_runtime = None
        if hasattr(block, "_rnr_block_index"):
            delattr(block, "_rnr_block_index")
        if hasattr(block, "_rnr_original_processor"):
            block.attn1.set_processor(block._rnr_original_processor)
            delattr(block, "_rnr_original_processor")
