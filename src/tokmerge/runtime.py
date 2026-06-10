"""Glue between MergeConfig and the vendored diffusers transformer model."""
from __future__ import annotations

import json
from pathlib import Path

from .merging import MergeConfig


LAYER_STRATEGIES = {
    "middle": lambda n: tuple(range(n // 3, 2 * n // 3)),
    "middle_wide": lambda n: tuple(range(n // 5, 4 * n // 5)),
    "late_off": lambda n: tuple(range(0, 4 * n // 5)),
    "all": lambda n: tuple(range(n)),
}


def resolve_layers(name: str | tuple | list, num_layers: int) -> tuple[int, ...]:
    """Convert a layer strategy name or explicit list to a tuple of block indices."""
    if isinstance(name, (list, tuple)):
        return tuple(int(i) for i in name)
    if isinstance(name, str) and name in LAYER_STRATEGIES:
        return LAYER_STRATEGIES[name](num_layers)
    raise ValueError(f"Unknown layer strategy: {name!r}. Choose from {list(LAYER_STRATEGIES)}")


def load_merge_config(path: str | Path) -> MergeConfig:
    """Load a JSON config and return a MergeConfig."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = raw.pop("layers", ())
    cfg = MergeConfig(**raw)
    # layers is resolved later when num_layers is known
    cfg._raw_layers = layers
    return cfg


def attach_merge_config(transformer, cfg: MergeConfig) -> None:
    """Attach merge config to transformer and set per-block block_index.

    Idempotent; safe to call each run.
    """
    num_layers = len(transformer.transformer_blocks)

    if hasattr(cfg, "_raw_layers"):
        cfg.layers = resolve_layers(cfg._raw_layers, num_layers)

    transformer._merge_cfg = cfg

    for idx, block in enumerate(transformer.transformer_blocks):
        block._block_index = idx
        block._merge_cfg = cfg


def detach_merge_config(transformer) -> None:
    """Remove merge config -> exact baseline behavior."""
    transformer._merge_cfg = None
    for block in transformer.transformer_blocks:
        block._merge_cfg = None
        if hasattr(block, "_block_index"):
            del block._block_index
