from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RnRConfig:
    method: str = "rnr_tome"
    model: str = "cogvideox-2b"
    encoder_first: bool = True
    similarity_type: str = "euclidean"
    reduce_mode: str = "replace"
    dst_stride: tuple[int, int, int] = (2, 2, 2)
    matching_cache_steps: int = 5
    schedule_file: str | None = None
    schedule_url: str | None = None
    q_reduce_ratio: float = 0.4
    kv_reduce_ratio: float = 0.2
    h_reduce_ratio: float = 0.0
    partition_mode: str = "random_chunk"
    partition_seed: int | None = 123
    layers: str | tuple[int, ...] = "middle_wide"
    protect_first_frame: bool = True
    protect_topk_ratio: float = 0.0
    cfg_consistent: bool = True
    prop_attn: bool = False
    disable_schedule: bool = False
    schedule: dict[str, Any] = field(default_factory=dict)
    block_skip: dict[str, int] = field(default_factory=lambda: {"first_n": 0, "last_n": 0})
    step_skip: dict[str, float] = field(default_factory=lambda: {"first_frac": 0.0, "last_frac": 0.0})


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{path} is not JSON-compatible YAML and PyYAML is not installed. "
            "Use the provided JSON-style YAML configs or install pyyaml."
        ) from exc


def _to_stride(value: Any) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(v) for v in value)
    raise ValueError(f"dst_stride must be an int or length-3 list, got {value!r}")


def load_rnr_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> RnRConfig:
    raw: dict[str, Any] = {}
    config_path = Path(path) if path is not None else None
    if path is not None:
        raw.update(_load_mapping(config_path))
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    if "dst_stride" in raw:
        raw["dst_stride"] = _to_stride(raw["dst_stride"])
    if isinstance(raw.get("layers"), list):
        raw["layers"] = tuple(int(v) for v in raw["layers"])
    if raw.get("schedule_file") and config_path is not None:
        schedule_path = Path(raw["schedule_file"])
        if not schedule_path.is_absolute():
            raw["schedule_file"] = str((config_path.parent / schedule_path).resolve())

    allowed = set(RnRConfig.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return RnRConfig(**filtered)
