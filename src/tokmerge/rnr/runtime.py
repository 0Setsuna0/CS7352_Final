from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import torch

from .partition import VisualLayout
from .reduce_restore import ReductionPlan, build_reduction_plan
from .rnr_config import RnRConfig
from .scheduler import RnRScheduler


@dataclass
class RnRStats:
    matching_calls: int = 0
    cache_hits: int = 0
    matching_seconds: float = 0.0
    q_tokens_before: int = 0
    q_tokens_after: int = 0
    kv_tokens_before: int = 0
    kv_tokens_after: int = 0
    h_tokens_before: int = 0
    h_tokens_after: int = 0

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        denom = max(1, self.matching_calls)
        data["cache_hit_rate"] = self.cache_hits / denom
        q_denom = max(1, self.q_tokens_before)
        kv_denom = max(1, self.kv_tokens_before)
        h_denom = max(1, self.h_tokens_before)
        data["q_reduction_ratio"] = 1.0 - self.q_tokens_after / q_denom
        data["kv_reduction_ratio"] = 1.0 - self.kv_tokens_after / kv_denom
        data["h_reduction_ratio"] = 1.0 - self.h_tokens_after / h_denom
        return data


@dataclass
class _CacheEntry:
    step_index: int
    plan: ReductionPlan


@dataclass
class RnRRuntime:
    config: RnRConfig
    num_layers: int
    scheduler: RnRScheduler = field(init=False)
    enabled: bool = True
    step_index: int = -1
    timestep_value: float = 1000.0
    progress: float = 0.0
    current_layout: VisualLayout | None = None
    stats: RnRStats = field(default_factory=RnRStats)
    _cache: dict[tuple, _CacheEntry] = field(default_factory=dict)
    _schedule_tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scheduler = RnRScheduler(self.config, self.num_layers)
        self._schedule_tensors = self._load_schedule_tensors()

    def _load_schedule_tensors(self) -> dict[str, torch.Tensor]:
        if not self.config.schedule_file:
            return {}

        schedule_path = Path(self.config.schedule_file)
        if not schedule_path.exists() and self.config.schedule_url:
            schedule_path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(self.config.schedule_url, schedule_path)

        if not schedule_path.exists():
            raise FileNotFoundError(
                f"RnR schedule file not found: {schedule_path}. "
                "Set schedule_url or use a fixed-ratio config without schedule_file."
            )

        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:
            raise RuntimeError("RnR schedule_file requires the safetensors package.") from exc

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(schedule_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if tensor.ndim != 2:
                    raise ValueError(f"Schedule tensor {key!r} must be [num_blocks, num_steps], got {tuple(tensor.shape)}")
                if tensor.shape[0] != self.num_layers:
                    raise ValueError(
                        f"Schedule tensor {key!r} has {tensor.shape[0]} blocks, expected {self.num_layers}."
                    )
                tensors[key] = tensor
        return tensors

    def reset_prompt(self) -> None:
        self.step_index = -1
        self.timestep_value = 1000.0
        self.progress = 0.0
        self.current_layout = None
        self.stats = RnRStats()
        self._cache.clear()

    def observe_transformer_call(self, timestep, layout: VisualLayout) -> None:
        if isinstance(timestep, torch.Tensor):
            value = float(timestep.float().mean().item())
        else:
            value = float(timestep)
        self.step_index += 1
        self.timestep_value = value
        self.progress = max(0.0, min(1.0, 1.0 - value / 1000.0))
        self.current_layout = layout

    def block_enabled(self, block_index: int) -> bool:
        return self.enabled and self.scheduler.block_enabled(block_index)

    def _schedule_feature(self, feature: str) -> str:
        return "v" if feature in ("kv", "k") else feature

    def _schedule_step_index(self, schedule_steps: int) -> int:
        if schedule_steps <= 1:
            return 0
        return max(0, min(schedule_steps - 1, int(round(self.progress * (schedule_steps - 1)))))

    def schedule_tier_for(self, feature: str, block_index: int) -> float | None:
        schedule_feature = self._schedule_feature(feature)
        tensor = self._schedule_tensors.get(schedule_feature)
        if tensor is None:
            return None
        step_idx = self._schedule_step_index(tensor.shape[1])
        return float(tensor[block_index, step_idx].item())

    def ratio_for(self, feature: str, block_index: int) -> float:
        tier = self.schedule_tier_for(feature, block_index)
        return self.scheduler.ratio_for(feature, block_index, self.progress, tier=tier)

    def _partition_generator(self, feature: str, block_index: int, ratio: float, device: torch.device):
        base_seed = getattr(self.config, "partition_seed", None)
        if base_seed is None:
            return None
        cache_steps = max(1, int(self.config.matching_cache_steps))
        cache_window = max(0, self.step_index // cache_steps)
        feature_offset = {"h": 11, "q": 23, "kv": 37, "v": 37, "k": 37}.get(feature, 0)
        seed = int(base_seed) + block_index * 1009 + cache_window * 9176 + feature_offset + int(ratio * 10000)
        try:
            generator = torch.Generator(device=device)
        except (TypeError, RuntimeError):
            if device.type != "cpu":
                return None
            generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def reduction_plan(
        self,
        feature: str,
        block_index: int,
        features: torch.Tensor,
        ratio: float,
    ) -> ReductionPlan:
        if self.current_layout is None:
            raise RuntimeError("RnRRuntime.current_layout is not set")

        layout = self.current_layout
        cache_key = (
            feature,
            block_index,
            tuple(features.shape),
            layout.frames,
            layout.height,
            layout.width,
            round(float(ratio), 6),
            tuple(self.config.dst_stride),
            self.config.partition_mode,
            self.config.similarity_type,
            self.config.reduce_mode,
            self.config.protect_first_frame,
            round(float(self.config.protect_topk_ratio), 6),
            str(features.device),
        )
        self.stats.matching_calls += 1
        entry = self._cache.get(cache_key)
        if entry is not None and self.step_index - entry.step_index < max(1, self.config.matching_cache_steps):
            self.stats.cache_hits += 1
            return entry.plan

        start = time.perf_counter()
        plan = build_reduction_plan(
            features=features,
            layout=layout,
            ratio=ratio,
            dst_stride=self.config.dst_stride,
            similarity_type=self.config.similarity_type,
            reduce_mode=self.config.reduce_mode,
            protect_first_frame=self.config.protect_first_frame,
            protect_topk_ratio=self.config.protect_topk_ratio,
            cfg_consistent=self.config.cfg_consistent,
            partition_mode=self.config.partition_mode,
            generator=self._partition_generator(feature, block_index, ratio, features.device),
            partition_cache_key=(feature, block_index, self.step_index // max(1, self.config.matching_cache_steps)),
        )
        self.stats.matching_seconds += time.perf_counter() - start
        self._cache[cache_key] = _CacheEntry(self.step_index, plan)
        return plan

    def record_tokens(self, feature: str, before: int, after: int) -> None:
        if feature == "q":
            self.stats.q_tokens_before += before
            self.stats.q_tokens_after += after
        elif feature == "kv":
            self.stats.kv_tokens_before += before
            self.stats.kv_tokens_after += after
        elif feature == "h":
            self.stats.h_tokens_before += before
            self.stats.h_tokens_after += after

    def stats_dict(self) -> dict[str, float | int]:
        data = self.stats.to_dict()
        data.update(
            {
                "method": self.config.method,
                "similarity_type": self.config.similarity_type,
                "reduce_mode": self.config.reduce_mode,
                "matching_cache_steps": self.config.matching_cache_steps,
                "schedule_file": self.config.schedule_file,
                "schedule_loaded": bool(self._schedule_tensors),
            }
        )
        return data
