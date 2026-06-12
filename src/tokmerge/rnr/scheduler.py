from __future__ import annotations

from .rnr_config import RnRConfig


LAYER_STRATEGIES = {
    "middle": lambda n: tuple(range(n // 3, 2 * n // 3)),
    "middle_wide": lambda n: tuple(range(n // 5, 4 * n // 5)),
    "late_off": lambda n: tuple(range(0, 4 * n // 5)),
    "all": lambda n: tuple(range(n)),
}


def resolve_layers(layers: str | tuple[int, ...], num_layers: int) -> tuple[int, ...]:
    if isinstance(layers, tuple):
        return tuple(int(i) for i in layers)
    if isinstance(layers, str) and layers in LAYER_STRATEGIES:
        return LAYER_STRATEGIES[layers](num_layers)
    raise ValueError(f"Unknown RnR layer strategy: {layers!r}")


class RnRScheduler:
    """AsymRnR-lite feature/block/timestep schedule.

    This implementation uses explicit ratios from the CLI/config and applies
    block and timestep gates. Similarity-threshold schedules are kept in config
    for experiment metadata, but they are not auto-estimated without a prior
    similarity distribution.
    """

    def __init__(self, config: RnRConfig, num_layers: int) -> None:
        self.config = config
        self.layers = resolve_layers(config.layers, num_layers)
        self.num_layers = num_layers

    def block_enabled(self, block_index: int) -> bool:
        if block_index not in self.layers:
            return False
        first_n = int(self.config.block_skip.get("first_n", 0))
        last_n = int(self.config.block_skip.get("last_n", 0))
        if block_index < first_n:
            return False
        if last_n > 0 and block_index >= self.num_layers - last_n:
            return False
        return True

    def timestep_enabled(self, progress: float) -> bool:
        if self.config.disable_schedule:
            return True
        first = float(self.config.step_skip.get("first_frac", 0.0))
        last = float(self.config.step_skip.get("last_frac", 0.0))
        if first > 0 and progress < first:
            return False
        if last > 0 and progress > 1.0 - last:
            return False
        return True

    def _fixed_ratio_for(self, feature: str) -> float:
        method = self.config.method
        if feature == "h":
            return max(0.0, float(self.config.h_reduce_ratio))
        if feature == "q":
            if method == "kv_rnr":
                return 0.0
            return max(0.0, float(self.config.q_reduce_ratio))
        if feature in ("kv", "k", "v"):
            if method in ("kv_rnr", "qv_rnr", "rnr_tome"):
                return max(0.0, float(self.config.kv_reduce_ratio))
            return 0.0
        raise ValueError(f"Unknown feature: {feature!r}")

    def _schedule_key_for(self, feature: str) -> str:
        return "v" if feature in ("kv", "k") else feature

    def ratio_from_tier(self, feature: str, tier: float) -> float:
        schedule = self.config.schedule or {}
        feature_schedule = schedule.get(self._schedule_key_for(feature), {})
        if not isinstance(feature_schedule, dict):
            return self._fixed_ratio_for(feature)

        ratio = 0.0
        for threshold, candidate in feature_schedule.items():
            if str(threshold) in ("enabled", "thresholds", "share_with_v"):
                continue
            try:
                threshold_value = float(threshold)
                candidate_ratio = float(candidate)
            except (TypeError, ValueError):
                continue
            if tier >= threshold_value:
                ratio = max(ratio, candidate_ratio)
        return ratio

    def ratio_for(self, feature: str, block_index: int, progress: float, tier: float | None = None) -> float:
        if not self.block_enabled(block_index) or not self.timestep_enabled(progress):
            return 0.0

        if tier is not None:
            return max(0.0, self.ratio_from_tier(feature, tier))
        return self._fixed_ratio_for(feature)
