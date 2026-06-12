r"""Standalone high-quality CogVideoX test with a single RnR toggle.

Normal usage:
  1. Edit ENABLE_TOKEN_MERGE, PROMPT_INDEX, and RNR_CONFIG_INDEX below.
  2. Run:
       .\.venv\Scripts\python.exe .\scripts\run_quality_rnr_test.py

When ENABLE_TOKEN_MERGE is True, this script enables the new SA-RnR-ToMe
attention-level reduction/restoration path. When False, it runs the original
CogVideoX baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import diffusers as diffusers_pkg
import torch
from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download


# User-facing switches.
ENABLE_TOKEN_MERGE = False
PROMPT_INDEX = 7
RNR_CONFIG_INDEX = 6


_MODEL_ID = "THUDM/CogVideoX-2b"
_SEED = 123
_HEIGHT = 480
_WIDTH = 720
_NUM_FRAMES = 49
_NUM_INFERENCE_STEPS = 40
_GUIDANCE_SCALE = 5.5
_FPS = 8
_DTYPE_NAME = "float16"
_OFFLOAD_MODE = "model"
_DEVICE = "cuda"
_DOWNLOAD_WORKERS = 1
_LOCAL_FILES_ONLY = False

RNR_CONFIG_PRESETS = [
    {
        "name": "quality_safe",
        "path": "configs/rnr/rnr_quality_safe.yaml",
    },
    {
        "name": "conservative",
        "path": "configs/rnr/rnr_conservative.yaml",
    },
    {
        "name": "balanced",
        "path": "configs/rnr/rnr_balanced.yaml",
    },
    {
        "name": "current_default",
        "path": "configs/rnr/rnr_current_default.yaml",
    },
    {
        "name": "aggressive",
        "path": "configs/rnr/rnr_aggressive.yaml",
    },
    {
        "name": "max_speed",
        "path": "configs/rnr/rnr_max_speed.yaml",
    },
    {
        "name": "official_base",
        "path": "configs/rnr/rnr_official_base.yaml",
    },
    {
        "name": "official_fast",
        "path": "configs/rnr/rnr_official_fast.yaml",
    },
]

QUALITY_PROMPTS = [
    {
        "name": "panda_studio_guitar",
        "prompt": (
            "A panda wearing a simple red scarf sits on a plain wooden stool in a clean studio with a soft warm gray backdrop "
            "and plays an acoustic guitar. The right paw strums the guitar in a steady rhythm, the left paw changes chords, "
            "and the panda gently nods its head to the music. The camera makes a slow smooth push-in, with soft studio lighting, "
            "a clear centered subject, minimal background detail, stable motion, realistic fur, crisp guitar shape, natural colors, "
            "high quality video."
        ),
    },
    {
        "name": "ceramic_cup_steam",
        "prompt": (
            "A white ceramic cup sits on a plain dark wooden table against a smooth warm gray wall. Thin steam rises from the cup "
            "in slow curling ribbons, moving naturally upward while the cup remains still. The camera is locked off with shallow depth "
            "of field, soft side lighting, minimal background detail, clear cup shape, realistic steam, high quality video."
        ),
    },
    {
        "name": "red_ball_roll",
        "prompt": (
            "A glossy red ball rolls slowly from left to right across a clean light gray studio floor, casting a soft shadow under it. "
            "The background is a plain gray wall with no objects. The camera tracks the ball smoothly at low height, simple composition, "
            "stable motion, clean reflections, natural lighting, high quality video."
        ),
    },
    {
        "name": "fox_turntable",
        "prompt": (
            "A small orange fox figurine stands on a simple black rotating turntable in a clean studio with a pale green backdrop. "
            "The turntable rotates slowly, showing the figurine from the front, side, and back while the camera remains still. "
            "Soft studio lighting, centered subject, minimal background detail, crisp edges, stable rotation, high quality video."
        ),
    },
    {
        "name": "paper_airplane",
        "prompt": (
            "A white paper airplane glides gently across an empty studio space with a smooth light blue backdrop and a pale floor. "
            "It enters from the left, floats in a shallow arc, and exits to the right while casting a faint moving shadow. "
            "The camera pans slowly to follow it, clean composition, minimal background, stable motion, high quality video."
        ),
    },
    {
        "name": "robot_wave_studio",
        "prompt": (
            "A small friendly silver robot stands alone on a matte white floor against a smooth pale blue studio backdrop. "
            "The robot slowly raises one arm and waves three times, then tilts its head with a gentle mechanical smile. "
            "The camera stays centered with a slow dolly-in, soft even lighting, simple clean background, crisp metal edges, "
            "stable motion, high quality video."
        ),
    },
    {
        "name": "rainy_cafe_window",
        "prompt": (
            "A ceramic coffee cup sits on a small wooden table beside a cafe window on a rainy evening. Raindrops slide down the glass, "
            "soft street lights shimmer outside, and a few blurred pedestrians pass in the background. Steam rises from the cup while "
            "the camera makes a slow sideways slide, warm interior lighting, clear foreground subject, natural reflections, high quality video."
        ),
    },
    {
        "name": "lantern_boat_river",
        "prompt": (
            "A small paper lantern boat floats along a quiet river at dusk, passing under a simple stone bridge. The lantern glows warmly, "
            "water ripples spread behind it, and reflected lights move across the surface. A few leaves drift slowly nearby while the camera "
            "tracks the boat from a low angle, calm background detail, stable motion, high quality video."
        ),
    },
    {
        "name": "bookstore_cat_walk",
        "prompt": (
            "An orange cat walks carefully along a wooden table inside a cozy bookstore with shelves in the background. Sunlight enters through "
            "a side window, dust motes float in the beam, and a thin curtain moves gently. The cat turns its head, steps around a small stack "
            "of books, and the camera follows with a smooth slow pan, warm colors, clear subject, high quality video."
        ),
    },
    {
        "name": "plaza_street_musician",
        "prompt": (
            "A street musician plays a violin in a small open plaza during golden hour. String lights hang between simple buildings, a few people "
            "walk softly in the background, and the musician's scarf moves in a light breeze. The bow moves steadily across the strings while "
            "the camera makes a gentle semicircle move, balanced background detail, stable body motion, high quality video."
        ),
    },
    {
        "name": "greenhouse_butterfly",
        "prompt": (
            "A blue butterfly flutters around a red flower inside a bright glass greenhouse. Sunlight passes through the glass panels, a few leaves "
            "sway gently, and soft shadows move on the table below. The butterfly lands briefly on the flower, opens and closes its wings, then "
            "takes off again as the camera slowly pushes closer, vivid but natural colors, high quality video."
        ),
    },
]

_SELECTED_PROMPT = QUALITY_PROMPTS[PROMPT_INDEX] if 0 <= PROMPT_INDEX < len(QUALITY_PROMPTS) else None
_PROMPT_NAME = _SELECTED_PROMPT["name"] if _SELECTED_PROMPT else "invalid_prompt"
_PROMPT = _SELECTED_PROMPT["prompt"] if _SELECTED_PROMPT else ""
_SELECTED_RNR_CONFIG = (
    RNR_CONFIG_PRESETS[RNR_CONFIG_INDEX] if 0 <= RNR_CONFIG_INDEX < len(RNR_CONFIG_PRESETS) else None
)
_RNR_CONFIG_NAME = _SELECTED_RNR_CONFIG["name"] if _SELECTED_RNR_CONFIG else "invalid_config"
_RNR_CONFIG = _SELECTED_RNR_CONFIG["path"] if _SELECTED_RNR_CONFIG else ""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_path() -> Path:
    mode = "rnr_on" if ENABLE_TOKEN_MERGE else "baseline"
    mode_suffix = f"{_RNR_CONFIG_NAME}_{mode}" if ENABLE_TOKEN_MERGE else mode
    return repo_root() / "outputs" / "quality_rnr" / f"quality_{_PROMPT_NAME}_{mode_suffix}.mp4"


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def validate_settings() -> None:
    if _SELECTED_PROMPT is None:
        raise ValueError(f"PROMPT_INDEX must be between 0 and {len(QUALITY_PROMPTS) - 1}, got {PROMPT_INDEX}.")
    if _SELECTED_RNR_CONFIG is None:
        raise ValueError(
            f"RNR_CONFIG_INDEX must be between 0 and {len(RNR_CONFIG_PRESETS) - 1}, got {RNR_CONFIG_INDEX}."
        )

    if _HEIGHT % 8 != 0 or _WIDTH % 8 != 0:
        raise ValueError("_HEIGHT and _WIDTH must both be divisible by 8.")

    if "2b" in _MODEL_ID.lower() and (_NUM_FRAMES > 49 or (_NUM_FRAMES - 1) % 8 != 0):
        raise ValueError("CogVideoX-2b expects _NUM_FRAMES to follow 8N+1 and not exceed 49.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This high-quality test expects an NVIDIA GPU.")


def configure_pipeline(pipe: CogVideoXPipeline) -> bool:
    model_name = _MODEL_ID.lower()
    use_dynamic_cfg = True

    if "2b" in model_name:
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        use_dynamic_cfg = False
    else:
        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )

    if _OFFLOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif _OFFLOAD_MODE == "model":
        pipe.enable_model_cpu_offload()
    elif _OFFLOAD_MODE == "none":
        pipe.to(_DEVICE)
    else:
        raise ValueError(f"Unknown offload mode: {_OFFLOAD_MODE}")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return use_dynamic_cfg


def maybe_reset_cuda_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_gpu_memory_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def peak_gpu_reserved_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / 1024**3


def resolve_model_snapshot() -> str:
    print("Resolving model snapshot...")
    print(f"  model_id={_MODEL_ID}")
    print(f"  download_workers={_DOWNLOAD_WORKERS}")
    return snapshot_download(
        repo_id=_MODEL_ID,
        local_files_only=_LOCAL_FILES_ONLY,
        max_workers=_DOWNLOAD_WORKERS,
    )


def attach_rnr(pipe: CogVideoXPipeline):
    if not ENABLE_TOKEN_MERGE:
        return None, None

    src_dir = repo_root() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from tokmerge.rnr import apply_rnr_to_cogvideox, load_rnr_config

    config_path = repo_root() / _RNR_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"RnR config not found: {config_path}")

    rnr_cfg = load_rnr_config(config_path)
    runtime = apply_rnr_to_cogvideox(pipe.transformer, rnr_cfg)
    runtime.reset_prompt()
    return rnr_cfg, runtime


class TransformerTimingHooks:
    """CUDA-event timing around each transformer forward call."""

    def __init__(self) -> None:
        self.step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._start: torch.cuda.Event | None = None
        self._handles: list = []

    def attach(self, transformer: torch.nn.Module) -> None:
        self._handles = [
            transformer.register_forward_pre_hook(self._pre_hook),
            transformer.register_forward_hook(self._post_hook),
        ]

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _pre_hook(self, module, args) -> None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start = start

    def _post_hook(self, module, args, output) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if self._start is not None:
            self.step_events.append((self._start, end))

    def compute(self) -> dict[str, float | None]:
        if not self.step_events:
            return {
                "transformer_seconds": None,
                "avg_step_seconds": None,
                "first_step_seconds": None,
            }

        torch.cuda.synchronize()
        step_ms = [start.elapsed_time(end) for start, end in self.step_events]
        total_s = sum(step_ms) / 1000.0
        first_s = step_ms[0] / 1000.0
        avg_s = sum(step_ms[1:]) / len(step_ms[1:]) / 1000.0 if len(step_ms) > 1 else first_s
        return {
            "transformer_seconds": round(total_s, 4),
            "avg_step_seconds": round(avg_s, 4),
            "first_step_seconds": round(first_s, 4),
        }


def write_metadata(
    path: Path,
    snapshot_path: str,
    load_seconds: float,
    inference_seconds: float,
    timing_data: dict[str, float | None],
    rnr_cfg,
    rnr_runtime,
    actual_num_frames: int,
) -> Path:
    rnr_stats = rnr_runtime.stats_dict() if rnr_runtime is not None else None
    metadata = {
        "model_id": _MODEL_ID,
        "prompt_index": PROMPT_INDEX,
        "prompt_name": _PROMPT_NAME,
        "prompt": _PROMPT,
        "rnr_config_index": RNR_CONFIG_INDEX,
        "rnr_config_name": _RNR_CONFIG_NAME,
        "selected_rnr_config": str((repo_root() / _RNR_CONFIG).resolve()),
        "seed": _SEED,
        "height": _HEIGHT,
        "width": _WIDTH,
        "requested_num_frames": _NUM_FRAMES,
        "actual_num_frames": actual_num_frames,
        "num_inference_steps": _NUM_INFERENCE_STEPS,
        "guidance_scale": _GUIDANCE_SCALE,
        "fps": _FPS,
        "dtype": _DTYPE_NAME,
        "offload_mode": _OFFLOAD_MODE,
        "diffusers_version": diffusers_pkg.__version__,
        "diffusers_path": str(Path(diffusers_pkg.__file__).resolve()),
        "snapshot_path": snapshot_path,
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "transformer_seconds": timing_data["transformer_seconds"],
        "avg_step_seconds": timing_data["avg_step_seconds"],
        "first_step_seconds": timing_data["first_step_seconds"],
        "rnr_enabled": rnr_cfg is not None,
        "rnr_config": str((repo_root() / _RNR_CONFIG).resolve()) if rnr_cfg else None,
        "rnr_method": rnr_cfg.method if rnr_cfg else None,
        "q_reduce_ratio": rnr_cfg.q_reduce_ratio if rnr_cfg else None,
        "kv_reduce_ratio": rnr_cfg.kv_reduce_ratio if rnr_cfg else None,
        "similarity_type": rnr_cfg.similarity_type if rnr_cfg else None,
        "reduce_mode": rnr_cfg.reduce_mode if rnr_cfg else None,
        "dst_stride": list(rnr_cfg.dst_stride) if rnr_cfg else None,
        "matching_cache_steps": rnr_cfg.matching_cache_steps if rnr_cfg else None,
        "rnr_stats": rnr_stats,
        "peak_gpu_memory_gib": None if peak_gpu_memory_gib() is None else round(peak_gpu_memory_gib(), 3),
        "peak_gpu_reserved_gib": None if peak_gpu_reserved_gib() is None else round(peak_gpu_reserved_gib(), 3),
        "output_path": str(path.resolve()),
    }

    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def print_summary(video_path: Path, metadata_path: Path, timing_data: dict[str, float | None]) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print("\nGeneration completed.")
    print(f"  video={video_path.resolve()}")
    print(f"  metadata={metadata_path.resolve()}")
    print(f"  rnr_enabled={metadata['rnr_enabled']}")
    print(f"  inference_seconds={metadata['inference_seconds']}")
    if timing_data["transformer_seconds"] is not None:
        print(f"  transformer_seconds={timing_data['transformer_seconds']}")
        print(f"  avg_step_seconds={timing_data['avg_step_seconds']}")
        print(f"  first_step_seconds={timing_data['first_step_seconds']}")
    print(f"  peak_gpu_memory_gib={metadata['peak_gpu_memory_gib']}")
    if metadata["rnr_stats"] is not None:
        stats = metadata["rnr_stats"]
        print(f"  rnr_cache_hit_rate={stats['cache_hit_rate']}")
        print(f"  q_reduction_ratio={stats['q_reduction_ratio']}")
        print(f"  kv_reduction_ratio={stats['kv_reduction_ratio']}")


def main() -> int:
    validate_settings()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    video_path = output_path()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== High-quality CogVideoX RnR test ===")
    print(f"  repo={repo_root()}")
    print(f"  rnr_enabled={ENABLE_TOKEN_MERGE}")
    print(f"  prompt_index={PROMPT_INDEX}")
    print(f"  prompt_name={_PROMPT_NAME}")
    print(f"  rnr_config_index={RNR_CONFIG_INDEX}")
    print(f"  rnr_config_name={_RNR_CONFIG_NAME}")
    print(f"  size={_WIDTH}x{_HEIGHT}")
    print(f"  frames={_NUM_FRAMES}")
    print(f"  steps={_NUM_INFERENCE_STEPS}")
    print(f"  offload={_OFFLOAD_MODE}")
    print(f"  output={video_path}")
    print(f"  diffusers_version={diffusers_pkg.__version__}")
    print(f"  diffusers_path={Path(diffusers_pkg.__file__).resolve()}")

    snapshot_path = resolve_model_snapshot()

    load_start = time.perf_counter()
    pipe = CogVideoXPipeline.from_pretrained(
        snapshot_path,
        torch_dtype=resolve_dtype(_DTYPE_NAME),
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - load_start

    use_dynamic_cfg = configure_pipeline(pipe)
    rnr_cfg, rnr_runtime = attach_rnr(pipe)
    if rnr_cfg:
        print(f"  rnr_config={(repo_root() / _RNR_CONFIG).resolve()}")
        print(f"  method={rnr_cfg.method}, q_ratio={rnr_cfg.q_reduce_ratio}, kv_ratio={rnr_cfg.kv_reduce_ratio}")
        print(f"  similarity={rnr_cfg.similarity_type}, reduce_mode={rnr_cfg.reduce_mode}")
        print(f"  dst_stride={rnr_cfg.dst_stride}, cache_steps={rnr_cfg.matching_cache_steps}")

    generator = torch.Generator().manual_seed(_SEED)
    timing_hooks = TransformerTimingHooks()
    timing_hooks.attach(pipe.transformer)

    print("\nRunning generation...")
    maybe_reset_cuda_stats()
    inference_start = time.perf_counter()
    result = pipe(
        prompt=_PROMPT,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        guidance_scale=_GUIDANCE_SCALE,
        use_dynamic_cfg=use_dynamic_cfg,
        generator=generator,
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start

    timing_data = timing_hooks.compute()
    timing_hooks.detach()

    frames = result.frames[0]
    export_to_video(frames, str(video_path), fps=_FPS)

    metadata_path = write_metadata(
        path=video_path,
        snapshot_path=snapshot_path,
        load_seconds=load_seconds,
        inference_seconds=inference_seconds,
        timing_data=timing_data,
        rnr_cfg=rnr_cfg,
        rnr_runtime=rnr_runtime,
        actual_num_frames=len(frames),
    )
    print_summary(video_path, metadata_path, timing_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
