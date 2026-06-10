r"""Standalone high-quality CogVideoX test with a single TokenMerge toggle.

Normal usage:
  1. Edit only ENABLE_TOKEN_MERGE below.
  2. Run:
       .\.venv\Scripts\python.exe .\scripts\run_quality_tokenmerge_test.py

The script writes both the video and a JSON metadata file under outputs/.
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


# The only user-facing switch: True enables TokenMerge, False runs without it.
ENABLE_TOKEN_MERGE = False


_MODEL_ID = "THUDM/CogVideoX-2b"
_TOKEN_MERGE_CONFIG = "configs/merge/kv_spatial_r20_mid.json"
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

_PROMPT = (
    "A panda wearing a small red jacket and a tiny hat sits on a wooden stool in a serene bamboo forest and performs "
    "an energetic acoustic guitar solo. The panda's right paw repeatedly strums the strings in fast rhythmic motion, "
    "the left paw changes chords along the neck of the guitar, the head nods to the beat, the shoulders and upper body "
    "sway from side to side, and the ears bounce subtly with each movement. Bamboo leaves wave visibly in the wind, "
    "loose dust and sunlight drift through the air, and the camera makes a gentle handheld push-in with slight natural "
    "parallax. Realistic fur texture, cinematic golden-hour lighting, crisp details, natural colors, expressive motion, "
    "high quality, dynamic performance."
)

_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, noisy, flickering, distorted, deformed, warped anatomy, duplicate subject, "
    "extra limbs, text, watermark, oversaturated, shaky camera, choppy motion, frozen pose, static subject, no movement"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_path() -> Path:
    mode = "on" if ENABLE_TOKEN_MERGE else "off"
    return repo_root() / "outputs" / "quality_tokenmerge" / f"quality_tokenmerge_{mode}.mp4"


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def validate_settings() -> None:
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


def resolve_model_snapshot() -> str:
    print("Resolving model snapshot...")
    print(f"  model_id={_MODEL_ID}")
    print(f"  download_workers={_DOWNLOAD_WORKERS}")
    return snapshot_download(
        repo_id=_MODEL_ID,
        local_files_only=_LOCAL_FILES_ONLY,
        max_workers=_DOWNLOAD_WORKERS,
    )


def attach_token_merge(pipe: CogVideoXPipeline):
    if not ENABLE_TOKEN_MERGE:
        return None

    src_dir = repo_root() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from tokmerge.runtime import attach_merge_config, load_merge_config

    config_path = repo_root() / _TOKEN_MERGE_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"TokenMerge config not found: {config_path}")

    merge_cfg = load_merge_config(config_path)
    attach_merge_config(pipe.transformer, merge_cfg)
    return merge_cfg


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
    merge_cfg,
    actual_num_frames: int,
) -> Path:
    metadata = {
        "model_id": _MODEL_ID,
        "prompt": _PROMPT,
        "negative_prompt": _NEGATIVE_PROMPT,
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
        "merge_enabled": merge_cfg.enabled if merge_cfg else False,
        "merge_config": str((repo_root() / _TOKEN_MERGE_CONFIG).resolve()) if merge_cfg else None,
        "merge_scope": merge_cfg.scope if merge_cfg else None,
        "merge_ratio": merge_cfg.ratio if merge_cfg else None,
        "merge_mode": merge_cfg.mode if merge_cfg else None,
        "rope_mode": merge_cfg.rope_mode if merge_cfg else None,
        "prop_attn": merge_cfg.prop_attn if merge_cfg else None,
        "partition": merge_cfg.partition if merge_cfg else None,
        "peak_gpu_memory_gib": None if peak_gpu_memory_gib() is None else round(peak_gpu_memory_gib(), 3),
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
    print(f"  merge_enabled={metadata['merge_enabled']}")
    print(f"  inference_seconds={metadata['inference_seconds']}")
    if timing_data["transformer_seconds"] is not None:
        print(f"  transformer_seconds={timing_data['transformer_seconds']}")
        print(f"  avg_step_seconds={timing_data['avg_step_seconds']}")
        print(f"  first_step_seconds={timing_data['first_step_seconds']}")
    print(f"  peak_gpu_memory_gib={metadata['peak_gpu_memory_gib']}")


def main() -> int:
    validate_settings()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    video_path = output_path()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== High-quality CogVideoX TokenMerge test ===")
    print(f"  repo={repo_root()}")
    print(f"  token_merge={ENABLE_TOKEN_MERGE}")
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
    merge_cfg = attach_token_merge(pipe)
    if merge_cfg:
        print(f"  merge_config={(repo_root() / _TOKEN_MERGE_CONFIG).resolve()}")
        print(f"  merge_scope={merge_cfg.scope}, ratio={merge_cfg.ratio}, mode={merge_cfg.mode}")
        print(f"  partition={merge_cfg.partition}")
        print(f"  merge_layers={merge_cfg.layers}")

    generator = torch.Generator().manual_seed(_SEED)
    timing_hooks = TransformerTimingHooks()
    timing_hooks.attach(pipe.transformer)

    print("\nRunning generation...")
    maybe_reset_cuda_stats()
    inference_start = time.perf_counter()
    result = pipe(
        prompt=_PROMPT,
        negative_prompt=_NEGATIVE_PROMPT,
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
        merge_cfg=merge_cfg,
        actual_num_frames=len(frames),
    )
    print_summary(video_path, metadata_path, timing_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
