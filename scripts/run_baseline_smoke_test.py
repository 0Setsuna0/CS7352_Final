from __future__ import annotations

import argparse
import diffusers as diffusers_pkg
import json
import time
from pathlib import Path

import torch
from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download


SMOKE_PROMPT = (
    "A small paper boat floats gently across a sunlit pond, soft ripples spreading outward, "
    "cinematic natural lighting, realistic motion."
)

QUALITY_PROMPT = (
    "A panda wearing a small red jacket and a tiny hat sits on a wooden stool in a serene bamboo forest and performs "
    "an energetic acoustic guitar solo. The panda's right paw repeatedly strums the strings in fast rhythmic motion, "
    "the left paw changes chords along the neck of the guitar, the head nods to the beat, the shoulders and upper body "
    "sway from side to side, and the ears bounce subtly with each movement. Bamboo leaves wave visibly in the wind, "
    "loose dust and sunlight drift through the air, and the camera makes a gentle handheld push-in with slight natural "
    "parallax. Realistic fur texture, cinematic golden-hour lighting, crisp details, natural colors, expressive motion, "
    "high quality, dynamic performance."
)

QUALITY_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, noisy, flickering, distorted, deformed, warped anatomy, duplicate subject, "
    "extra limbs, text, watermark, oversaturated, shaky camera, choppy motion, frozen pose, static subject, no movement"
)

PRESETS = {
    "smoke": {
        "prompt": SMOKE_PROMPT,
        "negative_prompt": "",
        "output": "outputs/baseline_smoke_test.mp4",
        "height": 256,
        "width": 384,
        "num_frames": 9,
        "num_inference_steps": 2,
        "guidance_scale": 6.0,
        "fps": 8,
        "dtype": "float16",
        "offload": "sequential",
    },
    "quality": {
        "prompt": QUALITY_PROMPT,
        "negative_prompt": QUALITY_NEGATIVE_PROMPT,
        "output": "outputs/cogvideox_quality.mp4",
        "height": 480,
        "width": 720,
        "num_frames": 49,
        "num_inference_steps": 40,
        "guidance_scale": 5.5,
        "fps": 8,
        "dtype": "float16",
        "offload": "sequential",
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CogVideoX generation with smoke or quality presets."
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS.keys()),
        default="smoke",
        help="Generation preset. Use 'smoke' for a quick env test or 'quality' for a much better result.",
    )
    parser.add_argument("--model-id", default="THUDM/CogVideoX-2b", help="Hugging Face model id.")
    parser.add_argument("--prompt", default=None, help="Text prompt for generation.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--output", default=None, help="Output video path.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--height", type=int, default=None, help="Output height in pixels.")
    parser.add_argument("--width", type=int, default=None, help="Output width in pixels.")
    parser.add_argument("--num-frames", type=int, default=None, help="Requested output frame count.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG guidance scale.")
    parser.add_argument("--fps", type=int, default=None, help="Output video FPS.")
    parser.add_argument(
        "--download-workers",
        type=int,
        default=1,
        help="Concurrent workers used by Hugging Face snapshot download.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only from local cache without attempting network access.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default=None,
        help="Computation dtype used to load the pipeline.",
    )
    parser.add_argument(
        "--offload",
        choices=("sequential", "model", "none"),
        default=None,
        help="Memory strategy for the pipeline.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target device when --offload none is used. Defaults to cuda.",
    )
    return parser.parse_args()


def apply_preset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    for field, value in preset.items():
        if getattr(args, field) is None:
            setattr(args, field, value)
    return args


def validate_generation_args(args: argparse.Namespace) -> None:
    model_name = args.model_id.lower()
    if "2b" in model_name:
        if args.num_frames > 49 or (args.num_frames - 1) % 8 != 0:
            raise ValueError("CogVideoX-2b expects num_frames to follow 8N+1 and not exceed 49.")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def configure_pipeline(pipe: CogVideoXPipeline, model_id: str, offload: str, device: str) -> bool:
    model_name = model_id.lower()
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

    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

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
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def download_snapshot(model_id: str, cache_dir: str | None, local_files_only: bool, download_workers: int) -> str:
    print("Resolving model snapshot...")
    print(f"  download_workers={download_workers}")
    if not local_files_only:
        print("  note=progress is counted by completed files, so it can stay at 0/N for a long time on large weights")
    return snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        max_workers=download_workers,
    )


def main() -> int:
    args = apply_preset_defaults(parse_args())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height and width must both be divisible by 8")

    validate_generation_args(args)

    if not torch.cuda.is_available() and args.offload != "none":
        raise RuntimeError("CUDA is not available; this smoke test expects an NVIDIA GPU.")

    dtype = resolve_dtype(args.dtype)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading pipeline...")
    print(f"  preset={args.preset}")
    print(f"  model_id={args.model_id}")
    print(f"  dtype={args.dtype}")
    print(f"  offload={args.offload}")
    print(f"  diffusers_version={diffusers_pkg.__version__}")
    print(f"  diffusers_path={Path(diffusers_pkg.__file__).resolve()}")

    snapshot_path = download_snapshot(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        download_workers=args.download_workers,
    )

    load_start = time.perf_counter()
    pipe = CogVideoXPipeline.from_pretrained(
        snapshot_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    load_elapsed = time.perf_counter() - load_start

    use_dynamic_cfg = configure_pipeline(pipe, args.model_id, args.offload, args.device)
    generator = torch.Generator().manual_seed(args.seed)

    print("Running generation...")
    print(
        f"  prompt={args.prompt}\n"
        f"  size={args.width}x{args.height}\n"
        f"  frames={args.num_frames}\n"
        f"  steps={args.num_inference_steps}"
    )

    maybe_reset_cuda_stats()
    infer_start = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        use_dynamic_cfg=use_dynamic_cfg,
        generator=generator,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_elapsed = time.perf_counter() - infer_start

    frames = result.frames[0]
    export_to_video(frames, str(output_path), fps=args.fps)

    metadata = {
        "model_id": args.model_id,
        "preset": args.preset,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "requested_num_frames": args.num_frames,
        "actual_num_frames": len(frames),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "offload": args.offload,
        "download_workers": args.download_workers,
        "diffusers_version": diffusers_pkg.__version__,
        "diffusers_path": str(Path(diffusers_pkg.__file__).resolve()),
        "snapshot_path": snapshot_path,
        "load_seconds": round(load_elapsed, 3),
        "inference_seconds": round(infer_elapsed, 3),
        "peak_gpu_memory_gib": None if peak_gpu_memory_gib() is None else round(peak_gpu_memory_gib(), 3),
        "output_path": str(output_path.resolve()),
    }

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Smoke test completed.")
    print(f"  video={output_path.resolve()}")
    print(f"  metadata={metadata_path.resolve()}")
    print(f"  load_seconds={metadata['load_seconds']}")
    print(f"  inference_seconds={metadata['inference_seconds']}")
    print(f"  peak_gpu_memory_gib={metadata['peak_gpu_memory_gib']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
