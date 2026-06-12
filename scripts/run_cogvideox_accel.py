from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import diffusers as diffusers_pkg
import torch
from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download


DEFAULT_PROMPT = (
    "A small paper boat floats gently across a sunlit pond, soft ripples spreading outward, "
    "cinematic natural lighting, realistic motion."
)


CSV_FIELDS = [
    "run_id",
    "status",
    "accel",
    "prompt_index",
    "prompt",
    "seed",
    "model_path",
    "height",
    "width",
    "num_frames",
    "num_inference_steps",
    "guidance_scale",
    "dtype",
    "enable_cpu_offload",
    "inference_seconds",
    "transformer_seconds",
    "avg_step_seconds",
    "first_step_seconds",
    "peak_gpu_memory_allocated_gib",
    "peak_gpu_memory_reserved_gib",
    "rnr_matching_seconds",
    "rnr_cache_hit_rate",
    "q_reduction_ratio",
    "kv_reduction_ratio",
    "method_config_hash",
    "output_path",
    "metadata_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CogVideoX baseline/TokenMerge/RnR runner.")
    parser.add_argument("--model_path", default="THUDM/CogVideoX-2b")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output_dir", default="results/run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--enable_cpu_offload", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_latency", action="store_true")
    parser.add_argument("--log_memory", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--accel", choices=("none", "naive_tome", "kv_rnr", "qv_rnr", "rnr_tome"), default="none")
    parser.add_argument("--merge_ratio", type=float, default=0.2)
    parser.add_argument("--q_reduce_ratio", type=float, default=None)
    parser.add_argument("--kv_reduce_ratio", "--v_reduce_ratio", dest="kv_reduce_ratio", type=float, default=None)
    parser.add_argument("--h_reduce_ratio", type=float, default=None)
    parser.add_argument("--similarity_type", choices=("cosine", "euclidean", "dot", "random"), default=None)
    parser.add_argument("--reduce_mode", choices=("replace", "mean"), default=None)
    parser.add_argument("--dst_stride", type=int, nargs=3, default=None)
    parser.add_argument("--partition_mode", choices=("random_chunk", "strided"), default=None)
    parser.add_argument("--matching_cache_steps", type=int, default=None)
    parser.add_argument("--schedule_file", default=None)
    parser.add_argument("--schedule_url", default=None)
    parser.add_argument("--disable_schedule", action="store_true")
    parser.add_argument("--schedule_config", default="configs/rnr/rnr_official_base.yaml")
    parser.add_argument("--benchmark_csv", default="results/benchmark.csv")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--download_workers", type=int, default=1)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        path = Path(args.prompt_file)
        prompts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [DEFAULT_PROMPT]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise ValueError("No prompts were provided.")
    return prompts


def resolve_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def configure_pipeline(pipe: CogVideoXPipeline, model_path: str, args: argparse.Namespace) -> bool:
    model_name = model_path.lower()
    use_dynamic_cfg = True
    if "2b" in model_name:
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        use_dynamic_cfg = False
    else:
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return use_dynamic_cfg


def reset_cuda_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_allocated_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def peak_reserved_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / 1024**3


class TransformerTimingHooks:
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
        if self._start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.step_events.append((self._start, end))

    def compute(self) -> dict[str, float | None]:
        if not self.step_events:
            return {"transformer_seconds": None, "avg_step_seconds": None, "first_step_seconds": None}
        torch.cuda.synchronize()
        step_ms = [start.elapsed_time(end) for start, end in self.step_events]
        total = sum(step_ms) / 1000.0
        first = step_ms[0] / 1000.0
        avg = sum(step_ms[1:]) / len(step_ms[1:]) / 1000.0 if len(step_ms) > 1 else first
        return {
            "transformer_seconds": round(total, 4),
            "avg_step_seconds": round(avg, 4),
            "first_step_seconds": round(first, 4),
        }


def attach_accel(pipe: CogVideoXPipeline, args: argparse.Namespace):
    from tokmerge.runtime import attach_merge_config, detach_merge_config
    from tokmerge.merging import MergeConfig
    from tokmerge.rnr import apply_rnr_to_cogvideox, detach_rnr_from_cogvideox, load_rnr_config

    detach_merge_config(pipe.transformer)
    detach_rnr_from_cogvideox(pipe.transformer)

    if args.accel == "none":
        return None, None

    if args.accel == "naive_tome":
        reuse_interval = max(1, args.matching_cache_steps or 5)
        cfg = MergeConfig(
            enabled=True,
            ratio=args.merge_ratio,
            mode="spatial",
            scope="block",
            rope_mode="pre_rope",
            prop_attn=False,
            match_feature="hidden_norm",
            temporal_window=1,
            protect_first_frame=True,
            partition="checkerboard_shifted",
            reuse_interval=reuse_interval,
        )
        cfg._raw_layers = "middle_wide"
        attach_merge_config(pipe.transformer, cfg)
        return cfg, None

    overrides = {
        "method": args.accel,
        "q_reduce_ratio": args.q_reduce_ratio,
        "kv_reduce_ratio": args.kv_reduce_ratio,
        "h_reduce_ratio": args.h_reduce_ratio,
        "similarity_type": args.similarity_type,
        "reduce_mode": args.reduce_mode,
        "dst_stride": tuple(args.dst_stride) if args.dst_stride is not None else None,
        "partition_mode": args.partition_mode,
        "matching_cache_steps": args.matching_cache_steps,
        "schedule_file": args.schedule_file,
        "schedule_url": args.schedule_url,
        "disable_schedule": True if args.disable_schedule else None,
    }
    cfg = load_rnr_config(args.schedule_config, overrides)
    runtime = apply_rnr_to_cogvideox(pipe.transformer, cfg)
    return cfg, runtime


def config_hash(accel_cfg: Any, args: argparse.Namespace) -> str:
    payload = {
        "accel": args.accel,
        "merge_ratio": args.merge_ratio,
        "q_reduce_ratio": args.q_reduce_ratio,
        "kv_reduce_ratio": args.kv_reduce_ratio,
        "h_reduce_ratio": args.h_reduce_ratio,
        "similarity_type": args.similarity_type,
        "reduce_mode": args.reduce_mode,
        "dst_stride": None if args.dst_stride is None else list(args.dst_stride),
        "partition_mode": args.partition_mode,
        "matching_cache_steps": args.matching_cache_steps,
        "schedule_file": args.schedule_file,
        "schedule_url": args.schedule_url,
        "disable_schedule": args.disable_schedule,
        "config": getattr(accel_cfg, "__dict__", str(accel_cfg)),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one_prompt(
    pipe: CogVideoXPipeline,
    use_dynamic_cfg: bool,
    prompt: str,
    prompt_index: int,
    args: argparse.Namespace,
    accel_cfg: Any,
    rnr_runtime,
    snapshot_path: str,
) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    video_dir = out_dir / "videos"
    log_dir = out_dir / "logs"
    video_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{args.accel}_p{prompt_index:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    video_path = video_dir / f"{run_id}.mp4"
    metadata_path = log_dir / f"{run_id}.json"

    if rnr_runtime is not None:
        rnr_runtime.reset_prompt()

    timing_hooks = None
    if args.log_latency and torch.cuda.is_available():
        timing_hooks = TransformerTimingHooks()
        timing_hooks.attach(pipe.transformer)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    reset_cuda_stats()
    start = time.perf_counter()
    try:
        result = pipe(
            prompt=prompt,
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
        inference_seconds = time.perf_counter() - start
        timing = timing_hooks.compute() if timing_hooks else {
            "transformer_seconds": None,
            "avg_step_seconds": None,
            "first_step_seconds": None,
        }
        frames = result.frames[0]
        if args.save_video:
            export_to_video(frames, str(video_path), fps=args.fps)
        output_path = str(video_path.resolve()) if args.save_video else ""
        status = "ok"
        error = None
    except Exception as exc:
        inference_seconds = time.perf_counter() - start
        timing = {"transformer_seconds": None, "avg_step_seconds": None, "first_step_seconds": None}
        output_path = ""
        status = "failed"
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        if timing_hooks is not None:
            timing_hooks.detach()

    rnr_stats = rnr_runtime.stats_dict() if rnr_runtime is not None else {}
    row = {
        "run_id": run_id,
        "status": status,
        "accel": args.accel,
        "prompt_index": prompt_index,
        "prompt": prompt,
        "seed": args.seed,
        "model_path": args.model_path,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "enable_cpu_offload": args.enable_cpu_offload,
        "inference_seconds": round(inference_seconds, 4),
        "transformer_seconds": timing["transformer_seconds"],
        "avg_step_seconds": timing["avg_step_seconds"],
        "first_step_seconds": timing["first_step_seconds"],
        "peak_gpu_memory_allocated_gib": None if peak_allocated_gib() is None else round(peak_allocated_gib(), 4),
        "peak_gpu_memory_reserved_gib": None if peak_reserved_gib() is None else round(peak_reserved_gib(), 4),
        "rnr_matching_seconds": None if not rnr_stats else round(float(rnr_stats["matching_seconds"]), 6),
        "rnr_cache_hit_rate": None if not rnr_stats else round(float(rnr_stats["cache_hit_rate"]), 6),
        "q_reduction_ratio": None if not rnr_stats else round(float(rnr_stats["q_reduction_ratio"]), 6),
        "kv_reduction_ratio": None if not rnr_stats else round(float(rnr_stats["kv_reduction_ratio"]), 6),
        "method_config_hash": config_hash(accel_cfg, args),
        "output_path": output_path,
        "metadata_path": str(metadata_path.resolve()),
        "error": error,
    }
    metadata = {
        **row,
        "snapshot_path": snapshot_path,
        "diffusers_version": diffusers_pkg.__version__,
        "diffusers_path": str(Path(diffusers_pkg.__file__).resolve()),
        "accel_config": getattr(accel_cfg, "__dict__", None),
        "rnr_stats": rnr_stats,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return row


def main() -> int:
    args = parse_args()
    prompts = load_prompts(args)

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height and width must be divisible by 8")
    if "2b" in args.model_path.lower() and (args.num_frames > 49 or (args.num_frames - 1) % 8 != 0):
        raise ValueError("CogVideoX-2B expects num_frames to follow 8N+1 and not exceed 49")

    print("Resolving model snapshot...")
    snapshot_path = snapshot_download(
        repo_id=args.model_path,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        max_workers=args.download_workers,
    )

    print("Loading pipeline...")
    pipe = CogVideoXPipeline.from_pretrained(
        snapshot_path,
        torch_dtype=resolve_dtype(args.dtype),
        local_files_only=True,
    )
    use_dynamic_cfg = configure_pipeline(pipe, args.model_path, args)
    accel_cfg, rnr_runtime = attach_accel(pipe, args)

    print("Running prompts...")
    print(f"  accel={args.accel}")
    print(f"  prompts={len(prompts)}")
    print(f"  output_dir={Path(args.output_dir).resolve()}")
    print(f"  benchmark_csv={Path(args.benchmark_csv).resolve()}")

    for idx, prompt in enumerate(prompts):
        row = run_one_prompt(pipe, use_dynamic_cfg, prompt, idx, args, accel_cfg, rnr_runtime, snapshot_path)
        append_csv(Path(args.benchmark_csv), row)
        print(f"  [{idx + 1}/{len(prompts)}] {row['status']} {row['run_id']} {row['inference_seconds']}s")
        if row["error"]:
            print(f"    error={row['error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
