r"""Run every TokenMerge config with the current high-quality settings.

This script imports the generation settings from run_quality_tokenmerge_test.py
and sweeps configs/merge/*.json. It saves one video and one metadata JSON per
run, then writes summary.csv and summary.md.

Run:
  .\.venv\Scripts\python.exe .\scripts\run_all_tokenmerge_configs.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import diffusers as diffusers_pkg
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

import run_quality_tokenmerge_test as quality


RUN_BASELINE = True
OUTPUT_DIR = quality.repo_root() / "outputs" / "config_sweep"
CONFIG_DIR = quality.repo_root() / "configs" / "merge"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"


def config_paths() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.json"), key=lambda path: path.name)


def experiment_name(config_path: Path | None) -> str:
    return "baseline_no_merge" if config_path is None else config_path.stem


def load_config_json(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def attach_config(pipe: CogVideoXPipeline, config_path: Path | None):
    src_dir = quality.repo_root() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from tokmerge.runtime import attach_merge_config, detach_merge_config, load_merge_config

    detach_merge_config(pipe.transformer)
    if config_path is None:
        return None

    merge_cfg = load_merge_config(config_path)
    attach_merge_config(pipe.transformer, merge_cfg)
    return merge_cfg


def run_generation(
    pipe: CogVideoXPipeline,
    use_dynamic_cfg: bool,
    snapshot_path: str,
    config_path: Path | None,
    load_seconds: float,
) -> dict[str, Any]:
    name = experiment_name(config_path)
    video_path = OUTPUT_DIR / f"{name}.mp4"
    metadata_path = OUTPUT_DIR / f"{name}.json"
    started_at = datetime.now().isoformat(timespec="seconds")

    config_raw = load_config_json(config_path)
    merge_cfg = attach_config(pipe, config_path)
    generator = torch.Generator().manual_seed(quality._SEED)

    timing_hooks = quality.TransformerTimingHooks()
    timing_hooks.attach(pipe.transformer)

    print("\n=== Running experiment ===")
    print(f"  name={name}")
    print(f"  config={config_path}")
    print(f"  output={video_path}")
    if merge_cfg:
        print(f"  scope={merge_cfg.scope}, ratio={merge_cfg.ratio}, mode={merge_cfg.mode}")
        print(f"  layers={merge_cfg.layers}")

    try:
        quality.maybe_reset_cuda_stats()
        inference_start = time.perf_counter()
        result = pipe(
            prompt=quality._PROMPT,
            negative_prompt=quality._NEGATIVE_PROMPT,
            height=quality._HEIGHT,
            width=quality._WIDTH,
            num_frames=quality._NUM_FRAMES,
            num_inference_steps=quality._NUM_INFERENCE_STEPS,
            guidance_scale=quality._GUIDANCE_SCALE,
            use_dynamic_cfg=use_dynamic_cfg,
            generator=generator,
        )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        timing_data = timing_hooks.compute()
    finally:
        timing_hooks.detach()

    frames = result.frames[0]
    export_to_video(frames, str(video_path), fps=quality._FPS)

    metadata = {
        "experiment": name,
        "status": "ok",
        "config_file": None if config_path is None else config_path.name,
        "config_path": None if config_path is None else str(config_path.resolve()),
        "model_id": quality._MODEL_ID,
        "prompt": quality._PROMPT,
        "negative_prompt": quality._NEGATIVE_PROMPT,
        "seed": quality._SEED,
        "height": quality._HEIGHT,
        "width": quality._WIDTH,
        "requested_num_frames": quality._NUM_FRAMES,
        "actual_num_frames": len(frames),
        "num_inference_steps": quality._NUM_INFERENCE_STEPS,
        "guidance_scale": quality._GUIDANCE_SCALE,
        "fps": quality._FPS,
        "dtype": quality._DTYPE_NAME,
        "offload_mode": quality._OFFLOAD_MODE,
        "diffusers_version": diffusers_pkg.__version__,
        "diffusers_path": str(Path(diffusers_pkg.__file__).resolve()),
        "snapshot_path": snapshot_path,
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "transformer_seconds": timing_data["transformer_seconds"],
        "avg_step_seconds": timing_data["avg_step_seconds"],
        "first_step_seconds": timing_data["first_step_seconds"],
        "merge_enabled": merge_cfg.enabled if merge_cfg else False,
        "merge_scope": merge_cfg.scope if merge_cfg else None,
        "merge_ratio": merge_cfg.ratio if merge_cfg else None,
        "merge_mode": merge_cfg.mode if merge_cfg else None,
        "rope_mode": merge_cfg.rope_mode if merge_cfg else None,
        "prop_attn": merge_cfg.prop_attn if merge_cfg else None,
        "match_feature": merge_cfg.match_feature if merge_cfg else None,
        "layers": list(merge_cfg.layers) if merge_cfg else None,
        "raw_layers": config_raw.get("layers"),
        "temporal_window": merge_cfg.temporal_window if merge_cfg else None,
        "protect_first_frame": merge_cfg.protect_first_frame if merge_cfg else None,
        "skip_early_ratio": merge_cfg.skip_early_ratio if merge_cfg else None,
        "peak_gpu_memory_gib": None if quality.peak_gpu_memory_gib() is None else round(quality.peak_gpu_memory_gib(), 3),
        "output_path": str(video_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "error": None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def failed_row(config_path: Path | None, error: BaseException) -> dict[str, Any]:
    name = experiment_name(config_path)
    return {
        "experiment": name,
        "status": "failed",
        "config_file": None if config_path is None else config_path.name,
        "config_path": None if config_path is None else str(config_path.resolve()),
        "model_id": quality._MODEL_ID,
        "seed": quality._SEED,
        "height": quality._HEIGHT,
        "width": quality._WIDTH,
        "requested_num_frames": quality._NUM_FRAMES,
        "num_inference_steps": quality._NUM_INFERENCE_STEPS,
        "guidance_scale": quality._GUIDANCE_SCALE,
        "fps": quality._FPS,
        "dtype": quality._DTYPE_NAME,
        "offload_mode": quality._OFFLOAD_MODE,
        "merge_enabled": config_path is not None,
        "output_path": str((OUTPUT_DIR / f"{name}.mp4").resolve()),
        "metadata_path": str((OUTPUT_DIR / f"{name}.json").resolve()),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "error": "".join(traceback.format_exception_only(type(error), error)).strip(),
    }


SUMMARY_FIELDS = [
    "experiment",
    "status",
    "config_file",
    "merge_enabled",
    "merge_scope",
    "merge_ratio",
    "merge_mode",
    "match_feature",
    "prop_attn",
    "raw_layers",
    "layers",
    "skip_early_ratio",
    "temporal_window",
    "offload_mode",
    "inference_seconds",
    "transformer_seconds",
    "avg_step_seconds",
    "first_step_seconds",
    "peak_gpu_memory_gib",
    "e2e_speedup_vs_baseline",
    "transformer_speedup_vs_baseline",
    "output_path",
    "error",
]


def add_speedups(rows: list[dict[str, Any]]) -> None:
    baseline = next((row for row in rows if row.get("experiment") == "baseline_no_merge" and row.get("status") == "ok"), None)
    if not baseline:
        return

    baseline_infer = baseline.get("inference_seconds")
    baseline_transformer = baseline.get("transformer_seconds")
    for row in rows:
        if row.get("status") != "ok":
            continue
        if baseline_infer and row.get("inference_seconds"):
            row["e2e_speedup_vs_baseline"] = round(float(baseline_infer) / float(row["inference_seconds"]), 4)
        if baseline_transformer and row.get("transformer_seconds"):
            row["transformer_speedup_vs_baseline"] = round(float(baseline_transformer) / float(row["transformer_seconds"]), 4)


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value).replace("\n", " ")


def write_summary(rows: list[dict[str, Any]]) -> None:
    add_speedups(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# TokenMerge Config Sweep",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model: `{quality._MODEL_ID}`",
        f"- Size: `{quality._WIDTH}x{quality._HEIGHT}`, frames: `{quality._NUM_FRAMES}`, steps: `{quality._NUM_INFERENCE_STEPS}`",
        f"- Offload: `{quality._OFFLOAD_MODE}`, dtype: `{quality._DTYPE_NAME}`, seed: `{quality._SEED}`",
        "",
        "| Experiment | Status | Ratio | Mode | Scope | Match | Prop | Layers | Skip | Inference s | Transformer s | E2E speedup | Transformer speedup | Peak GiB | Video | Error |",
        "|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        video = Path(row.get("output_path", "")).name if row.get("output_path") else ""
        md_lines.append(
            "| "
            + " | ".join(
                compact(value)
                for value in [
                    row.get("experiment"),
                    row.get("status"),
                    row.get("merge_ratio"),
                    row.get("merge_mode"),
                    row.get("merge_scope"),
                    row.get("match_feature"),
                    row.get("prop_attn"),
                    row.get("raw_layers"),
                    row.get("skip_early_ratio"),
                    row.get("inference_seconds"),
                    row.get("transformer_seconds"),
                    row.get("e2e_speedup_vs_baseline"),
                    row.get("transformer_speedup_vs_baseline"),
                    row.get("peak_gpu_memory_gib"),
                    video,
                    row.get("error"),
                ]
            )
            + " |"
        )
    SUMMARY_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    quality.validate_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    experiments: list[Path | None] = []
    if RUN_BASELINE:
        experiments.append(None)
    experiments.extend(config_paths())

    print("=== TokenMerge config sweep ===")
    print(f"  output_dir={OUTPUT_DIR}")
    print(f"  experiments={len(experiments)}")
    print(f"  offload={quality._OFFLOAD_MODE}")
    print(f"  size={quality._WIDTH}x{quality._HEIGHT}")
    print(f"  frames={quality._NUM_FRAMES}")
    print(f"  steps={quality._NUM_INFERENCE_STEPS}")

    snapshot_path = quality.resolve_model_snapshot()

    load_start = time.perf_counter()
    pipe = CogVideoXPipeline.from_pretrained(
        snapshot_path,
        torch_dtype=quality.resolve_dtype(quality._DTYPE_NAME),
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - load_start
    use_dynamic_cfg = quality.configure_pipeline(pipe)

    rows: list[dict[str, Any]] = []
    for idx, config_path in enumerate(experiments, start=1):
        print(f"\n--- Experiment {idx}/{len(experiments)} ---")
        try:
            row = run_generation(
                pipe=pipe,
                use_dynamic_cfg=use_dynamic_cfg,
                snapshot_path=snapshot_path,
                config_path=config_path,
                load_seconds=load_seconds,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"FAILED: {experiment_name(config_path)}")
            print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            row = failed_row(config_path, exc)
            Path(row["metadata_path"]).write_text(json.dumps(row, indent=2), encoding="utf-8")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        write_summary(rows)
        print(f"  summary_csv={SUMMARY_CSV}")
        print(f"  summary_md={SUMMARY_MD}")

    print("\nSweep completed.")
    print(f"  summary_csv={SUMMARY_CSV.resolve()}")
    print(f"  summary_md={SUMMARY_MD.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
