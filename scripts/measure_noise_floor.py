"""Run the same baseline config twice and record latent MSE as the non-determinism noise floor.

This quantifies the GPU kernel non-determinism so Phase 2+ can use it as the
acceptance threshold instead of demanding bit-identical output.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure baseline noise floor.")
    parser.add_argument("--preset", default="smoke", choices=("smoke", "quality"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).parent / "run_baseline_smoke_test.py"
    python = sys.executable
    out_dir = Path("outputs/noise_floor")
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_paths = []
    for run_idx in range(2):
        latent_path = out_dir / f"latents_run{run_idx}.pt"
        output_path = out_dir / f"video_run{run_idx}.mp4"
        cmd = [
            python, str(script),
            "--preset", args.preset,
            "--seed", str(args.seed),
            "--output", str(output_path),
            "--transformer-timing",
            "--save-latents", str(latent_path),
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")

        print(f"\n=== Run {run_idx} ===")
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"Run {run_idx} failed with exit code {ret.returncode}")
            return 1
        latent_paths.append(latent_path)

    lat0 = torch.load(str(latent_paths[0]), weights_only=True)
    lat1 = torch.load(str(latent_paths[1]), weights_only=True)

    mse = ((lat0.float() - lat1.float()) ** 2).mean().item()
    max_abs_diff = (lat0.float() - lat1.float()).abs().max().item()

    result = {
        "preset": args.preset,
        "seed": args.seed,
        "latent_shape": list(lat0.shape),
        "latent_dtype": str(lat0.dtype),
        "mse": mse,
        "max_abs_diff": max_abs_diff,
        "identical": mse == 0.0,
    }

    report_path = Path("report/baseline_noise_floor.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== Noise Floor Result ===")
    print(f"  MSE: {mse}")
    print(f"  Max abs diff: {max_abs_diff}")
    print(f"  Identical: {result['identical']}")
    print(f"  Saved to: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
