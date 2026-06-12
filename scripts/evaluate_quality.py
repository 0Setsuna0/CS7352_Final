from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np


FIELDS = [
    "pair_id",
    "prompt",
    "baseline_video",
    "candidate_video",
    "num_frames_compared",
    "frame_mse",
    "frame_psnr",
    "temporal_diff_baseline",
    "temporal_diff_candidate",
    "temporal_diff_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight baseline-vs-accelerated video quality evaluation.")
    parser.add_argument("--pairs_csv", required=True, help="CSV with baseline_video,candidate_video,prompt columns.")
    parser.add_argument("--output_dir", default="results/quality")
    parser.add_argument("--max_frames", type=int, default=16)
    return parser.parse_args()


def read_video(path: str, max_frames: int) -> np.ndarray:
    frames = []
    for idx, frame in enumerate(iio.imiter(path)):
        if idx >= max_frames:
            break
        arr = np.asarray(frame).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        frames.append(arr[..., :3])
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def frame_mse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.mean((a[:n] - b[:n]) ** 2))


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def temporal_diff(video: np.ndarray) -> float:
    if len(video) < 2:
        return 0.0
    return float(np.mean(np.abs(video[1:] - video[:-1])))


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_review(rows: list[dict[str, str]], output_dir: Path) -> None:
    lines = [
        "# Pairwise Quality Review",
        "",
        "Manual review checklist: blur, pixelation, flicker, structure distortion, motion instability.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['pair_id']}",
                "",
                f"- Prompt: {row.get('prompt', '')}",
                f"- Baseline: `{row['baseline_video']}`",
                f"- Candidate: `{row['candidate_video']}`",
                f"- Frame MSE: {row['frame_mse']}",
                f"- Frame PSNR: {row['frame_psnr']}",
                f"- Temporal diff delta: {row['temporal_diff_delta']}",
                "- Human notes: TODO",
                "",
            ]
        )
    (output_dir / "pairwise_review.md").write_text("\n".join(lines), encoding="utf-8")


def write_html(rows: list[dict[str, str]], output_dir: Path) -> None:
    body = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>Qualitative Grid</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} table{border-collapse:collapse;width:100%} "
        "td,th{border:1px solid #ccc;padding:8px;vertical-align:top} video{max-width:360px;width:100%}</style>",
        "</head><body><h1>Qualitative Grid</h1><table>",
        "<tr><th>Prompt</th><th>Baseline</th><th>Candidate</th><th>Metrics</th></tr>",
    ]
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row.get('prompt', '')}</td>"
            f"<td><video src=\"{row['baseline_video']}\" controls muted loop></video></td>"
            f"<td><video src=\"{row['candidate_video']}\" controls muted loop></video></td>"
            f"<td>MSE: {row['frame_mse']}<br>PSNR: {row['frame_psnr']}<br>"
            f"Temporal delta: {row['temporal_diff_delta']}</td>"
            "</tr>"
        )
    body.extend(["</table></body></html>"])
    (output_dir / "qualitative_grid.html").write_text("\n".join(body), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, pair in enumerate(read_pairs(Path(args.pairs_csv))):
        baseline_path = pair["baseline_video"]
        candidate_path = pair["candidate_video"]
        baseline = read_video(baseline_path, args.max_frames)
        candidate = read_video(candidate_path, args.max_frames)
        n = min(len(baseline), len(candidate))
        mse = frame_mse(baseline, candidate)
        t_base = temporal_diff(baseline[:n])
        t_candidate = temporal_diff(candidate[:n])
        row = {
            "pair_id": pair.get("pair_id") or f"pair_{idx:03d}",
            "prompt": pair.get("prompt", ""),
            "baseline_video": baseline_path,
            "candidate_video": candidate_path,
            "num_frames_compared": n,
            "frame_mse": round(mse, 8),
            "frame_psnr": round(psnr_from_mse(mse), 4) if math.isfinite(psnr_from_mse(mse)) else "inf",
            "temporal_diff_baseline": round(t_base, 8),
            "temporal_diff_candidate": round(t_candidate, 8),
            "temporal_diff_delta": round(t_candidate - t_base, 8),
        }
        rows.append(row)

    metrics_path = output_dir / "quality_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    write_review(rows, output_dir)
    write_html(rows, output_dir)
    print(f"quality_metrics={metrics_path.resolve()}")
    print(f"pairwise_review={(output_dir / 'pairwise_review.md').resolve()}")
    print(f"qualitative_grid={(output_dir / 'qualitative_grid.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
