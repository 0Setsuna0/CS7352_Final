from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def version_or_missing(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"missing ({exc.__class__.__name__}: {exc})"
    return getattr(module, "__version__", "unknown")


def module_path_or_missing(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"missing ({exc.__class__.__name__}: {exc})"
    return getattr(module, "__file__", "built-in")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    local_diffusers_root = (project_root / "cog_diffuser" / "diffusers").resolve()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"Torch import failed: {exc}")
        return 1

    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Torch path: {module_path_or_missing('torch')}")

    if torch.cuda.is_available():
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024 ** 3:.2f} GiB")

    for module_name in [
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "PIL",
        "imageio_ffmpeg",
        "sentencepiece",
        "numpy",
    ]:
        print(f"{module_name}: {version_or_missing(module_name)}")

    diffusers_path = Path(module_path_or_missing("diffusers")).resolve()
    print(f"diffusers path: {diffusers_path}")
    print(f"diffusers editable checkout active: {local_diffusers_root in diffusers_path.parents}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
