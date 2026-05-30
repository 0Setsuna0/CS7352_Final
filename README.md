# CS7352 Course Project

This workspace is set up for `CogVideoX-2B` inference and source-level experimentation on a single NVIDIA GPU.

## Prerequisites

- Windows with PowerShell
- `git`
- `uv` available on `PATH`
- An NVIDIA GPU with a working driver

`setup_env.ps1` uses `uv` to create `.venv`, provision `Python 3.12`, install CUDA-enabled PyTorch, install project requirements, and install the vendored `cog_diffuser/diffusers` source in editable mode.

## Quick Start

```powershell
git clone https://github.com/0Setsuna0/CS7352_Final.git
cd CS7352_Final

powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1

.\.venv\Scripts\Activate.ps1
python .\scripts\check_env.py
python .\scripts\run_baseline_smoke_test.py --preset smoke --download-workers 1
```

The first smoke-test run downloads the `THUDM/CogVideoX-2b` weights from the Hugging Face Hub and can take several minutes.

After the first successful run, you can rerun the quick smoke test without network access:

```powershell
python .\scripts\run_baseline_smoke_test.py --preset smoke --local-files-only
```

To generate a better-looking baseline video after the cache is ready:

```powershell
python .\scripts\run_baseline_smoke_test.py --preset quality --local-files-only
```

`python .\scripts\check_env.py` should report a `diffusers path` inside `cog_diffuser\diffusers` and `diffusers editable checkout active: True`.

## Source Layout

- `scripts/`: setup, environment checks, bootstrap, and smoke-test entry points
- `src/`: project-owned implementation code
- `configs/`: prompts and experiment configurations
- `docs/`: notes, design docs, and implementation writeups
- `report/`: course report assets and drafts
- `cog_diffuser/`: vendored external runtime source tracked in this repository

The vendored `cog_diffuser/diffusers` source is tracked directly in this repository. `scripts/setup_env.ps1` installs it in editable mode so local source edits affect runtime immediately.

The current baseline and most official CogVideoX demo scripts use the `diffusers` implementation of CogVideoX. In practice, source-level model changes for this project should usually target `cog_diffuser/diffusers`, not wrapper scripts alone.

## Vendored Diffusers

This repository now uses a single-repository workflow. The local `diffusers` source lives here:

```powershell
cog_diffuser\diffusers
```

After pulling new source changes from this repository, reinstall the editable package if needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

`python .\scripts\check_env.py` should report a `diffusers path` inside `cog_diffuser\diffusers`.

## GitHub Policy

Commit project code, vendored `diffusers` source, scripts, configs, docs, and report sources.

Do not commit virtual environments, Hugging Face caches, model weights, generated videos, or logs.

See `docs/github_setup.md` for the recommended GitHub repository settings and `CONTRIBUTING.md` for the daily branch and pull-request workflow.
See `docs/cogvideox_source_map.md` for which CogVideoX source tree to modify for baseline inference work.

## Notes

- The current plan targets `CogVideoX-2B`, not `5B`.
- A `12 GB` GPU needs memory-saving options such as CPU offload and VAE tiling during inference.
- Model weights will be downloaded on first use through the Hugging Face Hub.
- If a large model download is unstable, rerun the smoke test with `--download-workers 1` so it resumes more conservatively.
