# Contributing Guide

This repository is organized in single-repository mode for a course project. The main code, docs, report, and vendored `diffusers` runtime source all live here.

## Repository Roles

- `main`: stable branch for code that teammates can pull and run.
- `feat/*`: project feature work in this repository.
- `exp/*`: short-lived experiment branches for timing, metrics, or ablations.
- `fix/*`: bug fixes.
- `docs/*`: report, documentation, and setup updates.

## Daily Workflow

1. Pull the latest `main`.
2. Create a focused branch for one task or experiment.
3. Open a draft pull request early if the change will take more than one day.
4. Merge through pull request only. Do not push directly to `main`.

## Before Opening a Pull Request

- Keep commits focused and easy to review.
- Update docs or configs if the behavior or setup changes.
- Run `python .\scripts\check_env.py` if you changed environment setup.
- Run `python -m compileall scripts src`.
- Run `python .\scripts\run_baseline_smoke_test.py --download-workers 1` when inference code or model wiring changes.

## Third-Party Source Policy

- Keep project-owned changes in `src/`, `configs/`, `docs/`, and `report/`.
- Keep CogVideoX runtime source edits in `cog_diffuser/diffusers`.
- Do not create a nested git repository under `cog_diffuser/diffusers`.
- Reinstall the editable package after pulling or changing vendored runtime source:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

## Data and Artifact Policy

- Never commit `.env`, Hugging Face tokens, model weights, caches, generated videos, or logs.
- Share large outputs through cloud storage, a drive folder, or a model hub, then link them from `docs/` or the final report.
- Keep reproducible prompts, configs, and experiment notes in Git.
