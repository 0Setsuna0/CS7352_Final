# Vendored Diffusers Source

This directory is tracked in the main Git repository.

The runtime source used by the current CogVideoX baseline lives here:

```powershell
cog_diffuser\diffusers
```

Because this source tree is installed in editable mode, ordinary Python source changes take effect immediately after `git pull`.

Rerun `scripts/setup_env.ps1` only when `.venv` is missing or broken, `requirements.txt` changed, or packaging metadata under `cog_diffuser/diffusers` changed.

Current vendored source:

- `diffusers`: editable source tree used for CogVideoX model-level changes

Important:

- The current baseline in this project imports `diffusers.CogVideoXPipeline`, so edits under `cog_diffuser/diffusers` are what affect runtime first.
- After pulling ordinary source changes, no reinstall is needed. Rerun `scripts/setup_env.ps1` only if the environment or package metadata changed.

Do not commit downloaded model weights, generated videos, or Hugging Face cache files into this repository.
