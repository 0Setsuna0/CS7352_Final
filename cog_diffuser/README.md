# Vendored Diffusers Source

This directory is tracked in the main Git repository.

The runtime source used by the current CogVideoX baseline lives here:

```powershell
cog_diffuser\diffusers
```

After pulling changes, reinstall the editable package if needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

Current vendored source:

- `diffusers`: editable source tree used for CogVideoX model-level changes

Important:

- The current baseline in this project imports `diffusers.CogVideoXPipeline`, so edits under `cog_diffuser/diffusers` are what affect runtime first.
- After pulling source changes, rerun `scripts/setup_env.ps1` or manually install it editable so Python uses the vendored package in `.venv\Lib\site-packages`.

Do not commit downloaded model weights, generated videos, or Hugging Face cache files into this repository.
