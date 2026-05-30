# CogVideoX Source Map

If you need to modify "CogVideoX source code" for this course project, first decide which runtime path you are actually using.

## What This Repository Runs Today

The current baseline script, [`scripts/run_baseline_smoke_test.py`](/C:/WorkSpace/AI/CS7352_Final/scripts/run_baseline_smoke_test.py), imports:

- `diffusers.CogVideoXPipeline`
- `diffusers.CogVideoXDDIMScheduler`
- `diffusers.CogVideoXDPMScheduler`

So for the baseline inference path, the effective source tree is:

- `cog_diffuser/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py`
- `cog_diffuser/diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py`
- `cog_diffuser/diffusers/src/diffusers/models/attention_processor.py`

## Which Repository Should You Edit?

Use `cog_diffuser/diffusers` when:

- you are changing inference-time transformer behavior
- you are inserting token merging into CogVideoX blocks
- you are changing attention behavior, hidden states, or scheduler-facing pipeline logic
- you want the existing smoke test and future evaluation scripts to pick up your changes

Important: most official CogVideoX wrappers still call into `diffusers` for inference. Editing only wrappers usually does not change model internals.

## Recommended Edit Points For Token Merging

For a training-free token-merging project, the most likely entry points are:

1. `cog_diffuser/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py`
2. `cog_diffuser/diffusers/src/diffusers/models/attention_processor.py`
3. `cog_diffuser/diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py`

Typical split of responsibilities:

- Put token selection, merge, and restore logic near `CogVideoXBlock` or `CogVideoXTransformer3DModel`.
- Put attention-shape-sensitive logic in the attention processor only if you truly need to alter QKV attention execution.
- Put experiment toggles, layer selection, and runtime hooks in the pipeline or in project-owned wrapper code under `src/`.

## Make Sure Local Edits Are Active

After the first environment setup, ordinary Python source changes under `cog_diffuser/diffusers` take effect immediately because the package is installed in editable mode.

Rerun environment setup only if `.venv` is missing or broken, `requirements.txt` changed, or package metadata changed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

Or, if `.venv` already exists:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -e .\cog_diffuser\diffusers
```

Then verify with:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_env.py
```

You should see `diffusers path:` pointing inside `cog_diffuser/diffusers`.

## Sharing Source Changes With Teammates

Because `cog_diffuser/diffusers` is vendored into the main project repository, its source changes are committed here together with the rest of the project.
