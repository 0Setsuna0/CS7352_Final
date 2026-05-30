# GitHub Collaboration Setup

This project is managed as a single GitHub repository:

1. `CS7352_Final`: the course project repository for scripts, configs, docs, report sources, project-owned code, and vendored `cog_diffuser/diffusers` runtime source.

## Recommended Ownership

Preferred: create a GitHub organization or a team-owned account and host both repositories there.

If you keep `CS7352_Final` under a personal account, give at least one teammate admin access so the project is not blocked by a single owner account.

## Repository Settings Checklist

Apply these settings in the GitHub web UI for `CS7352_Final`:

1. Set the default branch to `main`.
2. Enable Issues.
3. Enable Discussions only if your team wants design discussion threads on GitHub instead of chat.
4. Enable Projects if you want a lightweight Kanban board for milestones and report tasks.
5. Add all teammates with `Write` access.
6. Give `Admin` access to two maintainers if possible.

## Branch Protection for `main`

Create a branch protection rule for `main` with these settings:

1. Require a pull request before merging.
2. Require at least one approval.
3. Require conversation resolution before merging.
4. Require status checks to pass before merging.
5. Include the `repo-check` workflow as a required status check after it appears.
6. Disable force pushes.
7. Disable branch deletion.

For a small course team, one approval is usually enough. If your team is larger or the grading stakes are high, use two approvals for changes touching evaluation code or the final report.

## Suggested Branch Naming

- `feat/baseline-inference`
- `feat/token-merging-hook`
- `exp/merge-ratio-20`
- `fix/memory-offload`
- `docs/final-report-outline`

## Labels and Milestones

Useful labels:

- `baseline`
- `token-merging`
- `evaluation`
- `report`
- `infra`
- `bug`

Suggested milestones:

- `Weeks 1-2 Baseline`
- `Weeks 3-4 Token Merging`
- `Weeks 5-6 Experiments and Report`

## Vendored Runtime Source

If you need to modify `diffusers`, do it directly inside this repository:

1. Edit files under `cog_diffuser/diffusers`.
2. Commit those changes in the same PR as your project code when they belong together.
3. Reinstall the editable package after pulling or after major dependency changes:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

This keeps the project workflow simple for a small team at the cost of a larger repository.

## Secrets and Large Files

- Keep tokens in local `.env` files only.
- Do not commit model weights, caches, or generated videos.
- Do not enable Git LFS for checkpoints or generated outputs for this class project unless your instructor explicitly requires artifact delivery through GitHub.

## Recommended Pull Request Rules

Every PR should answer:

1. What changed?
2. Why does the team need it now?
3. How was it validated?
4. Does it affect baseline metrics, prompts, or report claims?
