# AGENTS.md

## Repo Shape

- This is a marimo-based generative-model learning workspace covering VAE, Diffusion, DDIM, variants, and Flow Matching.
- `pyproject.toml` declares the notebook and scientific-computing environment. `tests/` covers reusable mathematical checks.
- The 30 chapter notebooks are the only maintained teaching source. Markdown files hold the route, status, records, and migration references.
- `.omo/` and `.codegraph` are local OpenCode tooling artifacts — ignore them.

## Source Map

- `notebooks/00_home.py` — course map, symbol index, status, and chapter entry points.
- `notebooks/part01_vae/` through `notebooks/part06_flow_matching/` — 30 formal chapters in causal learning order.
- `src/teaching/catalog.py` — single source of truth for chapter titles, prerequisites, new math, bridges, and status.
- `src/teaching/components.py` — shared chapter protocol and marimo teaching UI.
- `src/math_checks/` — numerical checks for KL, normalization, finite differences, and analytic-vs-numeric comparisons.
- `src/models/` — reusable model/training implementations kept outside teaching cells.
- `src/visualization/` — shared style and writable Matplotlib cache configuration.
- `learning-path.md` — current route and completion state.
- `docs/plans/VAE_学习_ToDO_清单.md` — active learning plan. Preserve its staged order: intuition → core formulas → minimal code → observations/ablations → ELBO derivation → research mapping.
- `docs/learning-profile/` — mission, teaching preferences, and learning records.
- `references/papers/vae/` — locally stored VAE papers and tutorials.
- `references/reading-list.md` — authoritative external links and reading notes.

## Derivation Verification

- `scripts/math_check.py` — 项目内可移植的轻量推导验证脚本

## Working Conventions

- Keep existing Chinese prose style when editing notes. English technical terms (`encoder`, `decoder`, `mu`, `logvar`, `ELBO`, `reparameterization trick`) are already mixed into the Chinese notes.
- When adding or editing learning tasks, keep checkbox syntax (`- [ ]` / `- [x]`) and completion-standard blocks consistent with `docs/plans/VAE_学习_ToDO_清单.md`.
- First VAE code implementation: minimal PyTorch + MNIST, explicit `mu`/`logvar`, handwritten reparameterization, separately logged `total loss`, `recon loss`, `kl loss`.
- Ablations tied to the checklist: remove KL, vary `beta` (`0.1`, `1`, `4+`), set `z = mu`, vary latent dimensions (`2`, `16`, `32/64`).

## Math Verification

- Use ML Math skill (`~/.config/opencode/skills/ml-math/`) for all mathematical derivations.
- Core principle: treat every derivation as a claim that should be checked with executable verification.
- Workflow: restate formula → build derivation map → check dimensions → explain at two levels → verify with code → visualize when useful.
- Available script: `scripts/math_check.py` for quick reusable checks. Chapter-specific visualizations remain inside the corresponding marimo notebook.
- For ELBO, KL divergence, reparameterization trick: always verify with code before accepting.
- Reference: `references/guides/math-checklist.md` for auditing long derivations, `references/guides/ml-math-roadmap.md` for background topics.

## Verification

- For Markdown-only changes, manually re-read the affected sections.
- Run all notebook and mathematical checks:

```bash
marimo check --strict notebooks
python -m pytest -q
python -m compileall -q src notebooks
```

- Open the course:

```bash
marimo edit notebooks/00_home.py
```

- Export a representative chapter with executed results:

```bash
marimo export html notebooks/part01_vae/04_kl_divergence.py \
  -o exports/04_kl_divergence.html
```

- Notebooks must not download data or start expensive training at import/open time. Use explicit `mo.ui.run_button()` controls.
