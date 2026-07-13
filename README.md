# PyTorch Genomic Models

A learning and research platform for building PyTorch-based genomic and multi-modal deep learning models, oriented toward agricultural pest control research with *Drosophila melanogaster* public data as the primary pre-training foundation.

## Quick Start

```bash
# Recommended: uv with dev tools (nbstripout, pytest, black)
uv venv
uv sync

# Run the end-to-end genomic classifier example
uv run python e2e_ex.py
```

### Optional dependency extras

| Extra | Install | Purpose |
|-------|---------|---------|
| `pipelines` / `curation` | `uv sync --extra pipelines` | Goal 3 data curation pipelines (pysam, scanpy, torch-geometric, …) |
| `midgut` | `uv sync --extra midgut` | Goal 2 multi-modal training (transformers, fair-esm, peft, mlflow, …) |
| `all` | `uv sync --extra all` | Both extras above |

Without uv, install base deps from `requirements.txt` (kept in sync with `[project.dependencies]` in `pyproject.toml`).

## Repository Layout

| Path | Description |
|------|-------------|
| `docs/` | Concept guides: MLM, k-mer pre-training, fine-tuning, attention |
| `e2e_explorer.md` | Readable end-to-end walkthrough of genomic model building blocks |
| `e2e_explorer.ipynb` | Runnable notebook companion (Markdown explainer + compute note before each code cell) |
| `e2e_ex.py` | End-to-end genomic classifier training script |
| `embedding_ex.py`, `convolution.py`, `attention.py`, … | Standalone PyTorch examples for individual concepts |
| `models/` | Goal 2 midgut multi-modal model code |
| `data/` | Goal 3 schemas, pipelines, and manifests |
| `infra/` | Goal 4 Terraform / ParallelCluster infrastructure |
| `MASTER_PLAN.md` | Program index linking all five goals + milestone phase |
| `PLAN_GOAL*.md` | Per-goal implementation plans |
| `SKILLS.md` | Agent skills registry and invocation guide |
| `pyproject.toml` | Canonical dependency manifest (`uv sync`) |
| `requirements.txt` | Pip-compatible mirror of base `[project.dependencies]` |

## Goals & Plans

| Goal | Title | Plan |
|------|-------|------|
| 1 | Docs & Notebook Reformatting | [PLAN_GOAL1_docs_reformat.md](PLAN_GOAL1_docs_reformat.md) |
| 2 | Agripest / Insect Midgut Multi-Modal Model | [PLAN_GOAL2_midgut_model.md](PLAN_GOAL2_midgut_model.md) |
| 3 | Public Data Curation & Drosophila Foundation Assets | [PLAN_GOAL3_dmel_data_curation.md](PLAN_GOAL3_dmel_data_curation.md) |
| 4 | Cloud Infrastructure (Terraform + Service Catalog) | [PLAN_GOAL4_terraform_infra.md](PLAN_GOAL4_terraform_infra.md) |
| 5 | Orchestration & Oversight | [PLAN_GOAL5_oversight.md](PLAN_GOAL5_oversight.md) |

See also [SKILLS.md](SKILLS.md) for the agent roster and skill dependency map.

## Contributing

- **Python:** Black formatting, type hints, Google-style docstrings, `isort` import ordering
- **Notebooks:** `nbstripout` pre-commit hook; never commit cell outputs; every code cell preceded by a Markdown explainer cell with a compute-load table
- **Branches:** `feat/goal-N-<slug>` per goal; PRs target `main` with title `[Goal N] <description>`

```bash
# One-time: enable output stripping on commit
uv run pre-commit install
```
