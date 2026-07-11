# Master Plan: PyTorch Genomic Models

> **Status:** Index — detailed plans have been broken out into per-goal files. See the Goals table below.

This project builds a set of PyTorch-based genomic and multi-modal deep learning models oriented toward agricultural pest control research, with *Drosophila melanogaster* public data as the primary pre-training foundation. Four parallel goals drive the work, each owned by a dedicated agent operating on its own branch.

---

## Goals Overview

| Goal | Title | Branch | Agent | Plan file | Status |
|------|-------|--------|-------|-----------|--------|
| 1 | Docs & Notebook Reformatting | `feat/goal-1-docs-reformat` | `docs-reformatter` | [PLAN_GOAL1_docs_reformat.md](PLAN_GOAL1_docs_reformat.md) | 🟡 Planning |
| 2 | Agripest / Insect Midgut Multi-Modal Model | `feat/goal-2-midgut-model` | `midgut-model-builder` | [PLAN_GOAL2_midgut_model.md](PLAN_GOAL2_midgut_model.md) | 🟡 Planning |
| 3 | Public Data Curation & Drosophila Foundation Assets | `feat/goal-3-dmel-data-curation` | `dmel-data-curator` | [PLAN_GOAL3_dmel_data_curation.md](PLAN_GOAL3_dmel_data_curation.md) | 🟡 Planning |
| 4 | Cloud Infrastructure (Terraform + Service Catalog) | `feat/goal-4-terraform-infra` | `infra-provisioner` | [PLAN_GOAL4_terraform_infra.md](PLAN_GOAL4_terraform_infra.md) | 🟡 Planning |

---

## Skills & Agent Reference

See **[SKILLS.md](SKILLS.md)** for:
- The full skills registry (skill name, description, tools, goal mapping, example task)
- The agent roster (name, goal, primary/secondary skills, inputs, outputs, success criteria)
- The skill dependency map (which skills must complete before others)
- How to invoke an agent (template + examples)

---

## Cross-Cutting Conventions

- **Python:** Black formatting, type hints, Google-style docstrings, `isort` import ordering
- **Notebooks:** `nbstripout` pre-commit hook; never commit cell outputs; every code cell preceded by a Markdown explainer cell
- **Terraform:** `terraform fmt` + `tflint` + `checkov` before every commit to `infra/`; wraps AWS Service Catalog products rather than raw resources
- **Networking:** Private subnets only; VPC endpoints for S3/SSM/ECR/CloudWatch; Bayer network connectivity via VPN/Direct Connect peering
- **Compute:** AWS ParallelCluster for multi-node GPU training; EC2 ImageBuilder for versioned custom GPU AMIs
- **Secrets:** Never committed; use AWS Secrets Manager or `.env` (gitignored)
- **Tagging:** All AWS resources tagged with `Project`, `Environment`, `Owner`, `Goal`
- **Branches:** `feat/goal-N-<slug>` per goal; all PRs target `main`
- **PR format:** Title `[Goal N] <description>`; body links to relevant `PLAN_GOAL*.md` and lists acceptance criteria
- **DatasetManifest:** All data exchange between `data-fetch-wizard` and this repo goes through a validated `DatasetManifest` JSON — schema in `data/schemas/dataset_manifest_schema.json`
- **data-fetch-wizard boundary:** Download, API calls, and caching belong in `data-fetch-wizard`; schema harmonisation, QC, tokenisation, and splits belong here
- **Execution order:** Goal 1 → Goal 4 (parallel) → Goal 3 → Goal 2 (each depends on the previous)

---

## Next Steps

Agents should be invoked against the individual `PLAN_GOAL*.md` files, not against this index. Each plan file contains:

- Full requirements and context
- Acceptance criteria checklist
- Work breakdown table
- Step-by-step agent instructions

See [SKILLS.md → How to invoke an agent](SKILLS.md#how-to-invoke-an-agent) for the exact request template and worked examples.
