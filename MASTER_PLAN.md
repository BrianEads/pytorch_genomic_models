# Master Plan: PyTorch Genomic Models

> **Status:** Active program — operational tracking lives in [PLAN_GOAL5_oversight.md](PLAN_GOAL5_oversight.md).
>
> **Current phase:** **M1 — Local validation** (pre-DFW gate)
>
> **Active pause-point:** **P0** — DFW AWS account `terraform apply` (user not ready yet)

This project builds PyTorch-based genomic and multi-modal deep learning models for agricultural pest control research, with *Drosophila melanogaster* public data as the primary pre-training foundation. Five parallel goals drive this stage of work; Goals 1–4 implement features on dedicated branches, while Goal 5 (conductor) maintains oversight, skills, and living checklists.

---

## Goals Overview

| Goal | Title | Branch | Agent | Plan file | Skill file | Status |
|------|-------|--------|-------|-----------|------------|--------|
| 1 | Docs & Notebook Reformatting | `feat/goal-1-docs-reformat` | `docs-reformatter` | [PLAN_GOAL1](PLAN_GOAL1_docs_reformat.md) | [goal-1-docs-reformatter](skills/goal-1-docs-reformatter/SKILL.md) | 🟢 Complete (local) |
| 2 | Agripest / Insect Midgut Multi-Modal Model | `feat/goal-2-midgut-model` | `midgut-model-builder` | [PLAN_GOAL2](PLAN_GOAL2_midgut_model.md) | [goal-2-midgut-model](skills/goal-2-midgut-model/SKILL.md) | 🟡 Skeleton done |
| 3 | Public Data Curation & Dmel Foundation | `feat/goal-3-dmel-data-curation` | `dmel-data-curator` | [PLAN_GOAL3](PLAN_GOAL3_dmel_data_curation.md) | [goal-3-dmel-curator](skills/goal-3-dmel-curator/SKILL.md) | 🟡 v1 pipelines |
| 4 | Cloud Infrastructure (Terraform + SC) | `feat/goal-4-terraform-infra` | `infra-provisioner` | [PLAN_GOAL4](PLAN_GOAL4_terraform_infra.md) | [goal-4-infra-provisioner](skills/goal-4-infra-provisioner/SKILL.md) | 🟡 Scaffold; apply paused |
| 5 | Orchestration & Oversight | `feat/goal-5-oversight` | `conductor` | [PLAN_GOAL5](PLAN_GOAL5_oversight.md) | [goal-5-orchestrator](skills/goal-5-orchestrator/SKILL.md) | 🟡 Active |

**Operational tracking:** status board, per-goal checklists, blockers, and open questions → [PLAN_GOAL5_oversight.md](PLAN_GOAL5_oversight.md)

---

## Milestone Timeline

| Milestone | Description | Gate | Status |
|-----------|-------------|------|--------|
| **M0** | Scaffolding — plans, stubs, tests on feature branches | Local `pytest` passes | ✅ |
| **M1** | Local validation — docs, unit tests, synthetic pipeline runs | Acceptance criteria met locally | 🟡 **Current** |
| **M2** | DFW S3 staging — raw data in DFW bucket via data-fetch-wizard | **P0:** user approves AWS apply + DFW alpha | ⏸ Paused |
| **M3** | Tokenised data — Goal 3 pipelines on staged raw; splits assigned | Manifest validates; HDF5/PT in `tokenised/` | ⬜ |
| **M4** | Cluster training — ParallelCluster + GPU tower pre-training | Terraform applied; AMI published | ⬜ |
| **M5** | Alpha evaluation — DFW → curation → training smoke; ODM workflow | End-to-end accession experiment | ⬜ |

```text
M0 ──▶ M1 ──▶ [P0 PAUSE] ──▶ M2 ──▶ M3 ──▶ M4 ──▶ M5
              ▲
              └── User not ready for DFW AWS apply yet
```

---

## Skills & Agent Reference

| Resource | Purpose |
|----------|---------|
| [SKILLS.md](SKILLS.md) | Master skills registry, dependency map, invocation templates |
| [skills/](skills/) | Per-goal Cursor agent skill files (`goal-N-*/SKILL.md`) |
| [DECISIONS.md](DECISIONS.md) | Cross-cutting decision log |

Each skill file includes: scope, branch, acceptance checklist, coordination rules, pause/escalate triggers, status board format, and dependencies.

---

## Cross-Cutting Conventions

- **Python:** Black formatting, type hints, Google-style docstrings, `isort` import ordering
- **Dependencies:** `pyproject.toml` + `uv sync` (canonical); `requirements.txt` mirrors base deps
- **Notebooks:** `nbstripout` pre-commit hook; never commit cell outputs; Markdown explainer + compute note before each code cell
- **Terraform:** `terraform fmt` + `tflint` + `checkov` before every commit to `infra/`; wraps AWS Service Catalog products
- **Networking:** Private subnets only; VPC endpoints for S3/SSM/ECR/CloudWatch
- **Compute:** AWS ParallelCluster for multi-node GPU training; EC2 ImageBuilder for versioned GPU AMIs
- **Secrets:** Never committed; AWS Secrets Manager or `.env` (gitignored)
- **Tagging:** All AWS resources tagged with `Project`, `Environment`, `Owner`, `Goal`
- **Branches:** `feat/goal-N-<slug>` per goal; PRs target `main`
- **PR format:** Title `[Goal N] <description>`; body links to `PLAN_GOAL*.md` and lists acceptance criteria
- **DatasetManifest:** All data exchange between `data-fetch-wizard` and this repo uses validated JSON — schema in `data/schemas/dataset_manifest_schema.json`
- **data-fetch-wizard boundary:** Download/API/caching in DFW; schema harmonisation, QC, tokenisation, splits here
- **Execution order:** Goal 1 → Goal 4 (parallel scaffold) → Goal 3 → Goal 2; Goal 5 coordinates throughout

---

## Resolved Program Decisions (summary)

See [DECISIONS.md](DECISIONS.md) for full log.

- Incremental v1 manifest: `genome`, `rnaseq`, `ppi` first
- Goal 2 owns ESM-2; Goal 3 PPI graph ships `x=None`
- Midgut tissue filter on bulk RNA-seq
- DFW dev bucket: `cs-cp-bifx-dfw-pytorch-genomic-data`
- **DFW AWS apply explicitly paused** until user ready (unlocks M2+ testing)

---

## Next Steps

1. **User:** Review agent skill files under `skills/`; edit before invoking goal agents.
2. **Goals 1–4:** Finish M1 local items (e.g. Goal 3 `split_assigner.py`, Goal 4 lint pass) — no AWS apply.
3. **When P0 clears:** Goal 4 `terraform apply` → DFW recipe run → Goal 3 on real raw data → Goal 2 GPU training.
4. **Conductor:** Update [PLAN_GOAL5_oversight.md](PLAN_GOAL5_oversight.md) status board after each agent session.

Agents should be invoked via [SKILLS.md → How to invoke an agent](SKILLS.md#how-to-invoke-an-agent), reading both the plan file and the matching skill file.
