# Goal 5 — Orchestration & Oversight

**Branch:** `feat/goal-5-oversight`
**Agent:** `conductor`

---

## Goal Summary

Maintain program-level coordination across Goals 1–4: track progress, consolidate decisions, manage pause-points and milestones, and keep agent skills and living checklists current. The conductor does **not** implement feature code in other goals' directories unless explicitly asked; it edits planning artefacts (`MASTER_PLAN.md`, `PLAN_GOAL*.md`, `SKILLS.md`, `DECISIONS.md`, `skills/`) and runs cross-goal status reviews.

After agent skill files are reviewed by the user, goal agents may be invoked against their individual plan files. A major external milestone is **DataFetch-Wizard (DFW) alpha evaluation and AWS account provisioning** — orchestrator directions must clearly separate work that can proceed **before** vs **after** that gate.

---

## Role of the Conductor Agent

| Responsibility | Detail |
|----------------|--------|
| **Status board** | Update the running status board below after each goal-agent session |
| **Checklists** | Keep per-goal checkboxes aligned with actual repo state (no invented progress) |
| **Decisions registry** | Record resolved decisions in `DECISIONS.md`; escalate open items to the user |
| **Pause-point enforcement** | Block or redirect agents from DFW `terraform apply`, S3 staging, and cluster training until gates clear |
| **Cross-goal dependencies** | Ensure Goal 3 manifest changes do not break Goal 2 loaders; Goal 4 outputs match Goal 3 S3 layout |
| **Skill maintenance** | Create/update `skills/goal-*/SKILL.md` when plan acceptance criteria or boundaries change |
| **Feedback loop** | Summarise blockers for the user; translate user decisions into plan/skill updates |

### What the conductor is **not** responsible for

- Raw data download or API calls (`data-fetch-wizard` owns those)
- Day-to-day tower/pipeline/infra implementation (Goals 1–4 agents)
- Applying Terraform to the DFW AWS account without explicit user approval

---

## Program Milestones & Gates

| ID | Milestone | Gate / exit criteria | Status |
|----|-----------|----------------------|--------|
| **M0** | Scaffolding | Repo layout, plans, skills, stubs on feature branches; local `pytest` passes on CPU | ✅ Complete |
| **M1** | Local validation | Unit tests green; docs/notebook readable; pipelines runnable on synthetic/local fixtures | 🟡 In progress |
| **M2** | DFW S3 staging | DFW alpha live; raw files in `s3://cs-cp-bifx-dfw-pytorch-genomic-data/raw/`; user approves AWS apply | ⏸ **PAUSE** — user not ready |
| **M3** | Tokenised data | Goal 3 pipelines run on staged raw data; split indices written; manifest validates | ⬜ Blocked on M2 |
| **M4** | Cluster training | Terraform applied; ParallelCluster up; ImageBuilder AMI published; Goal 2 tower pre-training on GPU | ⬜ Blocked on M2–M3 |
| **M5** | Alpha evaluation | End-to-end DFW → curation → training smoke test; ODM/GeneStack accession workflow exercised | ⬜ Blocked on M2 |

### Current program phase

**Phase: M1 — Local validation** (pre-DFW gate)

Work permitted now: docs polish, local tests, pipeline dry-runs on synthetic FASTA/TSV, model forward-pass tests, infra lint/validate (no apply), dataset research, notebook exploration in a separable local environment.

Work **paused** until user clears M2 gate: `terraform apply`, DFW recipe execution against live S3, ParallelCluster create, large-scale GPU training.

---

## Pause-Points

| ID | Pause-point | Owner | Trigger to resume |
|----|-------------|-------|-------------------|
| **P0** | **DFW AWS account apply** | User | User confirms readiness to provision/apply in the new DFW dev/testing AWS account |
| **P1** | **DFW alpha evaluation** | DFW / User | Alpha deployment validated; fetch recipes run successfully; S3 layout matches manifest |
| **P2** | **Internal data access** | User / platform | Decisions on microbial toxin DB, cell painting, single-cell internal datasets |
| **P3** | **Prod Bayer account infra** | Platform team | Service Catalog SSM paths confirmed; permission boundary reviewed for prod `terraform apply` |

> **P0 is the primary gate.** Until cleared, Goal 4 agents must stop at `terraform plan` / lint; Goal 3 agents must not assume raw files exist in S3; Goal 2 agents must not expect real checkpoint training runs.

---

## Running Status Board

_Update this section after every goal-agent session. Copy the template if starting fresh._

**Last updated:** 2026-07-13
**Program phase:** M1 — Local validation
**Active gate:** P0 — DFW AWS apply (user not ready)

| Goal | Branch | Agent | Phase | Last action | Next action | Blocker |
|------|--------|-------|-------|-------------|-------------|---------|
| 1 | `feat/goal-1-docs-reformat` | `docs-reformatter` | ✅ Done (local) | Docs deduped, notebook fixed, uv quick start | Open PR when user ready | None |
| 2 | `feat/goal-2-midgut-model` | `midgut-model-builder` | 🟡 Skeleton complete | Towers, fusion, loaders, incremental configs | Wire MLflow smoke test; await M3 data | P0 |
| 3 | `feat/goal-3-dmel-data-curation` | `dmel-data-curator` | 🟡 v1 pipelines | Schema, 3 pipelines, manifest w/ DFW bucket | `split_assigner.py`; v2 pipelines | P0 for real data |
| 4 | `feat/goal-4-terraform-infra` | `infra-provisioner` | 🟡 Scaffold complete | Full infra tree, DFW account alignment | Lint/validate only until P0 clears | P0 |
| 5 | `feat/goal-5-oversight` | `conductor` | 🟡 Active | Plans + skills authored | User review of skills; track M2 gate | — |

---

## Per-Goal Checklists

Reflect **actual** progress from prior agent sessions (uncommitted work on feature branches).

### Goal 1 — Docs & Notebook Reformatting

**Branch:** `feat/goal-1-docs-reformat` · **Skill:** [skills/goal-1-docs-reformatter/SKILL.md](skills/goal-1-docs-reformatter/SKILL.md)

- [x] Rename `docs/1_masked_lang_model.mdmd` → `.md`
- [x] Fence code blocks across all four docs
- [x] Fix broken `torch.tensor()` calls in `e2e_explorer.ipynb`
- [x] Remove `asc_slot://` artefacts
- [x] Markdown explainer + compute note before each notebook code cell
- [x] Split readable walkthrough to `e2e_explorer.md` (notebook remains runnable companion)
- [x] Reformat `README.md` with Quick Start (`uv sync`)
- [x] Configure `nbstripout` pre-commit hook
- [x] Add `scripts/goal1_build_docs_notebook.py` for notebook regeneration
- [ ] Open PR to `main` and merge (waiting on user)
- [ ] Optional: run `markdownlint` across `docs/` and fix findings

**Acceptance criteria met locally.** Remaining work is PR/merge and optional lint pass.

---

### Goal 2 — Midgut Multi-Modal Model

**Branch:** `feat/goal-2-midgut-model` · **Skill:** [skills/goal-2-midgut-model/SKILL.md](skills/goal-2-midgut-model/SKILL.md)

- [x] Directory structure under `models/midgut_multimodal/`
- [x] Per-modality tower stubs (scRNA, cell painting, popgen, Bt screening, protein ESM-2, genome k-mer, PPI graph)
- [x] `CrossAttentionFusion` + `LateFusionHead` (partial modality support)
- [x] Downstream heads (IC50, resistance, Cry scorer, perturbation)
- [x] Training configs including incremental v1 (`fusion_train_v1.yaml`, `pretrain_*`)
- [x] Training scripts (`train_tower.py`, `train_fusion.py`, `eval_downstream.py`)
- [x] Manifest-driven loaders (`data/manifest.py`, `factory.py`, `datasets.py`)
- [x] Midgut tissue filter on bulk RNA-seq loader
- [x] PPI node feature fill via ESM-2 (Goal 2 owns ESM-2)
- [x] `INCREMENTAL_TRAINING.md` documenting v1 modality path
- [x] Unit tests (`tests/test_midgut_towers.py`, `tests/test_manifest_loaders.py`)
- [ ] End-to-end MLflow logging smoke test on local synthetic batch
- [ ] Gene ID → protein FASTA mapping script for PPI node fill
- [ ] Real pre-training runs (blocked: **P0 / M3**)
- [ ] Open PR to `main`

---

### Goal 3 — Dmel Data Curation

**Branch:** `feat/goal-3-dmel-data-curation` · **Skill:** [skills/goal-3-dmel-curator/SKILL.md](skills/goal-3-dmel-curator/SKILL.md)

- [x] `dataset_manifest_schema.json`
- [x] `rnaseq_sample_schema.json`
- [x] `manifest_validate.py` CLI
- [x] v1 pipelines: `dmel_genome_tokenize.py`, `modencode_rnaseq_qc.py`, `flybase_ppi_graph.py`
- [x] Example manifest `dmel_foundation_manifest.json` with DFW S3 bucket/prefixes
- [x] `pyproject.toml` / `uv` extras for curation and midgut deps
- [x] `data/README.md` documenting DFW boundary
- [ ] `split_assigner.py` (chromosome / sample / line holdouts)
- [ ] `dgrp2_vcf_encode.py` pipeline
- [ ] `fca_scrna_qc.py` pipeline
- [ ] `vcf_window_schema.json`, `ppi_graph_schema.json`
- [ ] `notebooks/dmel_foundation_pretraining.ipynb`
- [ ] `tests/test_dmel_pipelines.py` (pipeline integration tests)
- [ ] Run v1 pipelines on DFW-staged raw data (blocked: **P0 / M2**)
- [ ] Open PR to `main`

---

### Goal 4 — Cloud Infrastructure

**Branch:** `feat/goal-4-terraform-infra` · **Skill:** [skills/goal-4-infra-provisioner/SKILL.md](skills/goal-4-infra-provisioner/SKILL.md)

- [x] Terraform module tree (`storage`, `efs_mount`, `imagebuilder`, `monitoring`)
- [x] Dev/prod env layouts under `infra/terraform/envs/`
- [x] Service Catalog SSM data source pattern in `data_sources.tf`
- [x] ParallelCluster configs (dev/prod) + custom actions
- [x] ImageBuilder component YAML (CUDA, conda, PyTorch, project deps)
- [x] Slurm submission scripts + teardown script
- [x] `README_infra.md` with DFW account + S3 layout alignment
- [ ] Confirm live SSM parameter paths against DFW account
- [ ] Run `terraform fmt`, `tflint`, `checkov` clean (local CI pass)
- [ ] `terraform apply` dev environment (blocked: **P0** — user not ready)
- [ ] ImageBuilder first AMI publish to SSM
- [ ] ParallelCluster cluster create + head-node EFS/S3 sync smoke test
- [ ] Open PR to `main`

---

### Goal 5 — Orchestration & Oversight

**Branch:** `feat/goal-5-oversight` · **Skill:** [skills/goal-5-orchestrator/SKILL.md](skills/goal-5-orchestrator/SKILL.md)

- [x] Complete this plan with milestones, pause-points, registries
- [x] Update `MASTER_PLAN.md` with Goal 5 and phase indicator
- [x] Create `skills/goal-*/SKILL.md` per-goal agent instructions
- [x] Update `SKILLS.md` master index
- [x] Create `DECISIONS.md` decision log
- [x] User review and edits to agent skill files
- [ ] Invoke goal agents against approved skills (post-review)
- [ ] Track M2 gate clearance and update status board

---

## Cross-Goal Dependency Tracking

```
Goal 1 (docs) ──────────────────────────────▶ no downstream blocker

Goal 4 (infra scaffold) ──P0 gate──▶ S3/EFS/cluster ──▶ Goal 2 GPU training
         ▲                                              ▲
         │ DFW bucket layout                            │
Goal 3 (curation) ──M2 raw staging──▶ tokenised HDF5 ───┘
         │
         └── manifest schema ──▶ Goal 2 loaders (incremental v1)

DFW (external) ──P0/P1──▶ raw S3 ──▶ Goal 3 pipelines ──▶ Goal 2 training
```

| Dependency | Upstream | Downstream | Status |
|------------|----------|------------|--------|
| Manifest schema | Goal 3 | Goal 2 loaders | ✅ Wired |
| v1 modalities (genome, rnaseq, ppi) | Goal 3 | Goal 2 towers | ✅ Scaffolded; data pending M2 |
| ESM-2 node features for PPI | Goal 2 | Goal 3 graph output (`x=None`) | ✅ Design locked |
| S3 bucket/prefixes | DFW + Goal 4 | Goal 3, Goal 2 | ✅ Named; apply paused |
| GPU cluster + AMI | Goal 4 | Goal 2, Goal 3 notebook | ⬜ Blocked P0 |
| Docs/notebook clarity | Goal 1 | All agents (onboarding) | ✅ Locally complete |

---

## Decisions Registry

Consolidated from all goals. Canonical log: [DECISIONS.md](DECISIONS.md).

| ID | Decision | Status | Source |
|----|----------|--------|--------|
| D1 | Incremental v1 manifest ships `genome`, `rnaseq`, `ppi` first | ✅ Resolved | Goal 2/3 |
| D2 | Goal 2 owns ESM-2; Goal 3 PPI graph writes `x=None` | ✅ Resolved | Goal 2/3 |
| D3 | Midgut tissue filter default on bulk RNA-seq loader | ✅ Resolved | Goal 2 |
| D4 | `uv` + `pyproject.toml` as canonical dependency manifest | ✅ Resolved | Goal 3 |
| D5 | DFW dev account + bucket `cs-cp-bifx-dfw-pytorch-genomic-data` | ✅ Resolved | Goal 3/4 |
| D6 | Late fusion (Option A) first; cross-attention (Option B) also implemented | ✅ Resolved | Goal 2 |
| D7 | Secret/internal data stays out of model weights; private retrieval/scoring only | ✅ Resolved | User |
| D8 | Terraform wraps Service Catalog; no raw VPC creation | ✅ Resolved | Goal 4 |
| D9 | **DFW AWS apply paused** until user ready | ✅ Resolved | User |
| D10 | Internal toxin DB / cell painting / scRNA access TBD | ⬜ Open | User |
| D11 | Fusion strategy MoE (Option C) deferred | ⬜ Open | Goal 2 |
| D12 | Prod vs dev account separation (DFW dev first) | ✅ Resolved | Goal 4 |

---

## Open Questions Registry

| ID | Question | Goal | Severity | Owner | Status |
|----|----------|------|----------|-------|--------|
| Q1 | When will user clear P0 (DFW AWS apply)? | 5 | High | User | ⬜ Open |
| Q2 | Access to internal microbial toxin protein DB? | 2 | High | User | ⬜ Open |
| Q3 | Access to internal cell painting / scRNA datasets? | 2, 3 | High | User | ⬜ Open |
| Q4 | Public insect Cell Painting data availability | 2 | High | Agent research | ⬜ Open |
| Q5 | Live SSM paths for Service Catalog networking product | 4 | Medium | Platform | ⬜ Open (partial answers in PLAN_GOAL4) |
| Q6 | ODM/GeneStack accession workflow for first DFW experiment | 5 | Medium | DFW / User | ⬜ Open |
| Q7 | S3 Files (EFS-backed) vs classic EFS for workflows | 4 | Low | Agent research | ⬜ Open |
| Q8 | Notebook as separable module — packaging for external env | 1, 5 | Low | User | ⬜ Open |
| Q9 | Bt outcome / resistance literature scraping effort scope | 2 | Medium | User | ⬜ Open |
| Q10 | ESM-2 length limit for full-length CRY proteins | 2 | Medium | Agent | ⬜ Open (mitigation documented) |

---

## Blocker Registry

| ID | Blocker | Affected goals | Owner | Resolution path |
|----|---------|----------------|-------|-----------------|
| B1 | **User not ready for DFW AWS apply** | 3, 4, 2 | User | User confirms P0 clearance; then Goal 4 `terraform apply` + DFW recipe run |
| B2 | No raw data in S3 yet | 3, 2 | DFW | DFW alpha + M2 milestone |
| B3 | `split_assigner.py` not implemented | 3, 2 | Agent | Goal 3 agent task (local, no AWS) |
| B4 | v2 modalities (VCF, scRNA) pipelines missing | 3 | Agent | Goal 3 agent after v1 stable |
| B5 | PRs not merged to `main` | All | User | Review uncommitted feature-branch work |
| B6 | Internal dataset access undecided | 2 | User | Q2, Q3 decisions |

---

## Feedback Loop Protocol

### User → Conductor

1. User updates pause-points (especially P0) or answers open questions.
2. Conductor records decisions in `DECISIONS.md` and updates this file's registries.
3. Conductor updates affected `skills/goal-*/SKILL.md` pause/escalate sections.

### Conductor → Goal agents

1. Before invoking a goal agent, conductor confirms program phase and active gates.
2. Invocation template: see [SKILLS.md → How to invoke an agent](SKILLS.md#how-to-invoke-an-agent).
3. Agent must read its skill file + plan file; must not cross boundary into DFW download code or `terraform apply` when P0 active.

### Goal agents → Conductor

After each session, agent (or user) reports:

- Checklist items completed / blocked
- New open questions or blockers
- Files changed on branch

Conductor updates **Running Status Board** and per-goal checklists above.

### Escalation triggers (agents must stop and report)

- Attempt to `terraform apply` while P0 active
- Need to add download/API logic to this repo (violates Goal 3 boundary)
- Manifest schema change that breaks Goal 2 loaders
- Missing Service Catalog permission for IAM action
- User data-access decision required (Q2, Q3)

---

## Agent Instructions — `conductor`

1. Read all `PLAN_GOAL*.md` files and skim current repo state on feature branches.
2. Keep this document's status board and checklists accurate — do not mark items complete without evidence in the repo.
3. Maintain `DECISIONS.md`, `MASTER_PLAN.md`, `SKILLS.md`, and `skills/goal-*/SKILL.md`.
4. When user clears P0, update phase to M2 and notify Goal 4 + DFW coordination paths in status board.
5. Do not commit unless the user explicitly requests it.

---

## Related Files

| File | Purpose |
|------|---------|
| [MASTER_PLAN.md](MASTER_PLAN.md) | Program index, milestone timeline, phase indicator |
| [SKILLS.md](SKILLS.md) | Skills registry and invocation templates |
| [DECISIONS.md](DECISIONS.md) | Canonical decision log |
| [skills/goal-5-orchestrator/SKILL.md](skills/goal-5-orchestrator/SKILL.md) | Conductor agent skill |
