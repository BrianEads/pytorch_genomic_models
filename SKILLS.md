# SKILLS.md — Agent Skills & Capabilities Reference

---

## Introduction

This document maps each project goal to its dedicated agent, the skills that agent must exercise, and the conventions shared across all goals. It serves as the canonical reference for anyone (human or automated agent) who needs to understand the capability requirements of this project before invoking an agent or reviewing a pull request.

A **skill** in this context is a discrete, testable capability: a combination of domain knowledge, tooling proficiency, and repeatable process that an agent can execute independently given the corresponding `PLAN_GOAL*.md` as input. Skills may span multiple goals; the dependency map in a later section clarifies ordering constraints.

---

## Skills Registry

| Skill name | Description | Relevant tools / libraries | Used by goal(s) | Example task |
|-----------|-------------|---------------------------|-----------------|-------------|
| `code-formatting` | Apply consistent Markdown fencing, heading hierarchy, and code style to documentation | `markdownlint`, `Black`, `pre-commit` | Goal 1 | Rename `.mdmd` → `.md`; add triple-backtick fences to all code blocks |
| `notebook-editing` | Insert, delete, and reorder cells in `.ipynb` JSON; validate notebook structure | `nbformat`, `nbstripout` | Goal 1 | Add Markdown explainer cells before every code cell in `e2e_explorer.ipynb` |
| `documentation-writing` | Write narrative explainer prose, compute-load tables, and biological context sections | Plain text + Markdown | Goal 1, Goal 2 | Write per-cell compute notes with wall-clock estimates and bottleneck analysis |
| `genomics-tokenization` | Convert genomic sequences (FASTA, VCF) into integer token tensors using k-mer, BPE, or one-hot encodings | `biopython`, `pysam`, `h5py`, `pyranges` | Goal 3 | Slide k=6 window over dm6 FASTA; write k-mer HDF5 with chromosome/position metadata |
| `bioinformatics-data-qc` | Apply per-modality QC filters: sample outlier removal, MAF filtering, confidence score thresholds | `scanpy`, `scrublet`, `pysam`, `pandas` | Goal 3 | Remove scRNA-seq cells with > 20% mitochondrial reads; filter VCF to PASS biallelic SNPs with MAF > 0.01 |
| `graph-dataset-construction` | Build PyTorch Geometric graph datasets from PPI edge lists; assign node features | `torch_geometric`, `pandas`, `h5py` | Goal 3 | Convert DroID edge TSV to PyG `Data` object with ESM-2 node features |
| `foundation-model-pretraining` | Design and run masked language model pre-training on genomic sequences or expression data | `torch`, `transformers`, `mlflow` | Goal 3, Goal 2 | Pre-train DNABERT-style k-mer MLM on dm6 genome windows |
| `protein-modeling` | Fine-tune ESM-2 on insecticidal protein families; annotate binding domains | `fair-esm`, `peft`, `biopython` | Goal 2 | LoRA fine-tune ESM-2-650M on CRY/Vip sequences; evaluate Pfam domain recall |
| `genomics-data-ingestion` | Process raw FASTQ/BAM into count matrices; integrate modENCODE and GEO datasets | Cell Ranger, STARsolo, `scanpy` | Goal 2, Goal 3 | Run STARsolo on modENCODE RNA-seq FASTQs; produce per-gene TPM matrix |
| `multi-modal-fusion` | Design and train cross-attention and late-fusion heads that combine embeddings from multiple towers | `torch`, `transformers` | Goal 2 | Implement `CrossAttentionFusionHead` that takes 5 CLS tokens and returns a fused embedding |
| `experiment-tracking` | Log hyperparameters, metrics, and artefacts to MLflow or W&B; compare runs | `mlflow`, `wandb`, `optuna` | Goal 2 | Set up MLflow experiment with per-epoch loss, AUROC, and checkpoint artefacts |
| `terraform-aws` | Write Terraform modules that wrap AWS Service Catalog products via SSM data sources; manage S3, EFS, IAM, CloudWatch, EC2 ImageBuilder within the permission boundary | `terraform` ≥ 1.5, AWS provider ~5.0, `tflint`, `checkov` | Goal 4 | Implement `storage` and `efs_mount` modules using VPC/subnet IDs read from Service Catalog SSM parameters |
| `cloud-cost-optimization` | Select instance types, configure ParallelCluster spot queues, set budget alerts, implement idle auto-stop and scale-in | AWS Cost Explorer, CloudWatch, ParallelCluster `scaledown_idletime` | Goal 4 | Configure CloudWatch billing alarm at $200/month + ParallelCluster 10-minute idle scale-in |
| `devops-bootstrap` | Write idempotent node startup scripts that mount EFS, sync S3 datasets, and activate the conda env; complements ImageBuilder AMI baking | `bash`, AWS CLI, `amazon-efs-utils` | Goal 4 | Write `on_node_start.sh` ParallelCluster custom action that mounts EFS and syncs datasets on boot |
| `parallelcluster-ops` | Configure, deploy, and operate AWS ParallelCluster for multi-node GPU training; define Slurm queues, custom AMI references, and shared storage | `aws-parallelcluster` CLI ≥ 3.7, Slurm, `torchrun` | Goal 4 | Write `cluster_config_prod.yaml` with `train` (spot p3/p4d) and `sweep` (spot job array) queues; submit DDP jobs via `sbatch` |
| `imagebuilder-ami` | Build versioned GPU AMIs using EC2 ImageBuilder pipelines; compose CUDA, Conda, PyTorch, and project dependency components; publish AMI ARN to SSM | EC2 ImageBuilder, Terraform `aws_imagebuilder_*` resources, SSM Parameter Store | Goal 4 | Create ImageBuilder pipeline that bakes CUDA 12.1 + pytorch-genomic conda env into a weekly-published AMI; store AMI ARN at `/bayer/pytorch-genomic/ami/gpu-training-latest` |
| `ddp-training` | Submit and monitor DistributedDataParallel training via Slurm + torchrun; handle spot interruption via Slurm requeue + checkpoint resume | `torch.distributed`, `torchrun`, Slurm `sbatch` | Goal 4, Goal 2 | Write `submit_training_job.sh` that issues `sbatch` with `--nodes` and `--gres=gpu:N`; training script checkpoints every 500 steps for safe requeue |
| `schema-design` | Write JSON Schema definitions and validation tooling for inter-component data contracts | `jsonschema`, Python stdlib `json` | Goal 3 | Write `dataset_manifest_schema.json` (draft-07) and `manifest_validate.py` CLI |

---

## Agent Roster

### `docs-reformatter`

| Field | Detail |
|-------|--------|
| **Goal** | Goal 1 — Docs & Notebook Reformatting |
| **Branch** | `feat/goal-1-docs-reformat` |
| **Primary skills** | `code-formatting`, `notebook-editing`, `documentation-writing` |
| **Secondary skills** | `genomics-tokenization` (for sequence chunk formatting standards) |
| **Inputs** | `docs/1_masked_lang_model.mdmd`, `docs/2_kmer_pretraining.md`, `docs/3_fine_tuning.md`, `docs/4_attention_deep_dive.md`, `e2e_explorer.ipynb`, `README.md` |
| **Outputs** | Renamed + reformatted docs (all `.md`), annotated `e2e_explorer.ipynb` with compute-load tables, reformatted `README.md` with Quick Start, `.pre-commit-config.yaml` with `nbstripout` hook |
| **Success criteria** | All docs pass `markdownlint`; notebook parses with `nbformat.validate()`; no `asc_slot://` artefacts remain; every code cell in the notebook is preceded by a Markdown explainer cell |

---

### `midgut-model-builder`

| Field | Detail |
|-------|--------|
| **Goal** | Goal 2 — Agripest / Insect Midgut Multi-Modal Model |
| **Branch** | `feat/goal-2-midgut-model` |
| **Primary skills** | `multi-modal-fusion`, `protein-modeling`, `experiment-tracking` |
| **Secondary skills** | `genomics-data-ingestion`, `ddp-training`, `documentation-writing` |
| **Inputs** | `PLAN_GOAL2_midgut_model.md`, pre-trained tower checkpoints (from Goal 3 or public sources), `data/manifests/*.json` |
| **Outputs** | `models/midgut_multimodal/` (towers, fusion head, downstream heads, configs), `scripts/midgut/` training scripts, `tests/test_midgut_towers.py` |
| **Success criteria** | All tower modules forward-pass without error on CPU with random input; fusion head produces correct output shape; unit tests pass; training script accepts `--config` and logs to MLflow |

---

### `dmel-data-curator`

| Field | Detail |
|-------|--------|
| **Goal** | Goal 3 — Public Data Curation & Drosophila Foundation Assets |
| **Branch** | `feat/goal-3-dmel-data-curation` |
| **Primary skills** | `genomics-tokenization`, `bioinformatics-data-qc`, `graph-dataset-construction`, `schema-design` |
| **Secondary skills** | `foundation-model-pretraining`, `documentation-writing` |
| **Inputs** | `PLAN_GOAL3_dmel_data_curation.md`, raw data staged by `data-fetch-wizard` to `data/raw/` |
| **Outputs** | `data/schemas/` (JSON schemas), `data/pipelines/` (QC + tokenisation scripts), `data/manifests/dmel_foundation_manifest.json`, `notebooks/dmel_foundation_pretraining.ipynb`, `tests/test_dmel_pipelines.py` |
| **Success criteria** | Manifest validates against schema; genome tokenisation pipeline runs on a synthetic FASTA in < 5 s; unit tests pass; split assigner produces non-overlapping, non-empty splits |

---

### `infra-provisioner`

| Field | Detail |
|-------|--------|
| **Goal** | Goal 4 — Cloud Infrastructure (Terraform + Service Catalog) |
| **Branch** | `feat/goal-4-terraform-infra` |
| **Primary skills** | `terraform-aws`, `parallelcluster-ops`, `imagebuilder-ami` |
| **Secondary skills** | `devops-bootstrap`, `cloud-cost-optimization`, `ddp-training`, `documentation-writing` |
| **Inputs** | `PLAN_GOAL4_terraform_infra.md`, confirmed SSM parameter paths from Bayer platform team, existing AWS account with appropriate IAM permission boundary |
| **Outputs** | `infra/terraform/` (Terraform modules wrapping Service Catalog), `infra/pcluster/` (ParallelCluster configs), `infra/imagebuilder/` (GPU AMI pipeline), `infra/scripts/` (Slurm job submission, teardown), `infra/README_infra.md` |
| **Success criteria** | `terraform validate` passes; `terraform fmt --check` passes; `tflint --recursive` no errors; `checkov` no HIGH/CRITICAL; `pcluster` config dry-run validates; `on_node_start.sh` is idempotent; ImageBuilder pipeline publishes AMI ARN to SSM on first run |

---

## Skill Dependency Map

Some skills produce artefacts that other skills depend on. Attempting to run a downstream skill before its upstream dependencies are complete will result in missing inputs or incompatible interfaces.

```
Goal 3 dependencies (must complete first):
  schema-design
    └─▶ genomics-tokenization      (tokenisation scripts write to paths defined in manifest schema)
    └─▶ bioinformatics-data-qc     (QC pipelines write to paths defined in manifest schema)
    └─▶ graph-dataset-construction (graph builder reads from schema-defined raw paths)

  genomics-tokenization
    └─▶ foundation-model-pretraining  (pre-training reads tokenised HDF5 output)

Goal 2 dependencies:
  foundation-model-pretraining  (from Goal 3)
    └─▶ multi-modal-fusion          (fusion head takes pre-trained tower checkpoints as input)

  protein-modeling
    └─▶ multi-modal-fusion          (ESM-2 tower checkpoint fed into fusion head)

  multi-modal-fusion
    └─▶ experiment-tracking         (fusion training must be logged; metrics guide downstream task selection)

Goal 4 dependencies:
  terraform-aws
    └─▶ imagebuilder-ami            (ImageBuilder pipeline is a Terraform resource; bucket + IAM must exist first)
    └─▶ parallelcluster-ops         (cluster references EFS filesystem ID and custom AMI from Terraform outputs)
    └─▶ cloud-cost-optimization     (CloudWatch alarms attach to provisioned resources)

  imagebuilder-ami
    └─▶ devops-bootstrap            (on_node_start.sh is the runtime complement to the baked AMI)
    └─▶ parallelcluster-ops         (cluster config references AMI ARN from ImageBuilder SSM parameter)

  parallelcluster-ops
    └─▶ ddp-training                (DDP Slurm job scripts depend on the cluster being operational)

Cross-goal dependencies:
  Goal 3 (foundation-model-pretraining output)
    └─▶ Goal 2 (multi-modal-fusion input)

  Goal 4 (parallelcluster-ops: running cluster)
    └─▶ Goal 2, Goal 3 (actual large-scale training runs)
```

**Recommended execution order:**
1. Goal 1 (no dependencies; improves readability before other work begins)
2. Goal 4 (no code dependencies; can run in parallel with Goal 3)
3. Goal 3 (data curation; required before large-scale Goal 2 training)
4. Goal 2 (model; depends on Goal 3 artefacts and Goal 4 infrastructure)

---

## Shared Conventions

All goals and agents must adhere to the following conventions.

### Python coding standards
- **Formatter:** `Black` (line length 88). Run `black .` before every commit.
- **Type hints:** All public functions and methods must have type annotations.
- **Docstrings:** Google style (Args / Returns / Raises / Example sections).
- **Imports:** `isort` ordering (stdlib → third-party → local).

### Notebook conventions
- `nbstripout` pre-commit hook must be active; never commit notebooks with cell outputs.
- Every code cell must be preceded by a Markdown explainer cell.
- Compute-load callout table (⚙️ Compute note) required for any cell that is non-trivial computationally.

### Terraform conventions
- `terraform fmt` must pass before every commit touching `infra/`.
- `tflint` must pass with no errors.
- `checkov` must report no HIGH or CRITICAL findings.
- All Terraform modules **read** VPC, subnet, and endpoint IDs from Service Catalog SSM parameters — never hardcode or re-create platform-managed resources.
- All resources must carry the `Project`, `Environment`, `Owner`, and `Goal` tags.

### Secrets management
- **Never commit secrets.** Credentials, API keys, and AWS keys must be stored in AWS Secrets Manager or in `.env` files that are `.gitignore`-d.
- Scan changed files with a secrets scanner (e.g., `detect-secrets`) before every commit.

### `DatasetManifest` contract
- All data exchange between `data-fetch-wizard` and this repo must go through a validated `DatasetManifest` JSON file.
- The canonical schema is `data/schemas/dataset_manifest_schema.json`.
- No pipeline script may assume a hardcoded path — all paths must be read from the manifest.
- See `PLAN_GOAL3_dmel_data_curation.md` for the full schema definition.

### Branch naming
- `feat/goal-1-docs-reformat`
- `feat/goal-2-midgut-model`
- `feat/goal-3-dmel-data-curation`
- `feat/goal-4-terraform-infra`

### PR conventions
- PR title: `[Goal N] <short description>`
- PR body: link to the relevant `PLAN_GOAL*.md`; checklist of acceptance criteria from that file.
- All PRs target `main`.

---

## How to Invoke an Agent

Use the following template to create a Copilot agent request for any goal. Fill in the bracketed fields.

```
@Copilot Please work on [GOAL TITLE].

Context:
- Read `[PLAN_GOAL_FILE].md` at the repo root for full requirements, acceptance criteria, and step-by-step agent instructions.
- Read `SKILLS.md` for shared conventions and skill dependency ordering.
- Branch: `[BRANCH_NAME]`

Your task:
[One paragraph summarising the specific deliverable — e.g., "Implement the per-modality tower stubs for the midgut multi-modal model and write unit tests verifying forward-pass shapes."]

Acceptance criteria (from plan file):
[Copy the numbered acceptance criteria list from the relevant PLAN_GOAL*.md]

Do not proceed beyond the steps in the agent instructions section of the plan file without checking back.
```

### Example — invoking `docs-reformatter`

```
@Copilot Please work on Goal 1 — Docs & Notebook Reformatting.

Context:
- Read `PLAN_GOAL1_docs_reformat.md` at the repo root for full requirements, acceptance criteria, and step-by-step agent instructions.
- Read `SKILLS.md` for shared conventions.
- Branch: `feat/goal-1-docs-reformat`

Your task:
Rename `docs/1_masked_lang_model.mdmd` to `docs/1_masked_lang_model.md`, add proper Markdown fencing to all code blocks across all four docs files, fix all broken artefacts in `e2e_explorer.ipynb`, add Markdown explainer cells with compute-load tables before every code cell, and reformat `README.md` with headers and a Quick Start section.

Acceptance criteria:
1. All docs use proper `.md` extension.
2. Every code block is fenced with triple backticks and a language specifier.
3. DNA/RNA sequence examples use the position-annotated `text` fenced format.
4. e2e_explorer.ipynb has a Markdown cell before every code cell.
5. All broken torch.tensor() calls are fixed.
6. All asc_slot:// artefacts are removed.
7. README.md has H1 title, H2 sections, Quick Start, and repository layout.
8. nbstripout pre-commit hook is configured.
```

### Example — invoking `infra-provisioner`

```
@Copilot Please work on Goal 4 — Cloud Infrastructure (Terraform + Service Catalog).

Context:
- Read `PLAN_GOAL4_terraform_infra.md` at the repo root for full requirements and step-by-step agent instructions.
- Read `SKILLS.md` for shared conventions.
- Branch: `feat/goal-4-terraform-infra`

Your task:
Create the full Terraform module directory tree under `infra/`, implementing the storage, EFS, ImageBuilder, and monitoring modules using VPC and subnet IDs read from Service Catalog SSM parameters. Write the ParallelCluster configuration files (dev and prod variants with appropriate Slurm queue definitions) and the Slurm job submission scripts. All Terraform must pass `terraform validate`, `terraform fmt --check`, `tflint`, and `checkov` with no HIGH/CRITICAL findings. The ParallelCluster config must pass `pcluster` dry-run validation.

Acceptance criteria:
(copy from PLAN_GOAL4_terraform_infra.md Step 13 and Step 14 section)
```
