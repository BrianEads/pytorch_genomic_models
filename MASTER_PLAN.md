# Master Plan: PyTorch Genomic Models — Four-Goal Roadmap

> **Status:** Draft — intended to be broken into four dedicated agent-driven sub-plans.
> Each goal maps to a dedicated branch, agent, and skill set.
> Related effort: [`data-fetch-wizard`](#goal-3-public-data-curation--drosophila-foundation-assets) overlaps with Goal 3; see the overlap/boundary notes there.

---

## Table of Contents
1. [Goal 1 — Docs & Notebook Reformatting](#goal-1--docs--notebook-reformatting)
2. [Goal 2 — Agripest / Insect Midgut Model](#goal-2--agripest--insect-midgut-model)
3. [Goal 3 — Public Data Curation & Drosophila Foundation Assets](#goal-3--public-data-curation--drosophila-foundation-assets)
4. [Goal 4 — Cloud Infra (Terraform)](#goal-4--cloud-infra-terraform)
5. [Agent & Skill Delegation Map](#agent--skill-delegation-map)
6. [Cross-Cutting Conventions](#cross-cutting-conventions)

---

## Goal 1 — Docs & Notebook Reformatting

### Problem statement
The repository currently contains:
- `docs/1_masked_lang_model.mdmd` — a **non-standard `.mdmd` extension** and the file body lacks proper Markdown fencing (all code blocks use bare `python` instead of triple-backtick fences, narrative text is run together with no section breaks, and sequence chunk examples are not highlighted).
- `docs/2_kmer_pretraining.md`, `docs/3_fine_tuning.md`, `docs/4_attention_deep_dive.md` — similar formatting issues (need audit).
- `e2e_explorer.ipynb` — the notebook has broken tensor literals (e.g., `torch.tensor()` with no content, inline slot references like `[...](asc_slot://start-slot-14)` that are rendering artefacts, and no Markdown explainer cells between code sections).

### Acceptance criteria
- [ ] All docs use proper `.md` extension (rename `.mdmd` → `.md`).
- [ ] Every code block is properly fenced with ` ```python ` … ` ``` `.
- [ ] DNA/RNA sequence examples use a `text` or `plaintext` fence with position numbers clearly annotated, e.g.:
  ```text
  pos:  1  2  3  4  5  6  ...
  seq:  A  T  G  C  A  G  ...
  ```
- [ ] k-mer chunking diagrams show the sliding window visually (ASCII or inline table).
- [ ] `e2e_explorer.ipynb` has a **Markdown cell before every code cell** explaining: (a) what the step does biologically, (b) expected input/output shapes, (c) compute load notes (see template below).
- [ ] Broken `torch.tensor()` calls and `asc_slot` artefacts are fixed with minimal working examples.
- [ ] `README.md` is reformatted as proper Markdown with headers, bullet lists, and a "Quick Start" section.

### Compute-load / bottleneck explainer template (for notebook cells)
Each section of the notebook should include a callout like:

```markdown
> **⚙️ Compute note**
> | Step | Typical wall-clock (CPU / single GPU) | Memory footprint | Bottleneck |
> |------|---------------------------------------|------------------|------------|
> | k-mer tokenisation (1 M bp seq) | ~2 s / ~0.3 s | ~50 MB | CPU-bound string ops |
> | Embedding lookup (batch 32, len 512) | ~1 ms / ~0.1 ms | ~8 MB | Negligible |
> | Conv1d motif scan | ~5 ms / ~0.5 ms | ~20 MB | Memory bandwidth |
> | Transformer encoder (2 layers) | ~200 ms / ~5 ms | ~500 MB | Attention O(n²) |
> | MLM pre-training epoch (100 k seqs) | ~4 h / ~12 min (A100) | ~8 GB | GPU compute |
```

### Suggested work breakdown
| Task | Files touched | Effort |
|------|---------------|--------|
| Rename + fix `1_masked_lang_model.mdmd` | `docs/1_masked_lang_model.mdmd` → `docs/1_masked_lang_model.md` | S |
| Audit & fix docs 2–4 | `docs/2_*.md`, `docs/3_*.md`, `docs/4_*.md` | M |
| Fix notebook artefacts | `e2e_explorer.ipynb` | M |
| Add Markdown explainer cells to notebook | `e2e_explorer.ipynb` | L |
| Reformat README | `README.md` | S |

### Delegated agent skills
- `code-formatting` — apply consistent fencing, headers
- `notebook-editing` — insert/edit Markdown cells in `.ipynb` JSON
- `documentation-writing` — narrative explainer prose and compute tables

---

## Goal 2 — Agripest / Insect Midgut Model

### Problem statement
Build a multi-modal deep learning model targeting **agricultural pest control**, specifically **insect midgut biology** — the primary site of action for Bt toxins and other biopesticides. The model should integrate five data modalities:

| Modality | Data type | Biological signal |
|----------|-----------|-------------------|
| Single-cell sequencing (scRNA-seq) | Gene expression per cell | Cell-type identity, midgut regionalization |
| Cell Painting | Morphological image features | Phenotypic response to compounds |
| Population genomics (resistant vs. sensitive) | VCF / allele frequencies | Resistance-associated loci (GWAS/selection scans) |
| Bt-toxin screening data | IC50 / mortality curves | Compound efficacy, receptor binding |
| Microbial insecticidal protein sequences (CRY, Vip, etc.) | Amino acid FASTA | Toxin family classification, binding domain prediction |

### Architecture plan

```
                        ┌────────────────────────────────────────┐
                        │         Multi-Modal Fusion Head         │
                        │  (cross-attention or concatenation MLP) │
                        └────────────┬───────────────────────────┘
           ┌────────────┬────────────┼────────────┬───────────────┐
           ▼            ▼            ▼            ▼               ▼
   ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │ scRNA-seq     │ │ Cell     │ │ PopGen   │ │ Bt Screen│ │ Protein Seq  │
   │ Transformer   │ │ Painting │ │ CNN/MLP  │ │  MLP     │ │ ESM-2 / ProtT5│
   │ (gene tokens) │ │ ResNet   │ │ (VCF feats)│ │        │ │ fine-tuned   │
   └──────────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

### Data requirements & sources
- **scRNA-seq**: Target species (e.g., *Spodoptera frugiperda*, *Helicoverpa armigera*, *Manduca sexta*). Check NCBI GEO (`insect midgut scRNA` query). *Drosophila* midgut atlas from FCA/FlyCell Atlas as proxy/pre-training donor.
- **Cell Painting**: JUMP-CP dataset (Recursion/Broad); insect cell lines (Sf9, Tn5B) if available. Will likely need in-house generation.
- **Population genomics**: Published WGS cohorts for Bt-resistance studies (e.g., *Helicoverpa*, *Plutella*). FlyBase population genomics data for proxy training (see Goal 3).
- **Bt screening**: CryDatabase, published literature IC50 tables, Bayer/Syngenta open datasets. NCBI BioAssay for any public Bt bioassay data.
- **Protein sequences**: NCBI RefSeq CRY proteins, Bt Nomenclature Committee database, UniProt Toxin keyword search.

### Model training pipeline

```
1. Per-modality pre-training
   ├─ scRNA-seq: masked gene expression modeling (scGPT-style)
   ├─ Cell Painting: contrastive / self-supervised (SimCLR on crops)
   ├─ PopGen: variant effect prediction pre-training (Enformer-style)
   ├─ Protein: load ESM-2 checkpoint, fine-tune on CRY/Vip family
   └─ Bt screening: simple MLP regressor baseline

2. Multi-modal fusion
   ├─ Option A: Late fusion — concatenate CLS tokens → MLP head
   ├─ Option B: Cross-attention fusion (preferred for interpretability)
   └─ Option C: Mixture-of-Experts gate per modality

3. Downstream tasks
   ├─ Bt toxin efficacy regression (IC50 prediction)
   ├─ Resistance locus prioritization (binary classification)
   ├─ Novel CRY protein candidate scoring
   └─ Midgut cell-type perturbation prediction
```

### Compute expectations
| Stage | Min hardware | Recommended | Wall-clock estimate |
|-------|-------------|-------------|---------------------|
| Per-modality pre-training (small) | 1× A10G 24 GB | 4× A100 80 GB | 6–48 h per modality |
| Multi-modal fusion fine-tuning | 1× A100 | 4× A100 | 2–12 h |
| Hyperparameter sweep (Optuna) | 4× A10G | 8× A100 | 12–48 h |

### Delegated agent skills
- `genomics-data-ingestion` — FASTQ→count matrix pipelines (Cell Ranger, STARsolo)
- `protein-modeling` — ESM-2 fine-tuning, domain annotation
- `multi-modal-fusion` — cross-attention architecture design
- `experiment-tracking` — MLflow / W&B integration

---

## Goal 3 — Public Data Curation & Drosophila Foundation Assets

### Relationship to `data-fetch-wizard`
The `data-fetch-wizard` effort handles **fetching and staging** raw public data. This goal defines **what to fetch, how to curate it, and how to feed it into model training**. The boundary:

| Concern | Owner |
|---------|-------|
| API calls, download scripts, caching | `data-fetch-wizard` |
| Schema harmonization, QC, tokenization, train/val/test splits | This repo (Goal 3) |
| Triggering fetch jobs from training pipelines | Shared interface (define a `DatasetManifest` JSON contract) |

This split lets you reuse `data-fetch-wizard` fetch primitives here while keeping model-specific curation logic in `pytorch_genomic_models`.

### Target public datasets

#### Transcriptomics (modENCODE & beyond)
| Dataset | Source | FTP / API | Content |
|---------|--------|-----------|---------|
| modENCODE RNA-seq compendium | modENCODE / ENCODE portal | `https://www.encodeproject.org` (organism=Drosophila melanogaster) | ~300 tissue/stage/treatment RNA-seq samples |
| FlyBase bulk RNA-seq | FlyBase | `https://flybase.org/cgi-bin/get_static_pages.pl?file=downloads/bulkdata7.html` | Curated expression tables |
| Single-cell atlas (FCA) | Fly Cell Atlas | `https://flycellatlas.org` | 10× scRNA-seq, 580k cells, all tissues |
| modENCODE ChIP-seq (TF binding) | ENCODE portal | Same as above | TF occupancy tracks → tokenisable BED |

#### Genome / variation
| Dataset | Source | Content |
|---------|--------|---------|
| *D. melanogaster* reference genome (dm6) | UCSC / FlyBase | FASTA + GFF3 gene models |
| DGRP2 (Drosophila Genetic Reference Panel) | `dgrp2.gnets.ncsu.edu` | 205 inbred lines, WGS VCFs, phenotypic data |
| DPGP3 population genomics | `dpgp.org` | African population, selection sweeps |
| Species-level VCFs (12 Drosophila genomes) | FlyBase comparative | Synteny, ortholog mapping |
| Ensembl Compara orthologs | Ensembl REST API | Orthologs to pest species |

#### Protein / interaction
| Dataset | Source | Content |
|---------|--------|---------|
| FlyBase genetic interaction data | FlyBase REST API | ~200k interactions |
| DroID protein-protein interaction | `droidb.org` | Physical + genetic |
| STRING DB (*Drosophila*) | `string-db.org` | Functional associations |
| DIOPT ortholog mapping | `www.flyrnai.org/diopt` | *Drosophila* → pest ortholog |

#### Functional screens
| Dataset | Source | Content |
|---------|--------|---------|
| FlyBase RNAi screen data | FlyBase | Phenotypic screen results |
| BDGP in situ expression | `insitu.fruitfly.org` | Spatial expression images + annotations |

### Curation pipeline

```
data-fetch-wizard (fetch layer)
         │
         ▼
┌─────────────────────────────────────────────────┐
│  1. QC & filtering                              │
│     - RNA-seq: TPM>1 filter, sample outlier QC  │
│     - VCF: MAF>0.01, FILTER=PASS, biallelic SNP │
│     - PPI: confidence score > 0.7               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  2. Tokenisation / featurisation                │
│     - Genome FASTA → k-mer tokens (k=6)         │
│     - RNA-seq counts → log1p normalise          │
│     - VCF → one-hot allele encoding per window  │
│     - PPI → adjacency matrix / graph edges      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  3. DatasetManifest JSON (shared contract)      │
│  {                                              │
│    "splits": {"train":0.8,"val":0.1,"test":0.1}│
│    "modalities": ["genome","rnaseq","ppi"],     │
│    "organism": "dmel",                          │
│    "assembly": "dm6",                           │
│    "fetch_recipe": "path/to/wizard/recipe.yaml" │
│  }                                              │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  4. Foundation model pre-training               │
│     Masked genome modelling (DNABERT-style)     │
│     + multi-task expression prediction head     │
└─────────────────────────────────────────────────┘
```

### Key files to create in this repo
- `data/schemas/dataset_manifest.json` — shared contract with `data-fetch-wizard`
- `data/pipelines/dmel_genome_tokenize.py` — FASTA → k-mer HDF5
- `data/pipelines/modencode_rnaseq_qc.py` — expression QC + normalisation
- `data/pipelines/flybase_ppi_graph.py` — PPI → PyG graph dataset
- `data/pipelines/dgrp2_vcf_encode.py` — VCF → allele feature tensors
- `notebooks/dmel_foundation_pretraining.ipynb` — end-to-end walkthrough

### Delegated agent skills
- `bioinformatics-data-fetch` (shared with `data-fetch-wizard`)
- `genomics-tokenization` — k-mer, BPE, single-nucleotide tokenisers
- `graph-dataset-construction` — PPI → PyTorch Geometric
- `foundation-model-pretraining` — MLM, expression prediction

---

## Goal 4 — Cloud Infra (Terraform)

### Motivation
M1 MacBook Pro constraints:
- 16–64 GB unified memory (no discrete CUDA GPU)
- Cannot run mixed-precision CUDA training
- Impractical for DDP multi-GPU or large-batch pre-training

### Target cloud providers
Start with **AWS** (p3/p4 instances), with modules written to be adaptable to GCP (A100 TPU pods) or Lambda Labs (cheapest A100s).

### Terraform module layout

```
infra/
├── terraform/
│   ├── main.tf                  # Root module, provider config
│   ├── variables.tf             # Instance type, region, spot vs on-demand
│   ├── outputs.tf               # Public IP, S3 bucket name
│   ├── modules/
│   │   ├── gpu_instance/        # EC2 GPU instance (p3.2xlarge → p4d.24xlarge)
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   ├── spot_fleet/          # Spot fleet for parallel hyperparameter search
│   │   │   └── ...
│   │   ├── storage/             # S3 bucket for datasets + checkpoints
│   │   │   └── ...
│   │   ├── networking/          # VPC, security groups (SSH + NFS)
│   │   │   └── ...
│   │   └── efs_mount/           # Shared EFS for multi-node training
│   │       └── ...
│   └── envs/
│       ├── dev/                 # Single p3.2xlarge, on-demand
│       └── prod/                # Multi-node p4d spot fleet
├── scripts/
│   ├── bootstrap_gpu_node.sh    # Install CUDA, PyTorch, project deps
│   ├── launch_ddp_job.sh        # torchrun wrapper + S3 sync
│   └── teardown.sh              # Graceful shutdown + checkpoint sync
└── README_infra.md
```

### Instance guide

| Use case | Instance | vCPUs | GPU | GPU RAM | Spot $/hr |
|----------|----------|-------|-----|---------|-----------|
| Development / notebook | `g4dn.xlarge` | 4 | 1× T4 | 16 GB | ~$0.16 |
| Single-modality pre-training | `p3.2xlarge` | 8 | 1× V100 | 16 GB | ~$0.90 |
| Multi-modal fusion training | `p3.8xlarge` | 32 | 4× V100 | 64 GB | ~$3.60 |
| Full foundation pre-training | `p4d.24xlarge` | 96 | 8× A100 | 320 GB | ~$10–13 |
| Hyperparameter sweep (parallel) | Spot fleet: 4–8× `p3.2xlarge` | — | — | — | ~$3.60 total |

### Key Terraform variables to expose
```hcl
variable "instance_type"       { default = "p3.2xlarge" }
variable "use_spot"            { default = true }
variable "spot_max_price"      { default = "1.50" }  # USD/hr ceiling
variable "num_nodes"           { default = 1 }
variable "project_name"        { default = "pytorch-genomic" }
variable "s3_dataset_bucket"   { default = "pytorch-genomic-datasets" }
variable "aws_region"          { default = "us-east-1" }
variable "key_pair_name"       { description = "Name of existing EC2 key pair" }
```

### Bootstrap script sketch (`bootstrap_gpu_node.sh`)
```bash
#!/bin/bash
set -euo pipefail
# Install CUDA 12 + cuDNN
# Install Miniconda → create env from environment.yml
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install transformers peft datasets biopython scanpy torch_geometric
# Mount EFS at /mnt/efs
# Sync S3 datasets to /mnt/efs/data
```

### DDP launch wrapper (`launch_ddp_job.sh`)
```bash
#!/bin/bash
# Usage: ./launch_ddp_job.sh <script.py> [--args ...]
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  "$@"
# On exit, sync checkpoints to S3
aws s3 sync ./checkpoints s3://$S3_CHECKPOINT_BUCKET/$(date +%Y%m%d_%H%M%S)/
```

### Cost controls
- Spot interruption handler → auto-checkpoint on `SIGTERM`
- Budget alert (AWS Cost Anomaly Detection) set at $200/month
- Auto-stop after N hours idle (CloudWatch alarm → EC2 stop action)

### Delegated agent skills
- `terraform-aws` — provider setup, EC2, VPC, S3, EFS modules
- `cloud-cost-optimization` — spot strategy, reserved capacity guidance
- `devops-bootstrap` — AMI selection, user-data scripts, EFS mount helpers

---

## Agent & Skill Delegation Map

| Goal | Suggested agent name | Primary skills | Inputs | Outputs |
|------|---------------------|---------------|--------|---------|
| Goal 1 | `docs-reformatter` | notebook-editing, documentation-writing | `docs/`, `e2e_explorer.ipynb`, `README.md` | Reformatted files, PR |
| Goal 2 | `midgut-model-builder` | genomics-data-ingestion, multi-modal-fusion, protein-modeling | Data manifests, pre-trained checkpoints | Model architecture code, training scripts, PR |
| Goal 3 | `dmel-data-curator` | bioinformatics-data-fetch, genomics-tokenization, foundation-model-pretraining | `DatasetManifest`, `data-fetch-wizard` API | Tokenised HDF5 datasets, pre-training scripts, PR |
| Goal 4 | `infra-provisioner` | terraform-aws, cloud-cost-optimization, devops-bootstrap | `infra/` skeleton | Working Terraform modules, PR |

Each agent operates on its own branch:
- `feat/goal-1-docs-reformat`
- `feat/goal-2-midgut-model`
- `feat/goal-3-dmel-data-curation`
- `feat/goal-4-terraform-infra`

---

## Cross-Cutting Conventions

### Repository structure (target state)
```
pytorch_genomic_models/
├── docs/                        # Goal 1: Reformatted docs
├── data/
│   ├── schemas/                 # Goal 3: DatasetManifest, vocab files
│   └── pipelines/               # Goal 3: Ingestion & tokenisation scripts
├── models/
│   ├── genomic_lm/              # Foundational genomic LM
│   ├── midgut_multimodal/       # Goal 2: Multi-modal midgut model
│   └── shared/                  # Shared layers (attention, positional enc, etc.)
├── infra/                       # Goal 4: Terraform
├── notebooks/                   # Reformatted + new notebooks
├── scripts/                     # Training launch scripts
├── tests/                       # Unit tests for all modules
├── environment.yml              # Conda env (CPU + CUDA variants)
├── pyproject.toml               # Package metadata
└── MASTER_PLAN.md               # This file
```

### Coding standards
- All Python: Black formatting, type hints, docstrings (Google style)
- Notebooks: `nbstripout` pre-commit hook to strip outputs before commit
- Terraform: `terraform fmt` + `tflint` in CI
- Secrets: never committed; use AWS Secrets Manager or `.env` (gitignored)

### Shared `DatasetManifest` contract (Goal 3 ↔ `data-fetch-wizard`)
```json
{
  "schema_version": "1.0",
  "project": "pytorch_genomic_models",
  "organism": "dmel",
  "assembly": "dm6",
  "modalities": {
    "genome": { "source": "flybase", "format": "fasta", "tokeniser": "kmer6" },
    "rnaseq": { "source": "modencode", "format": "tsv_tpm", "normalise": "log1p" },
    "ppi":    { "source": "droid", "format": "tsv_edges", "min_score": 0.7 }
  },
  "splits": { "train": 0.8, "val": 0.1, "test": 0.1 },
  "fetch_recipe": "data-fetch-wizard/recipes/dmel_foundation.yaml"
}
```

---

*This document will be broken into four dedicated sub-plan files (`PLAN_GOAL1.md` … `PLAN_GOAL4.md`) once the overall structure is agreed upon. Each sub-plan will include a detailed issue backlog and agent task list.*
