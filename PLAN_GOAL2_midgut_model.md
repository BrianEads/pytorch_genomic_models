# Goal 2 — Agripest / Insect Midgut Multi-Modal Model

**Branch:** `feat/goal-2-midgut-model`
**Agent:** `midgut-model-builder`

---

## Goal Summary

Design and implement a multi-modal deep learning model targeting **agricultural pest control**, with a specific focus on **insect midgut biology** — the primary site of action for *Bacillus thuringiensis* (Bt) toxins and other biopesticides. The model integrates five complementary data modalities (single-cell transcriptomics, cell morphology imaging, population genomics, biochemical screening, and protein sequences) through a shared cross-attention fusion head. Pre-training on *Drosophila melanogaster* public data (see Goal 3) provides a well-annotated biological anchor before transfer to hard-to-sequence pest species.

---

## Biological Context

### Insect midgut biology
The insect midgut is a highly regionalized epithelial organ that performs digestion, nutrient absorption, and immune sensing. In lepidopteran pests (*Spodoptera frugiperda*, *Helicoverpa armigera*, *Plutella xylostella*) and dipterans (*Aedes aegypti*, *Culex pipiens*), the midgut epithelium is the primary target of crystal (Cry) and vegetative insecticidal proteins (Vip) produced by *Bacillus thuringiensis*. After ingestion, Cry toxins bind to specific midgut receptors — cadherins (CAD), aminopeptidase N (APN), and alkaline phosphatase (ALP) — triggering pore formation, osmotic lysis, and larval mortality.

### Bt toxin mode of action and resistance
The sequential binding model of Bt intoxication involves: (1) proteolytic activation of the protoxin, (2) receptor binding at the brush border membrane, (3) oligomerization into a pre-pore complex, and (4) membrane insertion and ion channel formation. Field-evolved resistance in *Spodoptera* and *Helicoverpa* populations is predominantly driven by mutations in cadherin and ABCC transporter genes, often combined with reduced protoxin activation or altered midgut pH. Understanding which loci confer resistance — and which modalities reveal them first — is the core predictive task of this goal.

### Why multi-modal data and why *Drosophila*
No single data modality captures the full picture of Bt resistance: transcriptomics reveals cell-type-specific expression changes, population genomics identifies selection-swept loci, cell morphology captures phenotypic responses invisible to sequencing, and protein sequence encodes mechanistic binding determinants. *Drosophila melanogaster* is an ideal pre-training proxy because it shares deep midgut biology with pest species, has unparalleled public data resources (modENCODE, Fly Cell Atlas, DGRP2 — see Goal 3), and its genome is fully annotated with functional interaction data. Ortholog mapping via DIOPT enables direct weight transfer from a *Drosophila*-trained foundation to pest species with limited data.

---

## Data Modality Table

| Modality | Data type | Biological signal | Primary public sources | Expected volume | Preprocessing steps |
|----------|-----------|-------------------|----------------------|-----------------|---------------------|
| Single-cell RNA-seq (scRNA-seq) | Gene expression per cell (count matrix) | Cell-type identity, midgut regionalization, response to toxin | Fly Cell Atlas (FCA), NCBI GEO (`insect midgut scRNA` query), *Manduca sexta* midgut atlas | FCA: 580 k cells; per-species: 10–100 k cells | Cell Ranger / STARsolo → Seurat/Scanpy QC → log1p norm → gene token vocab |
| Cell Painting | 5-channel fluorescence images (DAPI, ER, mito, actin, nucleolus) | Morphological phenotype — compound response, cell health | JUMP-CP (Recursion/Broad Institute); Sf9/Tn5B insect cell lines if available | JUMP-CP: ~140 k compounds, ~1 M images | Illumination correction → CellProfiler feature extraction → z-score norm |
| Population genomics (resistant vs. sensitive) | VCF / allele frequency spectra | Resistance-associated loci, GWAS hits, selective sweep signatures | DGRP2 (205 *Dmel* lines), DPGP3, published *Helicoverpa/Plutella* WGS cohorts | ~200–1000 inbred lines per species | GATK genotyping → MAF filter → LD pruning → allele one-hot encoding |
| Bt-toxin biochemical screening | IC50 / mortality curves, binary activity calls | Compound efficacy, receptor binding, cross-resistance | CryDatabase, NCBI BioAssay, published IC50 tables | ~500–5000 protein–insect pairs | Log-transform IC50 → percentile normalise per toxin family |
| Insecticidal protein sequences (CRY, Vip, etc.) | Amino acid FASTA | Toxin family classification, binding domain prediction, novel candidate scoring | NCBI RefSeq CRY proteins, Bt Nomenclature Committee DB, UniProt `Toxin` keyword search | ~3000–8000 unique sequences | CD-HIT 90 % dedup → ESM-2 tokenisation → optional domain annotation (Pfam) |

---

## Architecture Diagram

```
                        ┌──────────────────────────────────────────────┐
                        │         Multi-Modal Fusion Head              │
                        │  (cross-attention or concatenation MLP)      │
                        │                                              │
                        │  [CLS_rna] [CLS_img] [CLS_pop] [CLS_bt]     │
                        │       [CLS_prot]  → linear → logits          │
                        └────────────┬─────────────────────────────────┘
           ┌────────────┬────────────┼────────────┬───────────────┐
           ▼            ▼            ▼            ▼               ▼
   ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │ scRNA-seq    │ │ Cell     │ │ PopGen   │ │ Bt Screen│ │ Protein Seq  │
   │ Transformer  │ │ Painting │ │ CNN/MLP  │ │  MLP     │ │ ESM-2 /      │
   │ (gene tokens)│ │ ResNet   │ │(VCF feats│ │          │ │ ProtT5       │
   │ scGPT-style  │ │ SimCLR   │ │per window│ │ IC50 reg)│ │ fine-tuned   │
   └──────────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

### Tower explanations

| Tower | Architecture | Pre-training objective | Rationale |
|-------|-------------|----------------------|-----------|
| scRNA-seq Transformer | Transformer encoder with gene-token vocabulary (~20 k tokens = one per gene) | Masked gene expression modelling (scGPT-style) on *Drosophila* FCA atlas | Learns cell-type embeddings transferable to pest midgut cell types via DIOPT orthologs |
| Cell Painting ResNet | ResNet-50 (5-channel input layer; ImageNet weights adapted) | Contrastive / self-supervised (SimCLR on random crops of well images) | Captures compound-induced morphological changes in insect cell lines |
| PopGen CNN/MLP | 1D CNN over allele frequency windows + MLP for summary statistics | Variant effect pre-training (Enformer-style) on DGRP2 VCFs | Identifies selection sweep signatures and resistance-associated haplotype blocks |
| Bt Screening MLP | Simple 3-layer MLP (protein embedding → IC50 regression) | Supervised regression on available IC50 data | Provides biochemical signal; bootstrapped from small data with protein embedding input |
| Protein Seq ESM-2 | ESM-2 (650M or 3B params) fine-tuned on CRY/Vip family | CRY/Vip family classification + binding domain annotation | State-of-the-art protein LM; already knows insecticidal protein families from UniProt |

### Fusion strategy options

- **Option A — Late fusion (baseline):** Concatenate all five CLS tokens → 2-layer MLP head. Simple to implement; no cross-modality interaction during encoding.
- **Option B — Cross-attention fusion (preferred):** CLS tokens from each tower act as queries; each tower's full sequence of tokens acts as keys/values. Enables the model to attend to which modality provides the most discriminative signal per sample. Adds interpretability via attention weight inspection.
- **Option C — Mixture-of-Experts (MoE) gate:** A learned router assigns per-sample weights to each tower's contribution before pooling. Useful when some modalities are missing for a given sample (common in practice).

---

## Model Training Pipeline

### Stage 1 — Per-modality pre-training

```
scRNA-seq Transformer
  Input:  gene count matrix (N_cells × N_genes), padded to max 2048 tokens
  Objective: masked gene expression modelling (mask 15% of genes, predict log1p value)
  Config:  d_model=256, n_layers=6, n_heads=8, batch=256 cells, lr=1e-4, epochs=50

Cell Painting ResNet
  Input:  5-channel 224×224 image crops
  Objective: SimCLR contrastive loss (temperature=0.07)
  Config:  ResNet-50 backbone, projection head 512→128, batch=512, lr=3e-4, epochs=100

PopGen CNN/MLP
  Input:  allele frequency matrix (W_window × N_alleles), stride 500 bp windows
  Objective: variant effect prediction (predict regulatory annotation from genome)
  Config:  Conv1d(128)×3, pool, MLP(256→128), batch=64 windows, lr=5e-5, epochs=20

Protein ESM-2 fine-tuning
  Input:  amino acid sequences (max 1024 residues)
  Objective: CRY family classification + Pfam domain boundary prediction
  Config:  ESM-2-650M, LoRA r=16, alpha=32, target=q_proj+v_proj, lr=2e-5, epochs=10

Bt Screening MLP
  Input:  ESM-2 protein embeddings (dim=1280) + compound ECFP fingerprint (dim=2048)
  Objective: IC50 regression (MSE loss on log-transformed IC50)
  Config:  MLP 3328→1024→256→1, dropout=0.3, lr=1e-3, epochs=200
```

### Stage 2 — Multi-modal fusion training

```
Fusion head training (cross-attention preferred)
  Input:  frozen or partially-thawed tower embeddings
  Strategy: warm up fusion head for 5 epochs with towers frozen, then unfreeze last 2 layers
  Config:  cross-attention d_model=256, n_heads=4, dropout=0.1, lr=1e-4, epochs=20
  Mixed precision: fp16 with GradScaler
  Gradient accumulation: 4 steps (effective batch = actual_batch × 4)
```

### Stage 3 — Downstream task fine-tuning

```
Each downstream task adds a lightweight head on top of the frozen fusion model.
Fine-tune with AdamW, cosine LR schedule, early stopping on val loss.
```

---

## Compute Requirements Table

| Stage | Min hardware | Recommended hardware | Wall-clock estimate |
|-------|-------------|---------------------|---------------------|
| scRNA-seq pre-training (*Dmel* FCA) | 1× A10G 24 GB | 4× A100 80 GB | 6–24 h per modality |
| Cell Painting SimCLR | 1× A100 40 GB | 4× A100 | 12–48 h |
| PopGen CNN pre-training | 1× V100 16 GB | 2× A100 | 4–12 h |
| ESM-2 fine-tuning (LoRA) | 1× A10G 24 GB | 1× A100 80 GB | 2–8 h |
| Bt Screening MLP | 1× GPU or CPU | 1× T4 / V100 | < 1 h |
| Multi-modal fusion training | 1× A100 80 GB | 4× A100 | 2–12 h |
| Hyperparameter sweep (Optuna) | 4× A10G | 8× A100 | 12–48 h |

---

## Downstream Task Definitions

### Task 1 — Bt toxin efficacy regression (IC50 prediction)

| Field | Detail |
|-------|--------|
| Input | Fusion embedding of a (protein sequence, insect genotype, cell morphology) triple |
| Output | log10(IC50) in µg/mL |
| Loss function | MSE on log-transformed IC50 |
| Evaluation metric | Pearson r, RMSE on held-out protein–insect pairs |

### Task 2 — Resistance locus prioritization (binary classification)

| Field | Detail |
|-------|--------|
| Input | PopGen tower embedding of a genomic window (500 bp); optional scRNA-seq context |
| Output | Binary: resistance-associated (1) / neutral (0) |
| Loss function | Binary cross-entropy with label smoothing (ε=0.1) |
| Evaluation metric | AUROC, precision-recall AUC on known resistance loci from literature |

### Task 3 — Novel CRY protein candidate scoring

| Field | Detail |
|-------|--------|
| Input | ESM-2 embedding of a candidate protein sequence |
| Output | Probability of insecticidal activity + predicted target order (Lepidoptera / Diptera / Coleoptera) |
| Loss function | Multi-label cross-entropy |
| Evaluation metric | Top-k precision against CryDatabase held-out set |

### Task 4 — Midgut cell-type perturbation prediction

| Field | Detail |
|-------|--------|
| Input | scRNA-seq tower embedding of control cells + compound/toxin descriptor |
| Output | Predicted post-perturbation cell-type distribution shift (KL-divergence target) |
| Loss function | KL-divergence |
| Evaluation metric | Spearman correlation of predicted vs. observed cell-type fraction shifts |

---

## Tooling & Stack

| Tool / Library | Role |
|----------------|------|
| `torch` ≥ 2.1 | Core DL framework (DDP, AMP, `torch.compile`) |
| `transformers` (HuggingFace) | Transformer building blocks, tokenizers, trainer API |
| `fair-esm` | ESM-2 model weights and tokenizer |
| `peft` | LoRA / adapter fine-tuning for ESM-2 and scRNA-seq Transformer |
| `scanpy` | scRNA-seq QC, normalization, cell-type annotation |
| `torch_geometric` | Graph neural network layers (for PPI tower in Goal 3 integration) |
| `torchvision` | ResNet backbone, image augmentations for Cell Painting |
| `mlflow` or `wandb` | Experiment tracking, hyperparameter logging |
| `optuna` | Hyperparameter optimization |
| `h5py` | HDF5 storage for tokenised datasets |
| `biopython` | FASTA parsing, sequence alignment utilities |
| `numpy`, `pandas`, `scipy` | Numerical utilities |

---

## Open Questions / Risks

| Question / Risk | Severity | Mitigation |
|-----------------|----------|-----------|
| Lack of public insect Cell Painting data (Sf9/Tn5B) | High | Use JUMP-CP human cell lines as domain transfer starting point; plan in-house Sf9 imaging if budget allows |
| Cross-species transfer quality (*Dmel* → lepidopteran pests) | High | Validate DIOPT ortholog coverage (target > 70% of Bt receptor genes); include *D. virilis* / *D. simulans* to broaden phylogenetic range |
| ESM-2 sequence length limit (1024 residues) | Medium | CRY proteins are typically 1100–1200 aa; use sliding-window chunking or ESM-2-3B with positional interpolation |
| IC50 data heterogeneity (different assay protocols across labs) | Medium | Normalise within assay protocol group; include assay protocol as a covariate in the MLP |
| Availability of *Spodoptera* / *Helicoverpa* scRNA-seq data | Medium | Fall back to *Manduca sexta* midgut atlas; use cross-species alignment via ortholog mapping |
| Compute cost for ESM-2-3B fine-tuning | Medium | Use LoRA to limit trainable parameters; profile on g4dn.xlarge before scaling |

---

## Agent Instructions — `midgut-model-builder`

Execute these steps in order. Read this entire document before starting.

### Step 1 — Set up branch and environment

```bash
git checkout -b feat/goal-2-midgut-model
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers fair-esm peft scanpy torch_geometric mlflow optuna h5py biopython
```

### Step 2 — Create directory structure

```bash
mkdir -p models/midgut_multimodal/{towers,fusion,downstream,configs}
mkdir -p scripts/midgut/
mkdir -p notebooks/
touch models/midgut_multimodal/__init__.py
touch models/midgut_multimodal/towers/__init__.py
touch models/midgut_multimodal/fusion/__init__.py
touch models/midgut_multimodal/downstream/__init__.py
```

### Step 3 — Implement per-modality tower stubs

Create one Python module per tower under `models/midgut_multimodal/towers/`:
- `scrna_transformer.py` — Transformer encoder; input: gene count tensor (B, N_genes); output: CLS token (B, d_model)
- `cell_painting_resnet.py` — ResNet-50 with 5-channel first layer; output: CLS embedding (B, d_model)
- `popgen_cnn.py` — 1D CNN over allele frequency windows; output: pooled embedding (B, d_model)
- `bt_screening_mlp.py` — 3-layer MLP; input: protein + compound embeddings; output: (B, d_model)
- `protein_esm2.py` — ESM-2 wrapper with LoRA; output: mean-pooled residue embedding (B, d_model)

All towers must:
- Accept `torch.Tensor` inputs with documented shapes in the module docstring.
- Return a tuple `(embedding: Tensor[B, d_model], attention_weights: Optional[Tensor])`.
- Use Google-style docstrings and type hints.

### Step 4 — Implement fusion head

Create `models/midgut_multimodal/fusion/cross_attention_fusion.py`:
- Accept a list of tower embeddings `List[Tensor[B, d_model]]`.
- Implement cross-attention fusion (Option B from the architecture section above).
- Also implement a `LateFusionHead` (Option A) as a fallback.
- Return fused embedding `Tensor[B, d_fusion]`.

### Step 5 — Implement downstream task heads

Create one module per downstream task under `models/midgut_multimodal/downstream/`:
- `ic50_regression.py` — linear head, MSE loss
- `resistance_classifier.py` — binary classification head, BCE loss
- `cry_candidate_scorer.py` — multi-label head
- `perturbation_predictor.py` — KL-divergence output

### Step 6 — Create training configurations

Create `models/midgut_multimodal/configs/`:
- `pretrain_scrna.yaml` — scRNA-seq pre-training hyperparameters
- `pretrain_esm2_finetune.yaml` — ESM-2 LoRA fine-tuning config
- `fusion_train.yaml` — fusion head training config
- `downstream_tasks.yaml` — shared downstream task config

### Step 7 — Write training scripts

Create `scripts/midgut/train_tower.py` — CLI script to pre-train a single tower given a config file.
Create `scripts/midgut/train_fusion.py` — CLI script to train the fusion head given pre-trained tower checkpoints.
Create `scripts/midgut/eval_downstream.py` — CLI script to evaluate all four downstream tasks.

Scripts must:
- Use `argparse` for CLI arguments.
- Log metrics to MLflow (or W&B if `--log-wandb` flag is passed).
- Support DDP via `torchrun` (read `LOCAL_RANK` from environment).
- Save checkpoints to `./checkpoints/<run_id>/` with the format `epoch={e}_loss={l:.4f}.pt`.

### Step 8 — Write unit tests

Create `tests/test_midgut_towers.py`:
- Test each tower with random-input tensors of the correct shape.
- Assert output shape is `(B, d_model)`.
- Assert no exceptions for a forward pass on CPU.

```bash
pytest tests/test_midgut_towers.py -v
```

### Step 9 — Commit and open PR

```bash
git add models/midgut_multimodal/ scripts/midgut/ tests/test_midgut_towers.py
git commit -m "feat(goal-2): midgut multi-modal model architecture, tower stubs, fusion head"
git push origin feat/goal-2-midgut-model
```

Open a pull request targeting `main` with the title: `[Goal 2] Midgut multi-modal model skeleton`.
