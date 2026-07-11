# Goal 3 — Public Data Curation & Drosophila Foundation Assets

**Branch:** `feat/goal-3-dmel-data-curation`
**Agent:** `dmel-data-curator`

---

## Goal Summary

Curate a high-quality set of *Drosophila melanogaster* public datasets — spanning transcriptomics, genome/variation, protein interactions, and functional screens — and transform them into model-ready tensors following a well-defined schema contract. This goal owns **schema harmonization, QC, tokenisation, and train/val/test splitting** for all modalities. Raw data download and staging is handled by the separate `data-fetch-wizard` effort; this goal defines the contract between the two systems and ensures that curated outputs flow cleanly into the pre-training pipeline described in Goal 2.

---

## `data-fetch-wizard` Boundary

This boundary must be respected at all times. Do not add download logic or API call code to this repository.

| Concern | Owner |
|---------|-------|
| API calls to FlyBase, ENCODE, modENCODE, NCBI GEO | `data-fetch-wizard` |
| HTTP/FTP download scripts and caching | `data-fetch-wizard` |
| Checksumming and raw file integrity verification | `data-fetch-wizard` |
| Schema harmonization (column renaming, unit normalisation) | This repo (Goal 3) |
| Per-modality QC filters and outlier removal | This repo (Goal 3) |
| Tokenisation and feature encoding | This repo (Goal 3) |
| Train / val / test split assignment | This repo (Goal 3) |
| Writing `DatasetManifest` JSON | This repo (Goal 3) |
| Triggering fetch jobs from within training pipelines | Shared via `DatasetManifest.fetch_recipe` pointer |

### `DatasetManifest` JSON contract spec

The `DatasetManifest` is the single contract that connects `data-fetch-wizard` outputs to this repo's curation pipelines. Every curated dataset must have one.

Full JSON schema (`data/schemas/dataset_manifest_schema.json`):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DatasetManifest",
  "description": "Contract between data-fetch-wizard and pytorch_genomic_models curation pipelines",
  "type": "object",
  "required": ["schema_version", "project", "organism", "assembly", "modalities", "splits"],
  "properties": {
    "schema_version": {
      "type": "string",
      "description": "Semantic version of this schema (e.g. '1.0')"
    },
    "project": {
      "type": "string",
      "description": "Repository or project name this manifest belongs to"
    },
    "organism": {
      "type": "string",
      "description": "NCBI taxonomy short name (e.g. 'dmel', 'cele', 'hsapiens')"
    },
    "assembly": {
      "type": "string",
      "description": "Genome assembly identifier (e.g. 'dm6', 'hg38')"
    },
    "modalities": {
      "type": "object",
      "description": "Map of modality name → modality config",
      "additionalProperties": {
        "type": "object",
        "required": ["source", "format"],
        "properties": {
          "source":     { "type": "string", "description": "Dataset origin label (e.g. 'flybase', 'modencode')" },
          "format":     { "type": "string", "description": "Raw file format (e.g. 'fasta', 'tsv_tpm', 'vcf', 'tsv_edges')" },
          "tokeniser":  { "type": "string", "description": "Tokenisation method to apply (e.g. 'kmer6', 'log1p', 'one_hot_allele')" },
          "normalise":  { "type": "string", "description": "Optional normalisation step (e.g. 'log1p', 'z_score')" },
          "min_score":  { "type": "number", "description": "Minimum confidence score for interaction data" },
          "raw_path":   { "type": "string", "description": "Path to raw staged file(s) written by data-fetch-wizard" },
          "output_path":{ "type": "string", "description": "Path where curation pipeline will write tokenised output" }
        }
      }
    },
    "splits": {
      "type": "object",
      "required": ["train", "val", "test"],
      "properties": {
        "train": { "type": "number", "minimum": 0, "maximum": 1 },
        "val":   { "type": "number", "minimum": 0, "maximum": 1 },
        "test":  { "type": "number", "minimum": 0, "maximum": 1 }
      },
      "description": "Fractional split proportions; must sum to 1.0"
    },
    "split_strategy": {
      "type": "string",
      "enum": ["chromosome_held_out", "sample_held_out", "random"],
      "description": "How to assign items to splits without data leakage"
    },
    "fetch_recipe": {
      "type": "string",
      "description": "Relative path to the data-fetch-wizard recipe YAML that produces the raw_path files"
    }
  }
}
```

Example manifest for the *Drosophila* foundation dataset:

```json
{
  "schema_version": "1.0",
  "project": "pytorch_genomic_models",
  "organism": "dmel",
  "assembly": "dm6",
  "modalities": {
    "genome": {
      "source": "flybase",
      "format": "fasta",
      "tokeniser": "kmer6",
      "raw_path": "data/raw/dmel_dm6.fasta",
      "output_path": "data/tokenised/dmel_genome_kmers.h5"
    },
    "rnaseq": {
      "source": "modencode",
      "format": "tsv_tpm",
      "normalise": "log1p",
      "raw_path": "data/raw/modencode_rnaseq_tpm.tsv",
      "output_path": "data/tokenised/modencode_rnaseq_log1p.h5"
    },
    "ppi": {
      "source": "droid",
      "format": "tsv_edges",
      "min_score": 0.7,
      "raw_path": "data/raw/droid_ppi_edges.tsv",
      "output_path": "data/tokenised/dmel_ppi_graph.pt"
    }
  },
  "splits": { "train": 0.8, "val": 0.1, "test": 0.1 },
  "split_strategy": "chromosome_held_out",
  "fetch_recipe": "data-fetch-wizard/recipes/dmel_foundation.yaml"
}
```

---

## Dataset Catalogue

### Transcriptomics

| Dataset | Source | URL / API endpoint | Content description | Expected size | Licence / access notes |
|---------|--------|--------------------|--------------------|--------------|-----------------------|
| modENCODE RNA-seq compendium | ENCODE portal | `https://www.encodeproject.org` (organism=Drosophila melanogaster, assay=RNA-seq) | ~300 tissue / stage / treatment bulk RNA-seq samples; TPM + raw count matrices | ~5 GB processed | CC BY 4.0; open download via ENCODE REST API |
| FlyBase bulk RNA-seq expression tables | FlyBase | `https://flybase.org/cgi-bin/get_static_pages.pl?file=downloads/bulkdata7.html` | Curated per-gene expression across >100 conditions | ~500 MB | FlyBase terms; free download |
| Fly Cell Atlas (FCA) | Fly Cell Atlas | `https://flycellatlas.org` | 10× scRNA-seq, 580 k cells, all adult tissues and stages | ~10 GB (loom/h5ad) | CC BY 4.0 |
| modENCODE ChIP-seq TF binding | ENCODE portal | Same as above (assay=ChIP-seq) | TF occupancy tracks for ~200 TFs; tokenisable BED/bigWig | ~20 GB raw | CC BY 4.0 |

### Genome / Variation

| Dataset | Source | URL / API endpoint | Content description | Expected size | Licence / access notes |
|---------|--------|--------------------|--------------------|--------------|-----------------------|
| *D. melanogaster* reference genome (dm6) | FlyBase / UCSC | `https://ftp.flybase.net/genomes/Drosophila_melanogaster/current/fasta/` | FASTA (chromosome sequences) + GFF3 gene models | ~180 MB FASTA | Public domain |
| DGRP2 (Drosophila Genetic Reference Panel) | DGRP2 portal | `https://dgrp2.gnets.ncsu.edu` | 205 inbred lines, WGS VCFs, phenotypic data for >50 traits | ~30 GB VCFs | Open for research; cite DGRP paper |
| DPGP3 African population genomics | DPGP | `https://www.dpgp.org` | ~200 African lines; selection sweep signatures | ~5 GB | Open; cite DPGP3 paper |
| 12 *Drosophila* species genomes | FlyBase comparative | `https://ftp.flybase.net/genomes/` | One FASTA per species; synteny blocks | ~2 GB total | Public domain |
| Ensembl Compara orthologs (*Dmel* → pests) | Ensembl REST API | `https://rest.ensembl.org/homology/id/` | One-to-one and one-to-many orthologs to *Spodoptera*, *Helicoverpa*, *Ae. aegypti* | ~50 MB TSV | Apache 2.0 |

### Protein / Interaction

| Dataset | Source | URL / API endpoint | Content description | Expected size | Licence / access notes |
|---------|--------|--------------------|--------------------|--------------|-----------------------|
| FlyBase genetic interaction data | FlyBase REST API | `https://api.flybase.org/api/v1.0/gene/` interactions endpoint | ~200 k genetic interactions (enhancer/suppressor) with phenotype annotations | ~100 MB | FlyBase terms; free |
| DroID protein-protein interactions | DroID | `https://droidb.org/data/` | Physical (Y2H, AP-MS) + genetic interactions; ~100 k edges | ~50 MB | Creative Commons |
| STRING *Drosophila* | STRING-DB | `https://string-db.org/cgi/download?species_text=Drosophila+melanogaster` | Functional association network; combined score ≥ 0.4 | ~20 MB | CC BY 4.0 |
| DIOPT ortholog mapping | DIOPT at DRSC | `https://www.flyrnai.org/diopt` / REST API | *Drosophila* → human and pest ortholog scores | ~30 MB | Open |

### Functional Screens

| Dataset | Source | URL / API endpoint | Content description | Expected size | Licence / access notes |
|---------|--------|--------------------|--------------------|--------------|-----------------------|
| FlyBase RNAi screen data | FlyBase | `https://flybase.org` FBrf screen reports | Phenotypic screen results for >10 k genes from DRSC and NIG-Fly libraries | ~200 MB | FlyBase terms; free |
| BDGP in situ expression atlas | BDGP | `https://insitu.fruitfly.org/cgi-bin/ex/insitu.pl` | Spatial expression images (RNA FISH) + stage/tissue annotations for ~7000 genes | ~5 GB images | Public domain |

---

## Curation Pipeline Diagram

```
data-fetch-wizard (fetch layer — NOT this repo)
         │  writes raw files to data/raw/
         ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — QC & Filtering                               │
│                                                          │
│  Transcriptomics (RNA-seq):                              │
│    - Remove samples with < 1000 genes detected          │
│    - Remove genes with TPM > 1 in < 5% of samples       │
│    - Flag and remove outlier samples (z-score > 3 on    │
│      total read count)                                   │
│                                                          │
│  Genome / VCF:                                           │
│    - FILTER=PASS only; biallelic SNPs only               │
│    - MAF > 0.01 across the panel                         │
│    - Remove sites in repetitive / low-complexity regions │
│                                                          │
│  PPI / interaction:                                      │
│    - Retain edges with combined confidence score > 0.7   │
│    - Deduplicate reciprocal edges                        │
│    - Remove self-loops                                   │
│                                                          │
│  scRNA-seq:                                              │
│    - min_genes=200, max_genes=5000 per cell              │
│    - max_pct_mito=20%                                    │
│    - Doublet removal (Scrublet score > 0.25)             │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — Tokenisation / Featurisation                  │
│                                                          │
│  Genome FASTA → k-mer tokens (k=6)                       │
│    kmer_vocab size = 4^6 = 4096 + special tokens         │
│    stride = 1 (overlapping), window = 512 tokens         │
│                                                          │
│  RNA-seq counts → log1p normalisation                    │
│    log1p(TPM) per sample; then z-score across genes      │
│                                                          │
│  VCF → one-hot allele encoding per window                │
│    500 bp windows, stride 250 bp                         │
│    REF/ALT encoded as (0,0)/(1,0)/(0,1) per position     │
│                                                          │
│  PPI → adjacency matrix / graph edge list                │
│    PyTorch Geometric Data object                         │
│    Node features: ESM-2 mean-pooled protein embeddings   │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — DatasetManifest JSON (shared contract)        │
│                                                          │
│  Written to data/manifests/<dataset_name>.json           │
│  Validated against data/schemas/dataset_manifest_schema  │
│  Contains: modality configs, split assignments,          │
│  output HDF5 paths, fetch_recipe pointer                 │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4 — Foundation model pre-training                 │
│                                                          │
│  Masked genome modelling (DNABERT-style k-mer MLM)       │
│  + multi-task expression prediction head                 │
│  (see Goal 2 for architecture and training config)       │
└──────────────────────────────────────────────────────────┘
```

---

## Files to Create

### `data/schemas/`

| File | Description |
|------|-------------|
| `data/schemas/dataset_manifest_schema.json` | JSON Schema (draft-07) defining the `DatasetManifest` contract; all manifests must validate against this |
| `data/schemas/rnaseq_sample_schema.json` | Schema for per-sample RNA-seq metadata (organism, tissue, stage, treatment, replicate) |
| `data/schemas/vcf_window_schema.json` | Schema for VCF window tensor metadata (chromosome, start, end, MAF stats) |
| `data/schemas/ppi_graph_schema.json` | Schema for PPI graph metadata (source DB, confidence threshold, node count, edge count) |

### `data/pipelines/`

| File | Description |
|------|-------------|
| `data/pipelines/dmel_genome_tokenize.py` | Read dm6 FASTA, slide k=6 window, write k-mer integer token HDF5; CLI: `--fasta --out --kmer-size --window --stride` |
| `data/pipelines/modencode_rnaseq_qc.py` | Read TPM TSV from ENCODE/modENCODE, apply sample and gene QC filters, log1p + z-score normalise, write HDF5 |
| `data/pipelines/flybase_ppi_graph.py` | Read DroID/FlyBase edge TSV, filter by confidence, build PyG `Data` object with ESM-2 node features, write `.pt` file |
| `data/pipelines/dgrp2_vcf_encode.py` | Read DGRP2 VCF, apply MAF/PASS filters, encode alleles as one-hot tensor per 500 bp window, write HDF5 |
| `data/pipelines/fca_scrna_qc.py` | Read Fly Cell Atlas h5ad/loom, apply cell QC, Scrublet doublet removal, log1p normalise, write Scanpy AnnData |
| `data/pipelines/manifest_validate.py` | CLI tool to validate a `DatasetManifest` JSON against the schema; exit non-zero on failure |
| `data/pipelines/split_assigner.py` | Given a manifest, assign each sample/region to train/val/test per the `split_strategy`; write split index files |

### `notebooks/`

| File | Description |
|------|-------------|
| `notebooks/dmel_foundation_pretraining.ipynb` | End-to-end walkthrough: load manifests → run pipelines → launch scRNA-seq and genome MLM pre-training on a small sample |

---

## Tokenisation Standards

| Modality | Method | Rationale |
|----------|--------|-----------|
| Genome FASTA | k-mer tokenisation with k=6 (hexamers); vocabulary = 4^6 = 4096 tokens + 5 special tokens (`[PAD]`, `[MASK]`, `[CLS]`, `[SEP]`, `[UNK]`) | k=6 is the standard in DNABERT and related genomic LMs; captures codon-level and short regulatory motif information; computationally tractable vocab size |
| RNA-seq expression | log1p(TPM) followed by per-gene z-score standardisation across samples | log1p compresses the extreme right tail of count distributions without losing zero signal; z-score makes expression levels comparable across experiments and sequencing depths |
| VCF / population genomics | One-hot allele encoding per position: REF homozygous = (1,0,0), ALT heterozygous = (0,1,0), ALT homozygous = (0,0,1), missing = (0,0,0); windowed into 500 bp tiles | One-hot preserves phasing information; fixed-size windows make tensors compatible with Conv1D models; missing data coded as zero vector (not imputed, to avoid introducing spurious signals) |
| PPI / interaction graph | Adjacency matrix representation as PyTorch Geometric `edge_index` (COO sparse format); node features from ESM-2 mean-pooled residue embeddings for each protein | Graph representation preserves network topology; ESM-2 embeddings provide biologically meaningful starting features without requiring separate protein pre-training at this stage |

---

## Train / Val / Test Split Strategy

### Genome (k-mer windows)

Use **chromosome-held-out** splitting to prevent data leakage between nearby genomic regions:
- Test set: chromosomes `chr2L`, `chrX` (held out entirely)
- Val set: chromosome `chr3L`
- Train set: all remaining chromosomes (`chr2R`, `chr3R`, `chr4`, `chrY`, etc.)

This ensures no k-mer windows from adjacent chromosomal positions appear in both train and test sets.

### RNA-seq samples

Use **sample-held-out** splitting — assign entire biological samples to a split:
- Randomly stratify by tissue category (midgut, fat body, brain, etc.) to preserve tissue representation in all splits.
- 80% train / 10% val / 10% test by sample count.
- Never split replicates of the same biological sample across train and test.

### VCF / population lines (DGRP2)

Use **line-held-out** splitting — assign entire inbred lines to a split:
- Test: 20 randomly held-out DGRP2 lines (from full list of 205).
- Val: 20 additional held-out lines.
- Train: remaining 165 lines.
- This prevents the model from seeing the same individual's genotype at train and test time.

### PPI graph

Split by **gene holdout** — designate 10% of genes as test-only nodes; their edges (protein interactions) are excluded from training and used only for link-prediction evaluation. This tests the model's ability to generalise to unseen proteins.

---

## Tooling & Stack

| Tool / Library | Role |
|----------------|------|
| `biopython` | FASTA parsing (`SeqIO`), k-mer iteration |
| `pysam` | VCF/BCF reading and filtering |
| `scanpy` | scRNA-seq QC, normalisation, AnnData I/O |
| `scrublet` | Doublet detection in scRNA-seq |
| `pyranges` | Genomic interval operations (BED/GFF windowing) |
| `torch_geometric` | PyG `Data` object for PPI graph |
| `h5py` | HDF5 output for tokenised tensors |
| `jsonschema` | Manifest validation against JSON Schema |
| `pandas`, `numpy` | Tabular data manipulation |
| FlyBase REST API | Gene metadata, interaction fetching |
| ENCODE portal API | RNA-seq and ChIP-seq experiment listing and download coordination (via `data-fetch-wizard`) |

---

## Agent Instructions — `dmel-data-curator`

Execute these steps in order. Do not add any download or API call code to this repository — those belong in `data-fetch-wizard`.

### Step 1 — Set up branch and environment

```bash
git checkout -b feat/goal-3-dmel-data-curation
pip install biopython pysam scanpy scrublet pyranges torch_geometric h5py jsonschema pandas numpy
```

### Step 2 — Create directory structure

```bash
mkdir -p data/schemas data/pipelines data/manifests data/raw data/tokenised notebooks
touch data/schemas/.gitkeep data/raw/.gitkeep data/tokenised/.gitkeep data/manifests/.gitkeep
```

Add `data/raw/` and `data/tokenised/` to `.gitignore` (these are generated data, not code).

### Step 3 — Write the `DatasetManifest` JSON schema

Create `data/schemas/dataset_manifest_schema.json` with the full schema from the `DatasetManifest` contract spec section above.

Create `data/pipelines/manifest_validate.py`:
```python
"""CLI tool to validate a DatasetManifest JSON file against the schema."""
import argparse, json, sys
import jsonschema

def main():
    parser = argparse.ArgumentParser(description="Validate a DatasetManifest JSON file.")
    parser.add_argument("manifest", help="Path to manifest JSON file to validate.")
    parser.add_argument("--schema", default="data/schemas/dataset_manifest_schema.json")
    args = parser.parse_args()
    with open(args.schema) as f:
        schema = json.load(f)
    with open(args.manifest) as f:
        manifest = json.load(f)
    jsonschema.validate(instance=manifest, schema=schema)
    print(f"✓ {args.manifest} is valid.")

if __name__ == "__main__":
    main()
```

### Step 4 — Implement genome tokenisation pipeline

Create `data/pipelines/dmel_genome_tokenize.py`. Key behaviour:
- Accept `--fasta`, `--out`, `--kmer-size` (default 6), `--window` (default 512 tokens), `--stride` (default 256 tokens).
- Iterate chromosomes in the FASTA; slide a window across each chromosome.
- For each window, generate the k-mer token sequence and encode as integer IDs using the 4^k vocabulary.
- Write output to HDF5 with datasets `tokens` (int16), `chromosome` (string), `start` (int32), `end` (int32).
- Log progress every 10 k windows.

### Step 5 — Implement RNA-seq QC pipeline

Create `data/pipelines/modencode_rnaseq_qc.py`. Key behaviour:
- Accept `--tpm-tsv`, `--out`, `--min-tpm` (default 1.0), `--min-samples-pct` (default 0.05).
- Filter genes not meeting TPM threshold in at least `min-samples-pct` fraction of samples.
- Identify and flag (do not silently drop) outlier samples by z-score on total read count.
- Apply log1p then per-gene z-score.
- Write output to HDF5: `expression` matrix (float32), `gene_ids`, `sample_ids`, `split` (assigned later by `split_assigner.py`).

### Step 6 — Implement PPI graph pipeline

Create `data/pipelines/flybase_ppi_graph.py`. Key behaviour:
- Accept `--edges-tsv` (DroID/FlyBase edge file), `--out`, `--min-score` (default 0.7).
- Filter edges below confidence threshold; remove self-loops and reciprocal duplicates.
- Assign integer node IDs to FlyBase gene IDs.
- Write a PyTorch Geometric `Data` object with `edge_index` and `gene_id` node attribute.
- Node features (`x`) should be left as `None` at this stage; they will be filled with ESM-2 embeddings in Goal 2.

### Step 7 — Implement VCF encoding pipeline

Create `data/pipelines/dgrp2_vcf_encode.py`. Key behaviour:
- Accept `--vcf`, `--out`, `--window` (default 500), `--stride` (default 250), `--min-maf` (default 0.01).
- Use `pysam.VariantFile` to iterate VCF records.
- Filter: FILTER=PASS, biallelic SNPs only, MAF > min-maf.
- Encode each 500 bp window as a one-hot tensor (window_size × 3) per line.
- Write HDF5: `alleles` (uint8), `chrom`, `start`, `end`, `line_id`.

### Step 8 — Implement split assigner

Create `data/pipelines/split_assigner.py`. Key behaviour:
- Read a `DatasetManifest` JSON.
- Apply the split strategy (`chromosome_held_out`, `sample_held_out`, or `random`).
- Write a split index file (`data/manifests/<dataset>_splits.json`) mapping each sample/region ID to `train`, `val`, or `test`.
- Validate that all three splits are non-empty; raise an error if any is empty.

### Step 9 — Write the example manifest

Create `data/manifests/dmel_foundation_manifest.json` with the example manifest from the contract spec section above. Validate it:

```bash
python data/pipelines/manifest_validate.py data/manifests/dmel_foundation_manifest.json
```

### Step 10 — Create the walkthrough notebook

Create `notebooks/dmel_foundation_pretraining.ipynb` with Markdown explainer cells (following Goal 1 formatting standards) and code cells demonstrating:
1. Load and validate the manifest.
2. Run `dmel_genome_tokenize.py` on a 1 Mb test region.
3. Run `modencode_rnaseq_qc.py` on a small subset of samples.
4. Run `flybase_ppi_graph.py` on the DroID edge file.
5. Assign splits and inspect the result.
6. Launch a toy masked genome MLM pre-training run for 10 steps.

### Step 11 — Write unit tests

Create `tests/test_dmel_pipelines.py`:
- Test `dmel_genome_tokenize.py` on a synthetic 1000 bp FASTA; assert output HDF5 has correct shape.
- Test `manifest_validate.py` accepts the example manifest and rejects a manifest missing required fields.
- Test `split_assigner.py` produces non-overlapping splits.

```bash
pytest tests/test_dmel_pipelines.py -v
```

### Step 12 — Commit and open PR

```bash
git add data/schemas/ data/pipelines/ data/manifests/ notebooks/dmel_foundation_pretraining.ipynb tests/test_dmel_pipelines.py .gitignore
git commit -m "feat(goal-3): dmel data curation pipelines, DatasetManifest schema, split logic"
git push origin feat/goal-3-dmel-data-curation
```

Open a pull request targeting `main` with the title: `[Goal 3] Drosophila data curation pipelines and DatasetManifest schema`.
