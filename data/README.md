# Data Curation (Goal 3)

Curated *Drosophila melanogaster* foundation assets for pre-training. Raw downloads are owned by **[data-fetch-wizard](https://github.com/bayer-int/cs-cp-bifx-data-fetch-wizard)** (DFW); this repo performs QC, tokenisation, and split assignment only.

## v1 Scope (incremental)

| Modality | Status | Pipeline | Manifest key |
|----------|--------|----------|--------------|
| Genome (dm6 k-mers) | **v1** | `pipelines/dmel_genome_tokenize.py` | `genome` |
| Bulk RNA-seq (modENCODE TPM) | **v1** | `pipelines/modencode_rnaseq_qc.py` | `rnaseq` |
| PPI graph (DroID) | **v1** | `pipelines/flybase_ppi_graph.py` | `ppi` |
| DGRP2 VCF windows | planned v2 | `pipelines/dgrp2_vcf_encode.py` | — |
| Fly Cell Atlas scRNA-seq | planned v2 | `pipelines/fca_scrna_qc.py` | — |

See `manifests/dmel_foundation_manifest.json` for the canonical v1 contract.

## data-fetch-wizard Integration

### Flow

```
DFW recipe YAML  →  S3 raw/  →  local sync  →  curation pipeline  →  S3 tokenised/ + manifests/
                         ↑                              ↑
              data-fetch-wizard                   this repo (Goal 3)
                                                         ↓
                                              S3 checkpoints/ (Goal 2 training)
```

### S3 layout (dev bucket)

**Bucket:** `cs-cp-bifx-dfw-pytorch-genomic-data`

```
s3://cs-cp-bifx-dfw-pytorch-genomic-data/
├── raw/                          ← DFW writes staged downloads
│   ├── dmel_dm6.fasta
│   ├── modencode_rnaseq_tpm.tsv
│   ├── modencode_rnaseq_samples.tsv
│   └── droid_ppi_edges.tsv
├── tokenised/                    ← Goal 3 curation pipelines
│   ├── dmel_genome_kmers.h5
│   ├── modencode_rnaseq_log1p.h5
│   └── dmel_ppi_graph.pt
├── manifests/                    ← DatasetManifest + split index files
│   ├── dmel_foundation_manifest.json
│   └── dmel_foundation_splits.json
└── checkpoints/                  ← Goal 2 training outputs
```

### Recipe expectation

Manifest field `fetch_recipe` points to the DFW recipe that produces all v1 raw files:

```
data-fetch-wizard/recipes/dmel_foundation.yaml
```

Expected recipe outputs (minimum):

| File | Format | Notes |
|------|--------|-------|
| `dmel_dm6.fasta` | multi-record FASTA | FlyBase dm6 reference |
| `modencode_rnaseq_tpm.tsv` | TSV, genes × samples | TPM values; first column `gene_id` |
| `modencode_rnaseq_samples.tsv` | TSV metadata | Columns: `sample_id`, `tissue`, `stage`, `treatment`, `replicate` |
| `droid_ppi_edges.tsv` | TSV edges | gene_a, gene_b, score columns |

### Local sync (dev / training nodes)

After DFW fetch completes, sync raw files before running pipelines:

```bash
aws s3 sync s3://cs-cp-bifx-dfw-pytorch-genomic-data/raw/ data/raw/
```

Upload tokenised outputs and manifests after curation:

```bash
aws s3 sync data/tokenised/ s3://cs-cp-bifx-dfw-pytorch-genomic-data/tokenised/
aws s3 sync data/manifests/ s3://cs-cp-bifx-dfw-pytorch-genomic-data/manifests/
```

Goal 4 Terraform bootstrap mirrors these paths on ParallelCluster nodes (see `PLAN_GOAL4_terraform_infra.md`).

## Midgut Tissue Filtering

Goal 2 targets insect midgut biology. Bulk RNA-seq QC supports an optional midgut subset:

```bash
python data/pipelines/modencode_rnaseq_qc.py \
  --tpm-tsv data/raw/modencode_rnaseq_tpm.tsv \
  --sample-metadata data/raw/modencode_rnaseq_samples.tsv \
  --midgut-only \
  --out data/tokenised/modencode_rnaseq_log1p.h5
```

Tissue labels matched (case-insensitive, substring): `midgut`, `gut`, `intestine`, `digestive`, `hindgut`, `foregut`, `proventriculus`.

The planned `fca_scrna_qc.py` pipeline (v2) will apply the same midgut filter on cell-type / tissue annotations in AnnData metadata.

## PPI / ESM-2 Boundary

PPI graphs are written with `x=None` node features. **Goal 2** owns ESM-2 embedding generation and attaches protein features at training time.

## Environment

Use [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
uv venv && uv sync          # from pyproject.toml
# or
uv pip install -r requirements.txt
```

Synthetic fixtures are acceptable for unit tests; integration runs assume S3-staged raw data via DFW.
