# Incremental training path for the midgut multi-modal model (Goal 2).
#
# Manifest v1 ships genome + bulk RNA-seq + PPI first. Additional modalities
# (scRNA, popgen, cell painting, Bt screening) are added as Goal 3 extends the
# DatasetManifest without breaking existing loaders.

## v1 modalities and tower mapping

| Manifest key | Goal 3 output | Goal 2 tower | Notes |
|--------------|---------------|--------------|-------|
| `genome` | `data/tokenised/dmel_genome_kmers.h5` | `GenomeKmerTransformerTower` | k-mer MLM pre-training |
| `rnaseq` | `data/tokenised/modencode_rnaseq_log1p.h5` | `ScRNATransformerTower` | Bulk samples; midgut tissue filter |
| `ppi` | `data/tokenised/dmel_ppi_graph.pt` | `PPIGraphTower` | Goal 3 ships `x=None`; Goal 2 fills ESM-2 |

Loaders skip modalities whose `output_path` is missing rather than failing.
Use `strict: true` in a tower config to require a specific modality.

## Midgut tissue alignment

Goal 3 retains midgut-labelled samples in RNA-seq/scRNA where possible.
`BulkRNASeqDataset` defaults to `tissue_filter="midgut"` and matches aliases:
`midgut`, `gut`, `intestine`, `hindgut`, `foregut`, `cardia`.

If no tissue metadata is present yet, all samples load with a warning.

## PPI node features (Goal 2 owns ESM-2)

Goal 3 `flybase_ppi_graph.py` writes PyG `Data` with `edge_index` and `gene_id`
but leaves `x=None`. Before GCN message passing:

1. Map `gene_id` → protein sequence (FlyBase FASTA; Goal 2 script TBD)
2. Tokenise sequences
3. Call `ProteinESM2Tower.encode_nodes()` or `PPIGraphTower.fill_node_features()`

## Incremental training stages

### Stage 1 — Per-modality pre-training (v1)

Run only configs whose manifest modalities are materialised:

```bash
# Genome k-mer MLM (when HDF5 exists)
python scripts/midgut/train_tower.py \
  --config models/midgut_multimodal/configs/pretrain_genome.yaml

# Bulk RNA-seq midgut subset → scRNA-style transformer
python scripts/midgut/train_tower.py \
  --config models/midgut_multimodal/configs/pretrain_rnaseq_midgut.yaml

# PPI graph with ESM-2 node fill
python scripts/midgut/train_tower.py \
  --config models/midgut_multimodal/configs/pretrain_ppi.yaml
```

### Stage 2 — Partial fusion (v1)

`fusion_train_v1.yaml` fuses whichever of genome / rnaseq / ppi are available.
`CrossAttentionFusion` accepts a variable-length embedding list (1–N modalities).
`LateFusionHead` uses `LazyLinear` so input width adapts to available towers.

```bash
python scripts/midgut/train_fusion.py \
  --config models/midgut_multimodal/configs/fusion_train_v1.yaml
```

### Stage 3 — Expand as Goal 3 adds modalities

When Goal 3 adds `scrna`, `popgen`, etc. to the manifest:

1. Tokenised `output_path` files appear on disk
2. `build_manifest_loaders()` auto-discovers them
3. Add tower checkpoints to `fusion_train.yaml` / downstream configs
4. No loader code changes required if schemas match Goal 3 contracts

## Config flags

| Flag | Location | Purpose |
|------|----------|---------|
| `manifest_path` | All train configs | DatasetManifest JSON |
| `required_modalities` | Tower configs | Fail if modality missing |
| `strict` | Tower configs | Raise on missing `output_path` |
| `tissue_filter` | RNA-seq configs | `"midgut"` or `null` for all tissues |
| `expected_modalities` | Fusion v1 config | Subset to attempt fusion |

## Dependencies

Install the `midgut` optional extra (see root `pyproject.toml`):

```bash
uv sync --extra midgut
# or: pip install -e ".[midgut]"
```

Goal 3 owns base curation deps (`biopython`, `h5py`, `scanpy`, …).
Goal 2 adds training deps via the `midgut` extra.
