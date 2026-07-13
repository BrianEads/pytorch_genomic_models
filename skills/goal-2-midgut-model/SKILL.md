---
name: goal-2-midgut-model
description: >-
  Build and train the insect midgut multi-modal model (Goal 2). Use for towers,
  fusion, downstream heads, manifest loaders, and training scripts under
  models/midgut_multimodal/ and scripts/midgut/.
---

# Goal 2 — Midgut Model Builder

**Branch:** `feat/goal-2-midgut-model`
**Plan:** [PLAN_GOAL2_midgut_model.md](../../PLAN_GOAL2_midgut_model.md)
**Agent name:** `midgut-model-builder`

## Scope

- `models/midgut_multimodal/` — towers, fusion, downstream, configs, data loaders
- `scripts/midgut/` — `train_tower.py`, `train_fusion.py`, `eval_downstream.py`
- `tests/test_midgut_towers.py`, `tests/test_manifest_loaders.py`
- `models/midgut_multimodal/INCREMENTAL_TRAINING.md`

**Out of scope:** raw download/API (DFW), tokenisation pipelines (Goal 3), Terraform (Goal 4).

## Key Boundaries

- **ESM-2 ownership:** Goal 2 fills PPI graph node features (`ProteinESM2Tower`, `PPIGraphTower.fill_node_features`).
- **Incremental v1:** Train on available manifest modalities only (`genome`, `rnaseq`, `ppi`); loaders skip missing paths unless `strict: true`.
- **Midgut filter:** `BulkRNASeqDataset` defaults to `tissue_filter="midgut"`.

## Acceptance Criteria Checklist

- [x] Tower stubs forward-pass on CPU with documented shapes
- [x] `CrossAttentionFusion` + `LateFusionHead` (1–N modalities)
- [x] Downstream heads (IC50, resistance, Cry, perturbation)
- [x] Training configs including `fusion_train_v1.yaml`, `pretrain_*`
- [x] CLI training scripts with `--config`, DDP via `torchrun`
- [x] Manifest-driven loaders (`load_manifest`, `build_manifest_loaders`)
- [x] Unit tests pass: `pytest tests/test_midgut_towers.py tests/test_manifest_loaders.py -v`
- [ ] MLflow logging verified on local synthetic run
- [ ] Gene ID → protein FASTA helper for PPI node fill
- [ ] Real pre-training on tokenised data (**blocked: M3 / P0**)
- [ ] PR to `main`

## Coordination Rules

- Read manifest from `data/manifests/dmel_foundation_manifest.json`; do not hardcode data paths.
- Schema changes in Goal 3 require updating loaders + tests here.
- Fusion: late fusion baseline first (D6); cross-attention available for incremental v1.
- Never add download logic — reference `fetch_recipe` in manifest only.

## When to Pause / Escalate

| Trigger | Action |
|---------|--------|
| **P0 active** — no tokenised HDF5 in S3/local | Local unit tests only; no GPU cluster jobs |
| Missing modality data after M3 | Train partial fusion per `INCREMENTAL_TRAINING.md` |
| Need internal toxin DB / cell painting data | Escalate Q2/Q3 to user |
| Manifest schema break | Coordinate with Goal 3 + conductor |
| ESM-2 full CRY length > 1024 aa | Document chunking approach; escalate if blocking |

## Status Board Update (after session)

```
Goal 2 | branch | completed: [...] | blocked: [...] | next: [...]
```

## Dependencies

| Upstream | Artefact |
|----------|----------|
| Goal 3 | `DatasetManifest`, tokenised HDF5/PT |
| Goal 3 | v1 modalities: genome, rnaseq, ppi |
| Goal 4 | ParallelCluster, EFS/S3 sync for large GPU runs (M4) |
| DFW | Raw staging (indirect via Goal 3) |

## Primary Skills

`multi-modal-fusion`, `protein-modeling`, `experiment-tracking`, `ddp-training` — see [SKILLS.md](../../SKILLS.md).
