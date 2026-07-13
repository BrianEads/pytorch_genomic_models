# DECISIONS.md — Cross-Cutting Decision Log

Canonical record of program decisions. The conductor maintains this file; goal agents append entries when user or platform decisions land.

**Format:** Newest decisions at the top of each section.

---

## Resolved

| Date | ID | Decision | Rationale | Goals |
|------|-----|----------|-----------|-------|
| 2026-07 | D9 | **DFW AWS apply paused** until user explicitly ready | New empty DFW account; alpha evaluation must precede large-scale testing | 4, 5 |
| 2026-07 | D5 | DFW dev bucket `cs-cp-bifx-dfw-pytorch-genomic-data` with fixed prefixes (`raw/`, `tokenised/`, `manifests/`, `checkpoints/`) | Single contract between DFW, Goal 3, Goal 4, Goal 2 | 3, 4 |
| 2026-07 | D1 | Incremental v1 manifest ships `genome`, `rnaseq`, `ppi` only | Unblocks Goal 2 training before v2 modalities exist | 2, 3 |
| 2026-07 | D2 | Goal 2 owns ESM-2; Goal 3 PPI pipeline writes `x=None` + `node_features_owner: goal2_esm2` | Avoid duplicate protein LM work; clear boundary | 2, 3 |
| 2026-07 | D3 | Bulk RNA-seq loader defaults to midgut tissue filter | Aligns with Bt/midgut biological focus | 2 |
| 2026-07 | D4 | `pyproject.toml` + `uv sync` as canonical deps; `requirements.txt` mirrors base | Reproducible local dev across goals | 1, 3 |
| 2026-07 | D6 | Late fusion (Option A) deployed first; cross-attention (Option B) also implemented | Baseline simplicity + preferred path available | 2 |
| 2026-07 | D7 | Secret/internal data stays out of model weights | Broad reusability; private retrieval/scoring/validation only | 2, 5 |
| 2026-07 | D8 | Terraform wraps Service Catalog; no raw VPC/subnet creation | Bayer platform governance | 4 |
| 2026-07 | D12 | DFW dev account first; prod Bayer account separate | Isolated alpha testing | 4 |

---

## Open

| ID | Question | Owner | Notes |
|----|----------|-------|-------|
| D10 | Internal microbial toxin DB access | User | Would enable protein-evolution / directed engineering track |
| D11 | MoE fusion (Option C) vs cross-attention only | User / Goal 2 | Deferred until modality availability clearer |
| D13 | Notebook packaging as separable module for external env | User | Goal 1 split (`e2e_explorer.md` + `.ipynb`) is first step |
| D14 | S3 Files (EFS-backed) vs classic EFS for some workflows | Goal 4 / platform | Investigation item from oversight plan |
| D15 | First ODM/GeneStack accession experiment scope | User / DFW | Tied to M5 alpha evaluation |

---

## Superseded

_None yet._
