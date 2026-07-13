---
name: goal-4-infra-provisioner
description: >-
  Provision and maintain AWS infrastructure for GPU training (Terraform + Service
  Catalog, ParallelCluster, ImageBuilder). Enforce pause-point P0 (no apply) and
  align S3/EFS paths with Goal 3 manifests and Goal 2 training scripts.
---

# Goal 4 — Infra Provisioner (Terraform + Service Catalog)

**Branch:** `feat/goal-4-terraform-infra`  
**Plan:** [PLAN_GOAL4_terraform_infra.md](../../PLAN_GOAL4_terraform_infra.md)  
**Agent name:** `infra-provisioner`

## Scope

- `infra/terraform/` — root module + modules (`storage`, `efs_mount`, `imagebuilder`, `monitoring`) + env wrappers
- `infra/pcluster/` — ParallelCluster configs + custom actions
- `infra/imagebuilder/` — ImageBuilder component YAMLs
- `infra/scripts/` — job submission + teardown helpers
- `infra/README_infra.md`

**Out of scope (owned by platform / prerequisites):**
- Creating VPCs/subnets/endpoints (Service Catalog owns)
- Defining the IAM permission boundary (platform owns)
- Running DataFetch-Wizard fetch recipes (DFW owns)

## Program Gate: P0 (enforce)

**P0 = user not ready for DFW AWS apply.** Until explicitly cleared:

- **Do not run** `terraform apply`, `pcluster create-cluster`, ImageBuilder pipeline executions, or any action that creates/updates cloud resources.
- Allowed: `terraform fmt/validate`, `tflint`, `checkov`, doc updates, config refactors, dependency alignment, dry-run validation.

## Dev Account Alignment (resolved)

Dev/testing staging uses the DFW account and bucket:

- **S3 bucket:** `cs-cp-bifx-dfw-pytorch-genomic-data`
- **Prefix layout:** `raw/` → `tokenised/` + `manifests/` → `checkpoints/`

The `on_node_start.sh` contract should sync:

- `s3://cs-cp-bifx-dfw-pytorch-genomic-data/raw/` → `/mnt/efs/data/raw/`
- `s3://cs-cp-bifx-dfw-pytorch-genomic-data/tokenised/` → `/mnt/efs/data/tokenised/`
- `s3://cs-cp-bifx-dfw-pytorch-genomic-data/manifests/` → `/mnt/efs/data/manifests/`

## Acceptance Criteria Checklist

### Scaffold (M0/M1)

- [x] `infra/` directory exists with terraform + pcluster + scripts + docs
- [x] Terraform modules scaffolded: `storage`, `efs_mount`, `imagebuilder`, `monitoring`
- [x] Env wrappers exist under `infra/terraform/envs/` (`dev`, `prod`)
- [x] ParallelCluster configs present under `infra/pcluster/` (`cluster_config_dev.yaml`, `cluster_config_prod.yaml`)
- [x] Bootstrap scripts present: `on_node_start.sh`, `on_node_configured.sh`
- [x] `infra/README_infra.md` documents DFW dev account + S3 prefix layout

### Validation (M1)

- [ ] `terraform fmt --check` passes under `infra/terraform/`
- [ ] `terraform validate` passes (dev env)
- [ ] `tflint --recursive` passes (no errors)
- [ ] `checkov` has no HIGH/CRITICAL (or exceptions documented)
- [ ] `pcluster` config validates (dry-run / validate mode only)
- [ ] `on_node_start.sh` is idempotent (safe to run multiple times)

### Provisioning (blocked by P0/M2)

- [ ] Remote state bucket + DynamoDB lock table confirmed/created (per plan)
- [ ] Service Catalog SSM parameter paths confirmed in dev account
- [ ] ImageBuilder IAM instance profile implemented within boundary
- [ ] ImageBuilder AMI pipeline successfully builds + publishes AMI ID/ARN to SSM
- [ ] ParallelCluster dev cluster created + accessible via SSM
- [ ] End-to-end smoke: S3 sync → EFS mount → single-GPU `torchrun` job

## Coordination Rules (critical)

1. **Single source of truth for data paths:** S3 prefixes and EFS mount paths must match:
   - Goal 3 manifests (`data/manifests/*.json`)
   - Goal 2 loaders/train scripts (`models/midgut_multimodal/data/*`, `scripts/midgut/*`)
2. **No “helpful” downloads:** if data is missing, escalate to DFW—don’t add curl/wget flows.
3. **Dependency contract:** prefer the repo `pyproject.toml` + `uv sync` extras; do not introduce separate env files unless required by ImageBuilder (and if so, generate from `pyproject.toml`).
4. **Least privilege:** keep IAM changes minimal and boundary-compliant; document all required actions in `infra/README_infra.md`.

## When to Pause / Escalate

| Trigger | Action |
|---------|--------|
| **P0 active** | Stop at lint/validate/docs; do not create resources |
| SSM paths unknown | Request platform team confirmation; keep placeholders clearly marked |
| Bucket/prefix mismatch with manifests | Notify conductor + Goal 3; update scripts/configs to converge |
| Checkov HIGH/CRITICAL | Hold for review; propose mitigations or exception process |

## Status Board Update (after each session)

```
Goal 4 | branch | completed: [...] | blocked: [...] | next: [...]
```

## Dependencies

- **Goal 3:** needs stable S3 prefix contract and (later) tokenised artefacts in `tokenised/` + manifests in `manifests/`
- **Goal 2:** needs `/mnt/efs/data/` and `/mnt/efs/checkpoints/` conventions and Slurm submission scripts
- **Platform team:** Service Catalog SSM outputs + permission boundary review
