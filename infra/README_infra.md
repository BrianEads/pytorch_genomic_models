# PyTorch Genomic Models — Cloud Infrastructure

Repeatable AWS infrastructure for multi-GPU model training using Terraform, AWS Service Catalog networking outputs, AWS ParallelCluster, and EC2 ImageBuilder.

## Dev environment — DFW AWS account

The **dev** environment targets a dedicated **DFW (data-fetch-wizard) dev/testing AWS account**. Raw and tokenised datasets are staged in S3 in this account — not on local disk.

| Component | Location |
|-----------|----------|
| data-fetch-wizard | [cs-cp-bifx-data-fetch-wizard](https://github.com/bayer-int/cs-cp-bifx-data-fetch-wizard) |
| Dev AWS account | DFW dev/testing account (newly provisioned) |
| Data bucket | `cs-cp-bifx-dfw-pytorch-genomic-data` |
| Terraform state | `cs-cp-bifx-dfw-pytorch-genomic-terraform-state` |

### S3 data layout (DFW account)

All data exchange between data-fetch-wizard, Goal 3 curation, and training uses a single bucket with fixed prefixes:

| S3 prefix | Written by | Read by | EFS mount (cluster) |
|-----------|------------|---------|---------------------|
| `raw/` | data-fetch-wizard | Goal 3 curation pipelines | `/mnt/efs/data/raw/` |
| `tokenised/` | Goal 3 curation pipelines | Goal 2 training scripts | `/mnt/efs/data/tokenised/` |
| `manifests/` | Goal 3 (`DatasetManifest` JSON) | Goal 2, validation | `/mnt/efs/data/manifests/` |
| `checkpoints/<run_id>/` | Goal 2 training (via Slurm) | Resume / eval | `/mnt/efs/checkpoints/<run_id>/` |

**data-fetch-wizard** downloads and stages raw files to `s3://cs-cp-bifx-dfw-pytorch-genomic-data/raw/` per recipe YAML (e.g. `recipes/dmel_foundation.yaml`). Goal 3 reads those raw files, runs QC/tokenisation in this repo, and writes outputs back to `tokenised/` and `manifests/`. On cluster boot, `on_node_start.sh` syncs all three prefixes from S3 to EFS on the head node.

Prod environment (separate Bayer account) uses `infra/terraform/envs/prod/` with bucket `pytorch-genomic-datasets`.

## Architecture summary

| Layer | Owner | This repo |
|-------|-------|-----------|
| VPC, subnets, VPC endpoints, NAT | Bayer Service Catalog | Read via SSM data sources |
| S3 datasets + checkpoints | Terraform `storage` module | Yes |
| EFS shared filesystem | Terraform `efs_mount` module | Yes |
| GPU AMI pipeline | Terraform `imagebuilder` module | Yes |
| Cost / idle monitoring | Terraform `monitoring` module | Yes |
| GPU compute cluster | ParallelCluster (`pcluster` CLI) | Config templates in `pcluster/` |

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Terraform | >= 1.5 | `brew install terraform` |
| AWS CLI v2 | latest | `brew install awscli` |
| pcluster CLI | >= 3.7 | `pip install aws-parallelcluster` |
| tflint | latest | `brew install tflint` |
| checkov | latest | `pip install checkov` |

**Before any `terraform apply` (DFW dev account):**

1. Configure AWS CLI credentials for the **DFW dev/testing account**
2. Verify Service Catalog SSM parameters exist in the DFW account:
   ```bash
   aws ssm get-parameters-by-path --path "/bayer/platform/networking/" --recursive
   ```
3. Create an EC2 key pair (SSH fallback; prefer SSM Session Manager)
4. Create remote state resources in the DFW account (one-time):

```bash
aws s3api create-bucket \
  --bucket cs-cp-bifx-dfw-pytorch-genomic-terraform-state \
  --region us-east-1
aws s3api put-bucket-versioning \
  --bucket cs-cp-bifx-dfw-pytorch-genomic-terraform-state \
  --versioning-configuration Status=Enabled
aws dynamodb create-table \
  --table-name cs-cp-bifx-dfw-pytorch-genomic-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

## Directory layout

```
infra/
├── terraform/          # Terraform root + modules (storage, efs_mount, imagebuilder, monitoring)
├── terraform/envs/     # dev (DFW account) and prod environment wrappers
├── pcluster/           # ParallelCluster cluster configs + bootstrap scripts
├── imagebuilder/       # Source YAML for ImageBuilder (mirrors terraform module components)
└── scripts/            # Slurm job submission and teardown helpers
```

## Deploy (dev — DFW account)

```bash
cd infra/terraform/envs/dev
cp terraform.tfvars.example terraform.tfvars   # edit alert_email, key_pair_name
terraform init
terraform plan
terraform apply
```

## Create ParallelCluster cluster

```bash
cd infra/pcluster
pcluster create-cluster --cluster-name pytorch-genomic-dev \
  --cluster-configuration cluster_config_dev.yaml
```

Access the head node via SSM Session Manager (no bastion required).

## Submit a training job

From the cluster head node (data already synced to `/mnt/efs/data/tokenised/`):

```bash
export S3_CHECKPOINT_BUCKET=cs-cp-bifx-dfw-pytorch-genomic-data
./infra/scripts/submit_training_job.sh scripts/midgut/train_fusion.py \
  --config configs/fusion_train.yaml
```

## Data flow (DFW integration)

```
data-fetch-wizard                    Goal 3 (this repo)              Goal 2 training
      │                                    │                              │
      ▼                                    ▼                              ▼
s3://…/raw/  ──► curation pipelines ──► s3://…/tokenised/  ──► on_node_start.sh ──► /mnt/efs/data/
                  (reads manifests)      s3://…/manifests/       (head node sync)      tokenised/
```

Goal 2 training scripts should read tokenised data from `/mnt/efs/data/tokenised/`, resolve paths via `DatasetManifest` in `/mnt/efs/data/manifests/`, accept `--checkpoint-dir`, and checkpoint every N steps for spot requeue safety.

## Destroy cleanly

```bash
./infra/scripts/teardown.sh pytorch-genomic-dev --terraform-env dev
```

EFS mount targets must be removed before `terraform destroy` to avoid orphaned charges.

## Lint and security

```bash
cd infra/terraform
terraform fmt -recursive
tflint --recursive
checkov -d . --framework terraform
```

## Open questions (blockers for `terraform apply`)

- Confirm Service Catalog SSM parameter paths in the **DFW account** (may differ from Bayer corporate defaults)
- Confirm IAM permission boundary scope in the DFW account
- Align exact S3 key layout with data-fetch-wizard output conventions once DFW recipes are finalised

See `PLAN_GOAL4_terraform_infra.md` for the full open-questions list.
