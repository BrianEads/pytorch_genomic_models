# Goal 4 — Cloud Infrastructure (Terraform + Service Catalog)

**Branch:** `feat/goal-4-terraform-infra`
**Agent:** `infra-provisioner`

---

## Goal Summary

Provision repeatable, cost-controlled AWS cloud infrastructure using Terraform so that multi-GPU model training (Goals 2 and 3) can be launched on-demand and graduated away from the constraints of a local M1 MacBook Pro. Infrastructure provisioning wraps **AWS Service Catalog products** rather than raw AWS resources — this aligns with Bayer's internal governance model and avoids duplicating network and security configurations that are already standardised by the platform team. Parallel GPU compute is orchestrated via **AWS ParallelCluster**, and custom GPU AMIs are built and versioned through **EC2 ImageBuilder**.

---

## Motivation — M1 MacBook Pro Constraints

| Constraint | Impact |
|-----------|--------|
| No discrete CUDA GPU (Apple Silicon unified memory) | Cannot run CUDA-accelerated training; PyTorch MPS backend is limited and not production-ready for large models |
| 16–64 GB unified memory shared between CPU and GPU | Insufficient for large-batch pre-training or multi-modal fusion with 650M+ parameter models |
| No NVLink / PCIe multi-GPU | Cannot use `DistributedDataParallel` (DDP) or tensor parallelism |
| Single node only | No multi-node training (NCCL collectives) |

### What cloud compute unlocks

- **Mixed precision (fp16/bf16)** with `torch.cuda.amp` — 2–4× throughput improvement on A100/V100 vs. fp32
- **DDP across multiple GPUs** via ParallelCluster managed job scheduler
- **Large-batch pre-training** — critical for genomic LM convergence (DNABERT uses batch ≥ 256 on 8× V100)
- **Spot instance fleet** managed by ParallelCluster for cost-optimised hyperparameter sweeps
- **Persistent EFS** for shared dataset access across a training cluster without per-node S3 sync latency
- **Private networking** with no direct public internet exposure, consistent with Bayer network security policy

---

## AWS Service Catalog Strategy

Bayer's AWS environment provisions foundational resources (VPCs, subnets, security baselines, IAM boundaries) through **AWS Service Catalog products**. Terraform wraps these products as data sources and outputs rather than creating the underlying resources from scratch.

### What Service Catalog owns vs. what Terraform owns

| Resource | Owner | Rationale |
|----------|-------|-----------|
| VPC and private subnet layout | Service Catalog product | Network design is standardised across the platform; duplication creates compliance risk |
| VPC endpoints (S3, SSM, ECR, CloudWatch, STS) | Service Catalog product | Endpoint policy and routing are centrally managed |
| Internet Gateway and NAT Gateway | Service Catalog product | Egress controls are governed at the platform level |
| Bayer network peering / VPN / Direct Connect attachment | Service Catalog product | Cross-environment routing requires network team approval and is provisioned centrally |
| IAM permission boundaries | Service Catalog product | Enforces least-privilege guardrails that Terraform resources must operate within |
| EC2 key pairs | User-managed pre-requisite | Each researcher creates their own key pair in the account |
| S3 bucket for datasets and checkpoints | Terraform (`storage` module) | Project-specific; scoped to the permission boundary provided by Service Catalog |
| EFS filesystem and mount targets | Terraform (`efs_mount` module) | Project-specific; uses subnets output by Service Catalog |
| ParallelCluster configuration | Terraform + `pcluster` CLI | Compute-layer concern; references VPC and subnet IDs from Service Catalog outputs |
| EC2 ImageBuilder pipelines | Terraform (`imagebuilder` module) | Project-specific GPU AMI recipes |
| CloudWatch alarms and budget alerts | Terraform (`monitoring` module) | Project-specific cost controls |

### Consuming Service Catalog outputs in Terraform

Service Catalog provisioned products expose their outputs via AWS Systems Manager Parameter Store or CloudFormation stack outputs. Terraform reads these as data sources:

```hcl
# Read VPC and subnet IDs written by the Service Catalog networking product
data "aws_ssm_parameter" "vpc_id" {
  name = "/bayer/platform/networking/vpc_id"
}

data "aws_ssm_parameter" "private_subnet_ids" {
  name = "/bayer/platform/networking/private_subnet_ids"
}

data "aws_ssm_parameter" "s3_endpoint_id" {
  name = "/bayer/platform/networking/s3_vpc_endpoint_id"
}
```

All Terraform modules must use these data sources rather than hardcoding VPC/subnet IDs.

---

## Networking Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Bayer AWS Account (Service Catalog managed)                         │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  VPC  (10.x.x.x/16 — allocated by Service Catalog)          │    │
│  │                                                             │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │    │
│  │  │ Private       │  │ Private       │  │ Private       │      │    │
│  │  │ Subnet AZ-a   │  │ Subnet AZ-b   │  │ Subnet AZ-c   │      │    │
│  │  │ (compute)     │  │ (compute)     │  │ (storage/EFS) │      │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │    │
│  │         │                 │                 │               │    │
│  │  ┌──────▼─────────────────▼─────────────────▼───────────┐   │    │
│  │  │  VPC Endpoints (Gateway + Interface)                  │   │    │
│  │  │  • S3 Gateway endpoint (no data egress cost)          │   │    │
│  │  │  • SSM, SSM Messages, EC2 Messages (interface)        │   │    │
│  │  │  • ECR API + ECR DKR (interface, for container pulls) │   │    │
│  │  │  • CloudWatch Logs + Monitoring (interface)           │   │    │
│  │  │  • STS (interface, for IAM role assumption)           │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  │                                                             │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │  Transit Gateway / VPN attachment                     │   │    │
│  │  │  → Bayer corporate network (on-prem)                  │   │    │
│  │  │  → Other Bayer AWS accounts (hub-and-spoke)           │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Internet Gateway (managed by Service Catalog)                       │
│  NAT Gateway in public subnet → private subnet egress                │
└──────────────────────────────────────────────────────────────────────┘
```

### Key networking constraints

- All compute nodes (ParallelCluster head node and compute fleet) run in **private subnets only**.
- Outbound internet access is via the Service Catalog-managed **NAT Gateway** (for pip installs, public dataset downloads during bootstrap). Direct internet inbound is not permitted.
- SSH access is via **AWS Systems Manager Session Manager** (no bastion host required; no port 22 open). The SSM VPC endpoint enables this from private subnets.
- S3 traffic (dataset sync, checkpoint writes) travels via the **S3 Gateway VPC endpoint** — no NAT charges, no internet exposure.
- Container image pulls (if using ECR for job containers) use the **ECR interface endpoints**.

---

## AWS ParallelCluster for GPU Compute

**AWS ParallelCluster** replaces the hand-rolled spot fleet and DDP launch scripts from earlier drafts. It provides:
- Slurm job scheduler with automatic compute node scaling (scale-out on job submit, scale-in on idle)
- Native spot instance support with automatic requeue on interruption
- Multi-instance-type queues (different queues for dev, training, hyperparameter sweep)
- EFS shared filesystem integration
- Built-in support for `torchrun` and MPI-based distributed training via Slurm job steps

### ParallelCluster configuration layout

```
infra/
├── pcluster/
│   ├── cluster_config.yaml          # Main ParallelCluster config (Slurm + queues)
│   ├── cluster_config_dev.yaml      # Dev variant (single g4dn.xlarge, on-demand)
│   ├── cluster_config_prod.yaml     # Prod variant (multi-queue: train + sweep)
│   └── custom_actions/
│       ├── on_node_start.sh         # Bootstrap: project deps, EFS mount, S3 sync
│       └── on_node_configured.sh    # Post-config: verify CUDA, MLflow agent start
```

### Example ParallelCluster queue layout

```yaml
# infra/pcluster/cluster_config.yaml (key sections)
Region: us-east-1
Image:
  Os: alinux2
  CustomAmi: !Sub "{{resolve:ssm:/bayer/pytorch-genomic/ami/gpu-training-latest}}"

HeadNode:
  InstanceType: m5.xlarge
  Networking:
    SubnetId: !Sub "{{resolve:ssm:/bayer/platform/networking/private_subnet_ids_az_a}}"
  Ssh:
    KeyName: !Ref KeyPairName  # fallback; prefer SSM Session Manager

Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: dev
      CapacityType: ONDEMAND
      Networking:
        SubnetIds:
          - !Sub "{{resolve:ssm:/bayer/platform/networking/private_subnet_ids_az_a}}"
      ComputeResources:
        - Name: g4dn-xlarge
          InstanceType: g4dn.xlarge
          MinCount: 0
          MaxCount: 2

    - Name: train
      CapacityType: SPOT
      Networking:
        SubnetIds:
          - !Sub "{{resolve:ssm:/bayer/platform/networking/private_subnet_ids_az_a}}"
          - !Sub "{{resolve:ssm:/bayer/platform/networking/private_subnet_ids_az_b}}"
      ComputeResources:
        - Name: p3-2xlarge
          InstanceType: p3.2xlarge
          MinCount: 0
          MaxCount: 8
          SpotPrice: "1.50"
        - Name: p4d-24xlarge
          InstanceType: p4d.24xlarge
          MinCount: 0
          MaxCount: 2
          SpotPrice: "13.00"

    - Name: sweep
      CapacityType: SPOT
      Networking:
        SubnetIds:
          - !Sub "{{resolve:ssm:/bayer/platform/networking/private_subnet_ids_az_a}}"
          - !Sub "{{resolve:ssm:/bayer/platform/networking/private_subnet_ids_az_b}}"
      ComputeResources:
        - Name: p3-2xlarge-sweep
          InstanceType: p3.2xlarge
          MinCount: 0
          MaxCount: 16
          SpotPrice: "1.50"

SharedStorage:
  - Name: efs-data
    StorageType: Efs
    MountDir: /mnt/efs
    EfsSettings:
      FileSystemId: !Sub "{{resolve:ssm:/bayer/pytorch-genomic/efs/filesystem_id}}"
```

### Slurm job submission examples

Single-node training:
```bash
sbatch --partition=train --nodes=1 --gres=gpu:4 \
  --wrap="torchrun --nproc_per_node=4 scripts/midgut/train_fusion.py --config configs/fusion_train.yaml"
```

Multi-node DDP:
```bash
sbatch --partition=train --nodes=4 --ntasks-per-node=8 --gres=gpu:8 \
  scripts/midgut/slurm_ddp_wrapper.sh train_fusion.py --config configs/fusion_train.yaml
```

Hyperparameter sweep (Optuna with Slurm job array):
```bash
sbatch --partition=sweep --array=0-31 --nodes=1 --gres=gpu:1 \
  --wrap="python scripts/midgut/optuna_worker.py --trial-id \$SLURM_ARRAY_TASK_ID"
```

---

## EC2 ImageBuilder for Custom GPU AMIs

**AWS EC2 ImageBuilder** builds, tests, and publishes versioned AMIs containing the full software stack. This avoids repeated bootstrap time (10–15 min per node) and guarantees environment reproducibility across the cluster.

### ImageBuilder pipeline layout

```
infra/
├── imagebuilder/
│   ├── components/
│   │   ├── cuda_install.yaml        # CUDA 12.1 + cuDNN 8 component
│   │   ├── miniconda_env.yaml       # Miniconda + pytorch-genomic conda env
│   │   ├── pytorch_stack.yaml       # torch, torchvision, transformers, fair-esm, etc.
│   │   ├── project_deps.yaml        # biopython, scanpy, torch_geometric, h5py, etc.
│   │   └── efs_utils.yaml           # amazon-efs-utils, nfs-common
│   ├── image_recipe.yaml            # Assembles components in order
│   ├── pipeline.yaml                # Build schedule + distribution config
│   └── infrastructure_config.yaml   # Instance type, subnet, IAM role for builder
```

### Component execution order

```
Base: Amazon Linux 2 (GPU-optimised, kernel with NVIDIA drivers)
  └─▶ cuda_install          (CUDA 12.1, cuDNN 8)
      └─▶ miniconda_env     (Miniconda 3, conda env from environment.yml)
          └─▶ pytorch_stack (torch 2.x cu121, torchvision, torchaudio)
              └─▶ project_deps (all Python deps for Goals 2 and 3)
                  └─▶ efs_utils (amazon-efs-utils for EFS mount at boot)
```

### AMI versioning and consumption

- ImageBuilder publishes the AMI ARN to SSM Parameter Store at `/bayer/pytorch-genomic/ami/gpu-training-latest` after every successful build.
- The ParallelCluster config references this SSM parameter as `CustomAmi` (see cluster config above), so re-deploying the cluster after a new AMI build automatically picks up the latest image.
- AMI builds are triggered on a scheduled pipeline (weekly) or on push to `main` via a CI step.

```hcl
# infra/terraform/modules/imagebuilder/main.tf (excerpt)
resource "aws_imagebuilder_image_pipeline" "gpu_training" {
  name                             = "${var.project_name}-gpu-training"
  image_recipe_arn                 = aws_imagebuilder_image_recipe.gpu_training.arn
  infrastructure_configuration_arn = aws_imagebuilder_infrastructure_configuration.builder.arn
  distribution_configuration_arn   = aws_imagebuilder_distribution_configuration.gpu_training.arn
  schedule {
    schedule_expression = "cron(0 4 ? * SUN *)"  # Weekly Sunday 04:00 UTC
  }
  tags = local.common_tags
}

resource "aws_imagebuilder_distribution_configuration" "gpu_training" {
  name = "${var.project_name}-gpu-training-dist"
  distribution {
    region = var.aws_region
    ami_distribution_configuration {
      name = "${var.project_name}-gpu-training-{{ imagebuilder:buildDate }}"
    }
  }
}

# Publish latest AMI ID to SSM after each build
resource "aws_ssm_parameter" "ami_latest" {
  name  = "/bayer/pytorch-genomic/ami/gpu-training-latest"
  type  = "String"
  value = aws_imagebuilder_image_pipeline.gpu_training.arn  # updated by EventBridge on build completion
}
```

---

## Full Terraform Module Directory Tree

```
infra/
├── terraform/
│   ├── main.tf                      # Root module: provider, backend, module calls
│   ├── variables.tf                 # All input variables (see reference below)
│   ├── outputs.tf                   # S3 bucket ARN, EFS ID, ParallelCluster config path
│   ├── backend.tf                   # S3 + DynamoDB remote state config
│   ├── .gitignore                   # Exclude *.tfstate, *.tfstate.backup, .terraform/
│   ├── versions.tf                  # Required Terraform and provider versions
│   ├── data_sources.tf              # SSM parameter reads for Service Catalog outputs
│   ├── modules/
│   │   ├── storage/                 # S3 bucket for datasets and checkpoints
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   ├── efs_mount/               # EFS filesystem + mount targets in private subnets
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   ├── imagebuilder/            # EC2 ImageBuilder pipeline for GPU AMIs
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── outputs.tf
│   │   │   └── components/          # ImageBuilder component YAML files (uploaded as aws_imagebuilder_component)
│   │   │       ├── cuda_install.yaml
│   │   │       ├── miniconda_env.yaml
│   │   │       ├── pytorch_stack.yaml
│   │   │       ├── project_deps.yaml
│   │   │       └── efs_utils.yaml
│   │   └── monitoring/              # CloudWatch alarms, budget alert, SNS topic
│   │       ├── main.tf
│   │       ├── variables.tf
│   │       └── outputs.tf
│   └── envs/
│       ├── dev/                     # Single g4dn.xlarge queue, on-demand
│       │   ├── main.tf
│       │   ├── terraform.tfvars.example
│       │   └── README.md
│       └── prod/                    # Multi-queue spot: train + sweep
│           ├── main.tf
│           ├── terraform.tfvars.example
│           └── README.md
├── pcluster/
│   ├── cluster_config.yaml          # ParallelCluster config template
│   ├── cluster_config_dev.yaml
│   ├── cluster_config_prod.yaml
│   └── custom_actions/
│       ├── on_node_start.sh         # Runs on every node at boot
│       └── on_node_configured.sh    # Runs after ParallelCluster configuration step
├── imagebuilder/
│   ├── components/                  # Source YAML for each ImageBuilder component
│   ├── image_recipe.yaml
│   ├── pipeline.yaml
│   └── infrastructure_config.yaml
├── scripts/
│   ├── submit_training_job.sh       # Slurm sbatch wrapper for training runs
│   ├── submit_sweep_job.sh          # Slurm job array wrapper for Optuna sweeps
│   └── teardown.sh                  # Checkpoint sync + cluster delete
└── README_infra.md
```

---

## Instance Selection Guide

| Use case | Instance | vCPUs | GPU | GPU RAM | Spot $/hr | ParallelCluster queue |
|----------|----------|-------|-----|---------|-----------|----------------------|
| Development / notebook / debugging | `g4dn.xlarge` | 4 | 1× T4 | 16 GB | ~$0.16 | `dev` (on-demand) |
| Single-modality pre-training (medium) | `p3.2xlarge` | 8 | 1× V100 | 16 GB | ~$0.90 | `train` (spot) |
| Single-modality pre-training (large) | `p3.8xlarge` | 32 | 4× V100 | 64 GB | ~$3.60 | `train` (spot) |
| Multi-modal fusion training | `p3.16xlarge` | 64 | 8× V100 | 128 GB | ~$7.20 | `train` (spot) |
| Full foundation pre-training | `p4d.24xlarge` | 96 | 8× A100 | 320 GB | ~$10–13 | `train` (spot) |
| Hyperparameter sweep (parallel) | `p3.2xlarge` ×4–16 | — | — | — | ~$3.60/node | `sweep` (spot, job array) |

---

## Terraform Variables Reference

Full content of `infra/terraform/variables.tf`:

```hcl
variable "aws_region" {
  type        = string
  description = "AWS region to deploy resources into."
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Tag prefix applied to all resources for cost allocation and identification."
  default     = "pytorch-genomic"
}

variable "environment" {
  type        = string
  description = "Deployment environment label: 'dev' or 'prod'."
  default     = "dev"
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be 'dev' or 'prod'."
  }
}

variable "key_pair_name" {
  type        = string
  description = "Name of an existing EC2 key pair for SSH fallback. Primary access is via SSM Session Manager."
  default     = ""
}

variable "s3_dataset_bucket" {
  type        = string
  description = "Name of the S3 bucket for storing datasets and training checkpoints."
  default     = "pytorch-genomic-datasets"
}

variable "enable_efs" {
  type        = bool
  description = "Whether to provision an EFS filesystem for shared multi-node dataset access."
  default     = true
}

variable "efs_throughput_mode" {
  type        = string
  description = "EFS throughput mode: 'bursting' or 'provisioned'."
  default     = "bursting"
}

variable "imagebuilder_schedule" {
  type        = string
  description = "Cron schedule expression for the ImageBuilder pipeline. Default: weekly Sunday 04:00 UTC."
  default     = "cron(0 4 ? * SUN *)"
}

variable "budget_alert_usd" {
  type        = number
  description = "Monthly cost threshold in USD for the CloudWatch budget alert."
  default     = 200
}

variable "auto_stop_idle_minutes" {
  type        = number
  description = "Consecutive minutes of ParallelCluster head node CPU idle before auto-stop alarm fires."
  default     = 60
}

# SSM parameter paths — written by Service Catalog products; read as data sources
variable "ssm_vpc_id_path" {
  type        = string
  description = "SSM Parameter Store path for the VPC ID provisioned by Service Catalog."
  default     = "/bayer/platform/networking/vpc_id"
}

variable "ssm_private_subnet_ids_path" {
  type        = string
  description = "SSM Parameter Store path for comma-separated private subnet IDs."
  default     = "/bayer/platform/networking/private_subnet_ids"
}

variable "ssm_s3_endpoint_id_path" {
  type        = string
  description = "SSM Parameter Store path for the S3 VPC Gateway endpoint ID."
  default     = "/bayer/platform/networking/s3_vpc_endpoint_id"
}
```

---

## Bootstrap Script Spec (`on_node_start.sh`)

With ImageBuilder, the full software stack is baked into the AMI. The `on_node_start.sh` custom action performs only **cluster-specific runtime setup** that cannot be baked in (EFS mount, S3 sync of latest datasets, MLflow agent start):

```bash
#!/bin/bash
# on_node_start.sh — ParallelCluster custom action: runs on every compute node at boot.
# Software stack is pre-installed in the custom AMI by ImageBuilder.
# Env vars injected by ParallelCluster: PCLUSTER_NODE_TYPE, PCLUSTER_SHARED_DIRS

set -euo pipefail

EFS_DNS="${EFS_DNS:?EFS_DNS env var must be set in ParallelCluster config}"
S3_DATASET_BUCKET="${S3_DATASET_BUCKET:?}"
PROJECT_NAME="${PROJECT_NAME:-pytorch-genomic}"

# 1. Mount EFS (amazon-efs-utils is pre-installed in AMI)
if ! mountpoint -q /mnt/efs; then
  mkdir -p /mnt/efs
  mount -t efs -o tls,iam "${EFS_DNS}:/" /mnt/efs
  echo "${EFS_DNS}:/ /mnt/efs efs tls,iam,_netdev 0 0" >> /etc/fstab
fi

# 2. Sync reference datasets from S3 to EFS (head node only; compute nodes use shared EFS)
if [[ "${PCLUSTER_NODE_TYPE}" == "HeadNode" ]]; then
  aws s3 sync "s3://${S3_DATASET_BUCKET}/data/" /mnt/efs/data/ \
    --exclude "*.raw" --only-show-errors
fi

# 3. Activate conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate pytorch-genomic

# 4. Log readiness
echo "Node $(hostname) ready at $(date) — AMI: $(curl -s http://169.254.169.254/latest/meta-data/ami-id)" \
  | tee /var/log/node_ready.log
```

---

## DDP / Slurm Launch Wrapper Spec

`infra/scripts/submit_training_job.sh` — wraps `sbatch` for a distributed training run:

```bash
#!/bin/bash
# submit_training_job.sh — Submit a distributed training job to Slurm on ParallelCluster.
# Usage: ./submit_training_job.sh <training_script.py> [--args ...]
#   QUEUE     - Slurm partition name (default: train)
#   NUM_NODES - Number of nodes (default: 1)
#   NUM_GPUS  - GPUs per node (default: 1)

set -euo pipefail

TRAINING_SCRIPT="${1:?ERROR: pass training script as first argument}"
shift

QUEUE="${QUEUE:-train}"
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:?}"

# Generate inline batch script for torchrun
sbatch \
  --partition="${QUEUE}" \
  --nodes="${NUM_NODES}" \
  --ntasks-per-node="${NUM_GPUS}" \
  --gres="gpu:${NUM_GPUS}" \
  --job-name="pytorch-genomic-${RUN_ID}" \
  --output="/mnt/efs/logs/slurm-%j.out" \
  --error="/mnt/efs/logs/slurm-%j.err" \
  << SBATCH_SCRIPT
#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate pytorch-genomic

# SIGTERM handler for spot interruption
trap 'echo "SIGTERM received — syncing checkpoints..."; \
      aws s3 sync /mnt/efs/checkpoints/${RUN_ID}/ \
        s3://${S3_CHECKPOINT_BUCKET}/${RUN_ID}/emergency/ --only-show-errors' SIGTERM

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes=${NUM_NODES} \
  --node_rank=\${SLURM_NODEID} \
  --master_addr=\$(scontrol show hostnames \${SLURM_JOB_NODELIST} | head -1) \
  --master_port=29500 \
  --rdzv_backend=c10d \
  ${TRAINING_SCRIPT} $@ \
  --checkpoint-dir /mnt/efs/checkpoints/${RUN_ID}/

# Post-run sync
aws s3 sync /mnt/efs/checkpoints/${RUN_ID}/ \
  s3://${S3_CHECKPOINT_BUCKET}/${RUN_ID}/ --only-show-errors
echo "Job ${RUN_ID} complete."
SBATCH_SCRIPT
```

---

## Cost Controls Checklist

- [ ] **Spot interruption via Slurm requeue** — ParallelCluster configures Slurm to requeue spot-interrupted jobs automatically (`job_submission_scheme: BEST_EFFORT`). Training scripts must write checkpoints every N steps so requeued jobs resume from the last checkpoint.
- [ ] **Budget alert** — AWS Cost Anomaly Detection monitor + CloudWatch billing alarm at `var.budget_alert_usd` (default $200/month). Alert via SNS → email.
- [ ] **ParallelCluster scale-in** — Set `scaledown_idletime = 10` minutes in the cluster config so idle compute nodes are terminated promptly.
- [ ] **Auto-stop head node** — CloudWatch alarm on head node CPU utilisation < 5% for `var.auto_stop_idle_minutes` → EC2 stop action. Head node is on-demand so idle cost is minimised.
- [ ] **S3 lifecycle rule** — Expire checkpoint objects older than 30 days to Glacier Instant Retrieval; keep latest 5 checkpoint versions indefinitely.
- [ ] **Tagging strategy** — All resources tagged: `Project`, `Environment`, `Owner`, `Goal`. Enables per-goal cost breakdown in AWS Cost Explorer.
- [ ] **EFS cleanup** — `teardown.sh` deletes the EFS mount target before `terraform destroy` to avoid orphaned charges.

---

## Remote State Setup

```bash
# Step 1 — Create S3 state bucket (one-time manual step)
aws s3api create-bucket \
  --bucket pytorch-genomic-terraform-state \
  --region us-east-1
aws s3api put-bucket-versioning \
  --bucket pytorch-genomic-terraform-state \
  --versioning-configuration Status=Enabled
aws s3api put-public-access-block \
  --bucket pytorch-genomic-terraform-state \
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Step 2 — Create DynamoDB lock table (one-time manual step)
aws dynamodb create-table \
  --table-name pytorch-genomic-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

`infra/terraform/backend.tf`:
```hcl
terraform {
  backend "s3" {
    bucket         = "pytorch-genomic-terraform-state"
    key            = "pytorch-genomic/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "pytorch-genomic-terraform-locks"
    encrypt        = true
  }
}
```

---

## Tooling & Stack

| Tool / Library | Version | Role |
|----------------|---------|------|
| Terraform | ≥ 1.5 | Infrastructure-as-code; wraps Service Catalog products via data sources |
| AWS provider | ~5.0 | Terraform AWS resource management |
| AWS ParallelCluster CLI (`pcluster`) | ≥ 3.7 | Cluster create/update/delete; config validation |
| EC2 ImageBuilder | (AWS-managed) | GPU AMI build, test, and publish pipeline |
| `tflint` | latest | Terraform linting and best-practice checking |
| `terraform fmt` | (built-in) | HCL formatting; run as pre-commit hook |
| `checkov` | latest | Static security analysis of Terraform configs |
| AWS CLI v2 | latest | S3 sync, SSM session, cluster operations in scripts |
| `pre-commit` | latest | `terraform fmt --check` + `tflint` on every commit to `infra/` |

Install:
```bash
pip install aws-parallelcluster checkov pre-commit
brew install terraform tflint  # or equivalent
```

---

## Open Questions — Platform & Expertise

| Question | Action needed |
|----------|--------------|
| Which Service Catalog products are available in the target Bayer AWS account? | Confirm with Bayer platform team; map product names to SSM parameter paths for VPC/subnet/endpoint IDs |
| What IAM permission boundary is enforced? | Review boundary policy before writing any IAM roles in Terraform — ensure S3, EFS, CloudWatch, SSM, ImageBuilder, ParallelCluster actions are within scope |
| Is Direct Connect or VPN the Bayer network attachment mechanism? | Confirm with network team; ensure cluster head node and compute nodes can reach Bayer internal artifact repositories (pip mirror, conda channel) via private routing |
| Does the target account allow public ECR pulls from private subnets? | Verify ECR interface endpoint is provisioned by Service Catalog; if not, request it or mirror needed images to private ECR |
| GPU driver version compatibility with selected CUDA version | Coordinate with Bayer platform team on the base AMI; ensure NVIDIA driver ≥ 530 is compatible with CUDA 12.1 on the chosen kernel version |

---

## Agent Instructions — `infra-provisioner`

Execute these steps in order. Read the entire document before starting. Engage with the Bayer platform team for items flagged in the Open Questions section before writing any networking or IAM Terraform resources.

### Step 1 — Set up branch and tooling

```bash
git checkout -b feat/goal-4-terraform-infra
pip install aws-parallelcluster checkov pre-commit
brew install terraform tflint  # or package manager equivalent
terraform --version   # must be >= 1.5
pcluster version      # must be >= 3.7
```

### Step 2 — Confirm Service Catalog SSM parameter paths

Before writing any Terraform data sources, verify the SSM parameter paths that the Service Catalog products write:

```bash
aws ssm get-parameters-by-path --path "/bayer/platform/networking/" --recursive
```

Update `variables.tf` with the confirmed paths if they differ from the defaults.

### Step 3 — Create directory structure and stub files

Create all directories from the full module tree above. Add `infra/terraform/.gitignore`:
```
*.tfstate
*.tfstate.backup
.terraform/
.terraform.lock.hcl
terraform.tfvars
```

### Step 4 — Write `versions.tf` and `backend.tf`

Use the content from the Remote State Setup section above.

### Step 5 — Write `data_sources.tf`

Read VPC ID, private subnet IDs, and S3 endpoint ID from SSM as shown in the Service Catalog strategy section above.

### Step 6 — Write the `storage` module

S3 bucket with versioning, lifecycle rules (Glacier after 30 days), public access block, and project tags. Use subnet and VPC IDs from data sources (not variables) for any bucket policy conditions.

### Step 7 — Write the `efs_mount` module

EFS filesystem and mount targets in each private subnet. Use subnet IDs from the SSM data source. Apply `amazon_efs_utils`-compatible mount options in the `on_node_start.sh` custom action.

### Step 8 — Write the `imagebuilder` module

Implement all five components as `aws_imagebuilder_component` resources (reading the YAML content from `modules/imagebuilder/components/`). Wire them into an `aws_imagebuilder_image_recipe`. Create the pipeline, distribution config, and SSM parameter output as shown in the EC2 ImageBuilder section above.

### Step 9 — Write the `monitoring` module

- SNS topic for alerts
- CloudWatch billing alarm at `var.budget_alert_usd`
- CloudWatch alarm for head node CPU idle auto-stop
- Cost Anomaly Detection monitor for the project tag

### Step 10 — Write root `main.tf` and `outputs.tf`

Wire all modules together. Outputs: S3 bucket ARN, EFS filesystem ID, ImageBuilder pipeline ARN, ParallelCluster config S3 path.

### Step 11 — Write ParallelCluster configuration files

Create `infra/pcluster/cluster_config.yaml` using the template in the ParallelCluster section above. Parameterise subnet IDs, EFS filesystem ID, and custom AMI ID using `pcluster` config variable substitution (SSM resolve syntax).

Create the `on_node_start.sh` and `on_node_configured.sh` custom actions.

### Step 12 — Write job submission scripts

Write `infra/scripts/submit_training_job.sh` and `infra/scripts/submit_sweep_job.sh` per the DDP/Slurm spec above. Write `infra/scripts/teardown.sh`.

### Step 13 — Lint and security check

```bash
cd infra/terraform
terraform fmt -recursive
tflint --recursive
checkov -d . --framework terraform
pcluster configure --config infra/pcluster/cluster_config.yaml  # dry-run validation
```

Address any `checkov` HIGH or CRITICAL findings before committing.

### Step 14 — Write `README_infra.md`

Document: prerequisites (Terraform, pcluster CLI, AWS CLI, permission boundary confirmation), how to set up remote state, how to apply dev env, how to create the ParallelCluster cluster, how to submit a training job, how to destroy cleanly.

### Step 15 — Commit and open PR

```bash
git add infra/
git commit -m "feat(goal-4): Terraform + Service Catalog infra, ParallelCluster config, ImageBuilder GPU AMI pipeline"
git push origin feat/goal-4-terraform-infra
```

Open a pull request targeting `main` with the title: `[Goal 4] Terraform + Service Catalog infrastructure, ParallelCluster, ImageBuilder`.
