# Dev environment — DFW AWS account

Deploy project storage, EFS, ImageBuilder, and monitoring into the **DFW dev/testing AWS account** provisioned for [data-fetch-wizard](https://github.com/bayer-int/cs-cp-bifx-data-fetch-wizard).

Raw and tokenised datasets are staged in S3 in this account — not on local disk. Goal 3 curation pipelines read raw files from `s3://<bucket>/raw/` and write tokenised outputs to `s3://<bucket>/tokenised/`.

## S3 layout (DFW account)

| Prefix | Owner | Contents |
|--------|-------|----------|
| `raw/` | data-fetch-wizard | Raw staged files per `DatasetManifest.fetch_recipe` |
| `tokenised/` | Goal 3 (this repo) | Model-ready tensors (`.h5`, `.pt`, etc.) |
| `manifests/` | Goal 3 (this repo) | `DatasetManifest` JSON files |
| `checkpoints/` | Goal 2 training | Training checkpoints (lifecycle → Glacier after 30d) |

Default bucket: `cs-cp-bifx-dfw-pytorch-genomic-data`

## Prerequisites

- Terraform >= 1.5
- AWS CLI configured with credentials for the **DFW dev/testing account**
- Service Catalog networking product provisioned in the DFW account (VPC + private subnets in SSM)
- Remote state bucket and DynamoDB lock table created in the DFW account (see `infra/README_infra.md`)

## Apply

```bash
cd infra/terraform/envs/dev
cp terraform.tfvars.example terraform.tfvars   # edit values
terraform init
terraform plan
terraform apply
```

## ParallelCluster

After Terraform apply, create the dev cluster:

```bash
pcluster create-cluster --cluster-name pytorch-genomic-dev \
  --cluster-configuration ../../pcluster/cluster_config_dev.yaml
```

On boot, the head node syncs `raw/`, `tokenised/`, and `manifests/` from the DFW S3 bucket to `/mnt/efs/data/`.
