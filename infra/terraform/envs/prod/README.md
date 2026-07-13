# Prod environment

Deploy production storage, EFS, ImageBuilder, and monitoring. Use the prod ParallelCluster config for multi-queue spot training.

## Prerequisites

Same as dev — see `infra/terraform/envs/dev/README.md` and `infra/README_infra.md`.

## Apply

```bash
cd infra/terraform/envs/prod
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

## ParallelCluster

```bash
pcluster create-cluster --cluster-name pytorch-genomic-prod \
  --cluster-configuration ../../pcluster/cluster_config_prod.yaml
```
