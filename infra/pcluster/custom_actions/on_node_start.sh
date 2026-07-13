#!/bin/bash
# on_node_start.sh — ParallelCluster custom action: runs on every compute node at boot.
# Software stack is pre-installed in the custom AMI by ImageBuilder.
#
# Data staging: raw and tokenised datasets live in the DFW dev AWS account S3 bucket
# (provisioned for data-fetch-wizard). Goal 3 curation pipelines write tokenised outputs
# and manifests back to the same bucket; this script syncs them to EFS for training.
#
# Env vars (set in ParallelCluster config):
#   EFS_DNS              — EFS filesystem DNS name
#   S3_DATASET_BUCKET    — DFW account data bucket (default: cs-cp-bifx-dfw-pytorch-genomic-data)
#   S3_RAW_PREFIX        — DFW raw staging prefix (default: raw/)
#   S3_TOKENISED_PREFIX  — Goal 3 tokenised output prefix (default: tokenised/)
#   S3_MANIFESTS_PREFIX  — DatasetManifest JSON prefix (default: manifests/)

set -euo pipefail

EFS_DNS="${EFS_DNS:?EFS_DNS env var must be set in ParallelCluster config}"
S3_DATASET_BUCKET="${S3_DATASET_BUCKET:?S3_DATASET_BUCKET must be set}"
S3_RAW_PREFIX="${S3_RAW_PREFIX:-raw/}"
S3_TOKENISED_PREFIX="${S3_TOKENISED_PREFIX:-tokenised/}"
S3_MANIFESTS_PREFIX="${S3_MANIFESTS_PREFIX:-manifests/}"
PROJECT_NAME="${PROJECT_NAME:-pytorch-genomic}"

# 1. Mount EFS (amazon-efs-utils is pre-installed in AMI)
if ! mountpoint -q /mnt/efs; then
  mkdir -p /mnt/efs
  mount -t efs -o tls,iam "${EFS_DNS}:/" /mnt/efs
  if ! grep -q "^${EFS_DNS}:/ /mnt/efs " /etc/fstab; then
    echo "${EFS_DNS}:/ /mnt/efs efs tls,iam,_netdev 0 0" >> /etc/fstab
  fi
fi

# 2. Sync DFW S3 datasets to EFS (head node only; compute nodes use shared EFS)
if [[ "${PCLUSTER_NODE_TYPE:-}" == "HeadNode" ]]; then
  mkdir -p /mnt/efs/data/raw /mnt/efs/data/tokenised /mnt/efs/data/manifests

  # Raw files staged by data-fetch-wizard (https://github.com/bayer-int/cs-cp-bifx-data-fetch-wizard)
  aws s3 sync "s3://${S3_DATASET_BUCKET}/${S3_RAW_PREFIX}" /mnt/efs/data/raw/ \
    --only-show-errors

  # Tokenised tensors written by Goal 3 curation pipelines (this repo)
  aws s3 sync "s3://${S3_DATASET_BUCKET}/${S3_TOKENISED_PREFIX}" /mnt/efs/data/tokenised/ \
    --only-show-errors

  # DatasetManifest JSON contract files
  aws s3 sync "s3://${S3_DATASET_BUCKET}/${S3_MANIFESTS_PREFIX}" /mnt/efs/data/manifests/ \
    --only-show-errors
fi

# 3. Activate conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate pytorch-genomic

# 4. Log readiness
echo "Node $(hostname) ready at $(date) — AMI: $(curl -s http://169.254.169.254/latest/meta-data/ami-id)" \
  | tee /var/log/node_ready.log
