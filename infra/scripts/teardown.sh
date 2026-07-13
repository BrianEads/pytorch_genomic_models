#!/bin/bash
# teardown.sh — Sync checkpoints to S3, delete ParallelCluster cluster, optionally destroy Terraform.
# Usage: ./teardown.sh <cluster-name> [--terraform-env dev|prod]
#
# WARNING: Destructive operations require explicit confirmation.

set -euo pipefail

CLUSTER_NAME="${1:?ERROR: pass ParallelCluster cluster name}"
TERRAFORM_ENV="${2:-}"

S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:?S3_CHECKPOINT_BUCKET must be set}"

echo "Syncing checkpoints from EFS to S3..."
aws s3 sync /mnt/efs/checkpoints/ "s3://${S3_CHECKPOINT_BUCKET}/" --only-show-errors || true

echo "Deleting ParallelCluster cluster: ${CLUSTER_NAME}"
read -r -p "Confirm cluster deletion [yes/N]: " confirm
if [[ "${confirm}" != "yes" ]]; then
  echo "Aborted cluster deletion."
  exit 1
fi

pcluster delete-cluster --cluster-name "${CLUSTER_NAME}" --yes

if [[ -n "${TERRAFORM_ENV}" ]]; then
  read -r -p "Run terraform destroy for env '${TERRAFORM_ENV}'? [yes/N]: " tf_confirm
  if [[ "${tf_confirm}" == "yes" ]]; then
    cd "$(dirname "$0")/../terraform/envs/${TERRAFORM_ENV}"
    terraform destroy
  fi
fi

echo "Teardown complete."
