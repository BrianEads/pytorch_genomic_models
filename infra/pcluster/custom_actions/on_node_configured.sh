#!/bin/bash
# on_node_configured.sh — Post-config: verify CUDA and prepare DFW-aligned runtime directories.

set -euo pipefail

source /opt/miniconda/etc/profile.d/conda.sh
conda activate pytorch-genomic

# Verify CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Shared runtime directories on EFS (mirrors DFW S3 prefix layout)
mkdir -p \
  /mnt/efs/data/raw \
  /mnt/efs/data/tokenised \
  /mnt/efs/data/manifests \
  /mnt/efs/logs \
  /mnt/efs/checkpoints

echo "Node $(hostname) configured at $(date)" | tee -a /var/log/node_ready.log
