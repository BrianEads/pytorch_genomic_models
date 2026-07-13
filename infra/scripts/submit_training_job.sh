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
# DFW dev default: cs-cp-bifx-dfw-pytorch-genomic-data
S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:?S3_CHECKPOINT_BUCKET must be set}"

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

aws s3 sync /mnt/efs/checkpoints/${RUN_ID}/ \
  s3://${S3_CHECKPOINT_BUCKET}/${RUN_ID}/ --only-show-errors
echo "Job ${RUN_ID} complete."
SBATCH_SCRIPT
