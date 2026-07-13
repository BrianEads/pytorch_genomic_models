#!/bin/bash
# submit_sweep_job.sh — Submit an Optuna hyperparameter sweep as a Slurm job array.
# Usage: ./submit_sweep_job.sh <worker_script.py> <num_trials>
#   QUEUE - Slurm partition (default: sweep)

set -euo pipefail

WORKER_SCRIPT="${1:?ERROR: pass worker script as first argument}"
NUM_TRIALS="${2:?ERROR: pass number of trials as second argument}"

QUEUE="${QUEUE:-sweep}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

sbatch \
  --partition="${QUEUE}" \
  --array="0-$((NUM_TRIALS - 1))" \
  --nodes=1 \
  --gres=gpu:1 \
  --job-name="pytorch-genomic-sweep-${RUN_ID}" \
  --output="/mnt/efs/logs/sweep-%A_%a.out" \
  --error="/mnt/efs/logs/sweep-%A_%a.err" \
  --wrap="source /opt/miniconda/etc/profile.d/conda.sh && conda activate pytorch-genomic && python ${WORKER_SCRIPT} --trial-id \$SLURM_ARRAY_TASK_ID --run-id ${RUN_ID}"
