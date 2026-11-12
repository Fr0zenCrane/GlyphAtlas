#!/usr/bin/env bash
# Example only. Set partition/account/resources for your own cluster.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
set -euo pipefail
: "${ATLAS_REPO:?Shared checkout path}"
: "${ATLAS_DATASET:?Prepared shared dataset path}"
: "${ATLAS_OUTPUT:?New shared output path}"
: "${ATLAS_STEPS:?Explicit total optimizer-update budget}"
mapfile -t ATLAS_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${ATLAS_HOSTS[0]}" MASTER_PORT="${MASTER_PORT:-29500}"
export ATLAS_NNODES="$SLURM_NNODES" ATLAS_NPROC=8
export ATLAS_CONFIG="${ATLAS_CONFIG:-configs/c4_node_shard_bs128.json}"
# Default is zero optimizer updates. Set 0 only for an explicitly budgeted run.
export ATLAS_VALIDATE_ONLY="${ATLAS_VALIDATE_ONLY:-1}"
srun bash -c 'export ATLAS_NODE_RANK="$SLURM_PROCID"; cd "$ATLAS_REPO"; args=(); if [[ "$ATLAS_VALIDATE_ONLY" == 1 ]]; then args+=(--validate-only); fi; exec bash tools/launch.sh --config "$ATLAS_CONFIG" --dataset "$ATLAS_DATASET" --output "$ATLAS_OUTPUT" --steps "$ATLAS_STEPS" "${args[@]}"'
