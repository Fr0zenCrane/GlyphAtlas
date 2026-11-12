#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export GLYPH_OMNI_ROOT="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
for ATLAS_ARG in "$@"; do
  if [[ "$ATLAS_ARG" == --validate-only ]]; then
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}" NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
  fi
done
ATLAS_NPROC="${ATLAS_NPROC:-8}"
ATLAS_PYTHON="${ATLAS_PYTHON:-$PWD/.venv/bin/python}"
ATLAS_NNODES="${ATLAS_NNODES:-1}"
if [[ "$ATLAS_NNODES" == 1 ]]; then
  exec "$ATLAS_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$ATLAS_NPROC" runtime/train.py "$@"
fi
: "${ATLAS_NODE_RANK:?Set the zero-based node rank}"
: "${MASTER_ADDR:?Set the first node address reachable by all nodes}"
MASTER_PORT="${MASTER_PORT:-29500}"
exec "$ATLAS_PYTHON" -m torch.distributed.run --nnodes="$ATLAS_NNODES" \
  --nproc_per_node="$ATLAS_NPROC" --node_rank="$ATLAS_NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" runtime/train.py "$@"
