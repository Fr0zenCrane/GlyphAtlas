#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.10 -m venv .venv
.venv/bin/python -m pip install 'pip<26'
.venv/bin/python -m pip install --no-deps torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# CUDA wheels are large; use NVIDIA's official package index for their exact pins.
mapfile -t ATLAS_CUDA_PACKAGES < <(.venv/bin/python -c "from pathlib import Path; print('\n'.join(x for x in Path('requirements.lock.txt').read_text().splitlines() if x.startswith('nvidia-')))")
.venv/bin/python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url "${ATLAS_CUDA_INDEX_URL:-https://pypi.nvidia.com}" "${ATLAS_CUDA_PACKAGES[@]}"
# All transitive dependencies, including CUDA libraries, are pinned below.
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
