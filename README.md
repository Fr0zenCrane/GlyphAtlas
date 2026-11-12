# GlyphAtlas

Training and inference code for regional glyph-atlas conditioning in scene text editing, built on FLUX.1-Fill-dev and FLUX-Text.

This release candidate is derived from the dev implementation at `3917d5b`. It preserves the model, routed attention, OCR objective, sampler, and distributed optimizer implementation. Public interfaces add source-organized data restoration, explicit JSON inputs, single-GPU inference, input validation, and custom dataset preparation. The data admission check additionally supports the repaired public release and its source-indexed layout.

## Supported configuration

| Component | Support |
| --- | --- |
| Image canvas | 512 × 512, supplied explicitly; no automatic image or polygon resizing |
| Editable regions | 1–8 quadrilaterals per image |
| Inference | One CUDA GPU, BF16; one request or a JSON/JSONL batch |
| Atlas cell height | 48, 64, or 80 at inference; 64 or randomized height during training |
| Training | Full transformer, FSDP, at least 8 GPUs; global batch 128 |
| Checkpoints | Full GlyphAtlas transformer exports for inference; full optimizer checkpoints for exact resume |

Arbitrary-resolution teaser workflows and alternative multilingual shaping adapters are not part of this dev-based release. The default tokenization uses Han characters, Latin/digit words, and individual clusters for other scripts. Font coverage does not establish correct Arabic joining or bidirectional layout. See [inference details](docs/INFERENCE.md).

## Install

Use Linux, Python 3.10 and an NVIDIA GPU supporting BF16. The environment pins PyTorch 2.4.0/CUDA 12.1, Diffusers 0.32.2, and Pillow 9.5.0.

```bash
git clone https://github.com/Fr0zenCrane/GlyphAtlas.git
cd GlyphAtlas
bash tools/setup_env.sh
source .venv/bin/activate
python tools/bootstrap_upstream.py
python tools/download_assets.py --verify-source
```

`bootstrap_upstream.py` downloads only the pinned upstream source files and verifies every file hash. Upstream implementation is a separate dependency, excluded from this source archive. Read [licensing and attribution](docs/LICENSING.md) before distribution.

Accept the base model's access terms on [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), then authenticate locally (`huggingface-cli login` or `HF_TOKEN`). Credentials are never embedded in configs.

## Edit an image

```bash
python tools/download_assets.py --inference-assets

# Inspect the native mask and glyph rendering without loading model weights.
python runtime/infer.py --input examples/edit.json --output runs/input-preview --dry-run

# Supply a full GlyphAtlas transformer export. No weights are bundled here.
python runtime/infer.py \
  --input examples/edit.json \
  --checkpoint /path/to/checkpoint/transformer \
  --output runs/bookshop --steps 10 --guidance 1 --atlas-mode fixed64
```

Each request produces `result.png`, `glyph.png`, `mask.png`, and a JSON record with the actual prompt, texts, seed, settings, input/output hashes and Atlas token count. The result is the raw pipeline image. Existing output directories are rejected to prevent mixing runs.

There is no hard-coded public GlyphAtlas weight URL in this candidate. `--checkpoint` accepts a local complete export; official model hosting can be added when the release weights are selected. `--base-model` can point to an existing FLUX.1-Fill-dev Diffusers directory.

## Published data → training

The dataset is [GlyphAtlas-Data-1.5M](https://huggingface.co/datasets/Fr0zencr4nE/GlyphAtlas-Data-1.5M/tree/f843011b7184e7cab058446ad60e3ab905e3a3e1), pinned to `f843011b7184e7cab058446ad60e3ab905e3a3e1`: **1,535,582 training parents across 16 source directories and 1,456 source-exclusive TARs**. The source catalog and every shard hash are pinned by `configs/glyph_release.json`; the portable catalog itself is kept byte-identical to the published file.

```bash
# Inspect before downloading. Full TAR payload: about 393.5 GB.
python tools/prepare_data.py --list-sources
python tools/prepare_data.py --sources all --dry-run

# Download, hash-check, and restore to the trainer's actual dataset format.
python tools/prepare_data.py --sources all --cache data/hf_cache --output data/glyphatlas
python tools/verify_dataset.py --dataset data/glyphatlas --all-images

# Obtain the base-model/OCR/font assets; base-model access terms apply.
python tools/download_assets.py --training-assets
python tools/doctor.py --dataset data/glyphatlas --training-assets --gpu --require-all-data

# Check a one-epoch budget, then a real backward pass with ZERO updates.
python runtime/train.py --config configs/train_glyphatlas_1p5m.json --dataset data/glyphatlas --output runs/glyphatlas --epochs 1 --dry-run
bash tools/launch.sh --config configs/train_glyphatlas_1p5m.json --dataset data/glyphatlas --output runs/glyphatlas-validation --epochs 1 --validate-only

# Start training only when ready.
bash tools/launch.sh --config configs/train_glyphatlas_1p5m.json --dataset data/glyphatlas --output runs/glyphatlas --epochs 1
```

One full-data epoch has **11,996 optimizer updates at global batch 128**, dropping 94 parents at that epoch boundary. The full-data config rejects missing sources, smoke subsets, and a different repository/revision/catalog. To experiment with a source subset, use `--sources art textocr`, a new output directory and the generic `configs/c4_bs128.json` recipe with an explicit step horizon.

The importer creates `dataset.sqlite`, `ready.json`, and `images/<source>/<record_type>/...`. The native loader then selects one annotation/crop variant for each parent, renders global glyphs and masks, and feeds the regional Atlas encoder. A parent with two crop views still contributes one sample per epoch. No masks or glyph images need to be manually prepared. See [DATA.md](docs/DATA.md) for the complete mapping and offline restoration.

The dataset repository contains data only. Training/inference implementation is kept in this separate code tree.

## Train on your own data

```bash
python tools/download_assets.py --training-assets

# Prepare a custom dataset (at least 128 distinct images for training).
python tools/prepare_dataset.py --input /path/to/train.jsonl --output data/custom
python tools/verify_dataset.py --dataset data/custom --all-images
python tools/doctor.py --dataset data/custom --training-assets --gpu

# Explicit horizon, checked without starting training.
python runtime/train.py --config configs/c4_bs128.json \
  --dataset data/custom --output runs/train --steps 12500 --dry-run

# Zero optimizer updates: validate a complete accumulated backward pass.
bash tools/launch.sh --config configs/c4_bs128.json \
  --dataset data/custom --output runs/train-validation --steps 12500 --validate-only

bash tools/launch.sh --config configs/c4_bs128.json \
  --dataset data/custom --output runs/train --steps 12500
```

The 12,500-step value is an explicit example budget, not an automatically selected full-data epoch count or a claim of paper reproduction. Use `--epochs` instead when the budget should follow your actual dataset size. Default warmup is 500 updates, so the total horizon must exceed 500.

The default recipe trains from fresh official FLUX weights with the combined flow/mask objective and PP-OCRv3 feature loss weighted by `0.25 × 2t`. It does not require a separately trained second stage. `configs/c4_random_resolution_bs128.json` changes the Atlas-height sampling policy. Numerical coefficients are inherited from dev and deliberately guarded.

See [training and resume](docs/TRAINING.md), [data formats](docs/DATA.md), and [validation scope](docs/VALIDATION.md). No dataset images, model weights, private experiment outputs, credentials, or dev Git history are included.

## Repository map

```text
runtime/infer.py           Single-GPU JSON/JSONL inference
runtime/train.py           Distributed training entry point
runtime/scripts/glyph_atlas/   Atlas encoder, routing and model integration
tools/prepare_data.py      Source-aware download and training-data restoration
tools/prepare_dataset.py   Portable custom dataset importer
tools/prepare_glyph.py     Legacy importer; directs this release to prepare_data.py
tools/launch.sh            Single/multi-node training launcher
configs/                  Fixed and random-height recipes
examples/                 Synthetic input fixtures, not model results
tests/                    Numerical and interface regression tests
provenance/               Pinned upstream/assets and dev-template identity
```

Maintainer: Luozheng Qin. See [NOTICE](NOTICE) for upstream attribution.
