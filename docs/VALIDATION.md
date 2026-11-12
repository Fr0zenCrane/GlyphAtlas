# Validation scope

Release-candidate checks performed on 2026-11-11:

- The model, loss, sampler and numerical training loop from dev commit `3917d5b` remain unchanged. Twenty-four inherited runtime files match byte for byte; `runtime/train.py` changes only its full-data admission check to support the repaired release and source-indexed layout. New public interfaces are separate files.
- 40 CPU tests passed, covering distributed batch/gradient semantics, learning-rate endpoints, checkpoint completeness, resolution schedule state, explicit pipeline precision, input rejection, data import, checkpoint validation and native renderer parity.
- The new request adapter matches the native inference renderer for image, glyph, hint, positions, line glyphs, attention mask, texts and rendered prompt at three seeds. It preserves caller Python/NumPy RNG state.
- The synthetic dataset imports and passes database/image-hash checks and the native training-loader doctor. A separate 128-image synthetic fixture passes training plan validation at BS128, eight ranks, accumulation 16 and a 12,500-update horizon, without launching training.
- Real single-GPU inference completed on one NVIDIA A800-SXM4-80GB with an existing full GlyphAtlas transformer export. One request and a two-request batch completed at 512 × 512, Atlas64, ten steps and guidance 1. The repeated first request produced identical PNG bytes. Both batch requests passed actual backbone-guidance and denoising-call-count checks.
- The inference run checked all checkpoint shard hashes, strict model loading, and exact Atlas tensor reload. The GPU was released afterward.

The smoke checkpoint is identified by weight-manifest SHA-256 `58f7ef7d5fcdbcaecd157f74d1f7ba7d50b84512905eaffdb965f0f8986851e0`. It is an existing development checkpoint used for interface compatibility checks, not bundled or designated as the official release model. The synthetic sign is a functional fixture, not benchmark or paper evidence.

The candidate has **not** been validated by a new full distributed training run, a fresh installation of every dependency on a clean machine, or a full benchmark evaluation. No GPU-memory minimum, throughput comparison, paper-score reproduction or comprehensive multilingual correctness claim is made. The smoke tests do not establish Arabic shaping support.

## Source-organized dataset integration

The source catalog is built by reading all 1,448 canonical TARs after checking their complete SHA-256 hashes (393,484,810,240 bytes). All 1,535,582 parent keys are unique across the 16 exclusive source indexes. Source counts, aliases and augmentation counts are reconciled against the original release.

The integration was additionally checked by restoring the full ArT and Bharat Scene Text selections (1,093 parents / 1,132 images), verifying original image bytes and native sample loading, and downloading/restoring a 939-parent ArT shard from the actual public Hugging Face revision. Both restored layouts are accepted by the unchanged native dataset loader; the downloaded sample also passes training-plan validation. These checks do not claim that all 1.5M images were newly decoded or visually reviewed.

## Published source-exclusive data integration

Validated against data commit `f843011b7184e7cab058446ad60e3ab905e3a3e1`. The separate code tree binds the exact catalog hash and dataset revision. The default full-data plan covers 16 sources, 1,456 TARs and 1,535,582 parents. A fresh anonymous download/restoration of COCO-Text and CTW1500 produced 1,450 parents / 1,464 images, passed full image-hash checks and native loader rendering, and passed the generic training recipe's dry-run. The full-data recipe correctly rejects this subset. Forty CPU regression tests passed. No new optimizer update, GPU training run, or full 1.5M-image restoration was performed by this check. No source code was published to the dataset repository.

## Training validation before GitHub publication (2026-11-12)

The release was additionally exercised on eight A800 80GB GPUs using the real
COCO-Text and CTW1500 subset restored from the pinned data commit: 1,450 parents
and 1,464 images. These are bounded functional checks, not a convergence or
paper-score reproduction run. The machine used the existing pinned environment.

- All 40 CPU tests passed both in the source directory and in the separate GitHub
  checkout. That checkout independently downloaded and hash-checked 102 upstream
  source files and the OCR/font assets. Dependency consistency (`pip check`),
  source syntax, the full-data download plan and a real-data training dry-run passed.
- The unchanged full-model trainer completed one AdamW update at global batch 128
  (eight ranks, 16 accumulation steps per rank), fixed Atlas64, with a 501-update
  LR horizon and 500-update warmup. Its first batch contained 124 full-image and
  four crop-view parents. Flow, mask and timestep-weighted OCR losses and gradients
  were finite; ODM and CTC remained zero.
- The actual full checkpoint was saved and reloaded, including FP32 master
  parameters, FP32 AdamW moments, RNG/sampler state and Atlas caches. A BF16
  inference export was also written. The resumed trainer completed the next global batch's backward
  pass without another optimizer update.
- The random-Atlas configuration completed a fresh full global-batch backward
  pass with zero optimizer updates.
- Small-model eight-rank tests checked exact parameter/AdamW/RNG/sampler restore,
  FP32 and BF16 accumulation, clipping and replica-gradient synchronization.
  Two logical four-GPU nodes matched the world-sharded reference, including the
  historical resolution-schedule boundary. This does not test a physical network
  between separate hosts.

The model, loss and numerical training loop were not changed for these checks.
An outdated legacy-config path in the distributed validation script was corrected.
The machine-readable scope and runtime hashes are in
[`validation_20261112.json`](../provenance/validation_20261112.json).

## Run the CPU checks

After installing the pinned environment, upstream source and font assets:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tools/download_assets.py --verify-source
python tools/doctor.py --dataset data/custom
```

Use the README's `--validate-only` launch before a budgeted production training run on a new cluster. It performs real backward computation with zero optimizer updates.

## Build a clean source archive

```bash
python tools/package_release.py --output dist/GlyphAtlas-source.zip
```

The packager excludes installed upstream source, model weights, datasets, outputs, local environment files, symlinks and Python bytecode. It scans included text for private workspace paths and credential patterns, writes a per-file SHA-256 manifest into the archive, and verifies ZIP CRCs and every member hash. The manifest describes archive contents; it does not itself constitute a license grant.
