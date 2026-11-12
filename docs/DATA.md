# Preparing the published data for GlyphAtlas

Pinned dataset: [Fr0zencr4nE/GlyphAtlas-Data-1.5M](https://huggingface.co/datasets/Fr0zencr4nE/GlyphAtlas-Data-1.5M/tree/f843011b7184e7cab058446ad60e3ab905e3a3e1), commit `f843011b7184e7cab058446ad60e3ab905e3a3e1`. It contains 1,535,582 parents and 1,577,375 images/crop views. All 16 source directories contain their own TARs; every TAR belongs to exactly one primary source.

## The data path into training

1. `tools/prepare_data.py` reads the pinned `source_catalog.json`, downloads the selected source TARs, and verifies their SHA-256 hashes.
2. It verifies image hashes and source assignments, preserving original texts, polygons, captions, aliases and parent keys. It only changes portable image paths while restoring the files.
3. It writes one SQLite row per parent, plus `ready.json` containing counts, catalog identity, repository and commit. Every view of a parent stays attached to that row.
4. `runtime.dataset.Dataset` reads those rows. `full_image` records rotate annotation variants; `crop_views` records rotate the corresponding image and its annotation together, using the stable parent key and epoch.
5. The inherited native renderer constructs the image tensor, global glyph image, mask/hint, region positions and text list. The existing Atlas encoder consumes the same text/position pairs. The trainer accepts the prepared directory directly.

No image-to-text conversion, additional OCR filtering, extra cropping, manual mask preparation or per-view oversampling is introduced by this integration. Native training still uses the dev 512px, eight-region contract; the storage retains every original annotation variant even when the native renderer selects a bounded set of targets for one presentation.

## Complete dataset

```bash
python tools/prepare_data.py --sources all --dry-run
python tools/prepare_data.py --sources all --cache data/hf_cache --output data/glyphatlas
python tools/verify_dataset.py --dataset data/glyphatlas --all-images
python tools/doctor.py --dataset data/glyphatlas --require-all-data
python runtime/train.py --config configs/train_glyphatlas_1p5m.json --dataset data/glyphatlas --output runs/glyphatlas --epochs 1 --dry-run
```

Install upstream source/fonts and training assets first as described in the README before native-renderer/model checks. For training, use the README's `tools/launch.sh` commands. The preserved recipe uses full-transformer FSDP on at least eight GPUs, global batch 128 and native timestep-weighted OCR feature supervision. A one-epoch example is 11,996 updates; it is not a claim of a new completed training run or paper-score reproduction.

The full payload is 393,484,820,480 bytes. The download cache, restored image files and SQLite metadata coexist; allow space for both the TARs and restored data. Checkpoints need additional capacity. The script never starts training during download or restoration.

## Source subsets and offline copies

```bash
python tools/prepare_data.py --sources art textocr --output data/art-textocr
python runtime/train.py --config configs/c4_bs128.json --dataset data/art-textocr --output runs/subset --steps 12500 --dry-run

# Use an already downloaded copy of the same release without downloading TARs again.
python tools/prepare_data.py --sources all --local-snapshot /path/to/GlyphAtlas-DATA-ONLY --output data/glyphatlas-offline
```

Selectors are printed by `--list-sources`. Examples include `art`, `laion`, `wukong`, `textocr`, `lsvt`, `rects` and `ctw1500`. Selecting LSVT includes both of its preserved record forms from the same `LSVT/` directory. A source subset is a different dataset; use a fresh output directory. The full-data config intentionally rejects it.

For a marked I/O smoke set, add `--max-shards 1`. Training-plan checks then require `--allow-smoke-data`, at least 128 parents, and a horizon longer than the default 500-update warmup. Tiny individual sources should be combined rather than silently oversampled.

## Restored directory

```text
data/glyphatlas/
  recipe.json
  ready.json
  dataset.sqlite
  progress.json
  images/
    art/full_image/...
    lsvt/full_image/...
    lsvt/crop_views/...
    textocr/crop_views/...
```

Re-running the exact command resumes completed shard imports. A different source selection, revision, catalog or image layout requires a new output directory. `ready.json` is written only after all selected parents have been restored. A completed database is hash-checked before reuse. Preserve the fixed total LR horizon and complete optimizer checkpoint when resuming actual training; see [TRAINING.md](TRAINING.md).

`configs/data_sources.json` is the exact portable catalog downloaded from the published commit. `configs/glyph_release.json` binds its hash to the repository and immutable revision. The code refuses a changed default catalog. `--repo-id` and `--revision` support an explicit mirror or other publication; such an override is deliberately not accepted by the pinned full-data recipe unless the release config is updated consistently.

## Source identity and scope

Each parent has one primary source. A full-image parent shared across origins uses the lexicographically first normalized source ID; crop-view parents retain their published primary source. All aliases and annotation/crop variants remain attached, and `source_tags` records overlapping membership. Selecting several sources never duplicates a parent.

The dataset viewer and `datasets.load_dataset` expose the Parquet parent indexes, not image tensors ready for the model. Use the restoration path above to obtain actual training inputs. The separate benchmark-fitting pack is absent from this data release. The reorganization adds no new test-overlap removal and should not be treated as proof of disjointness from every benchmark.

## Your own data

For custom JSON/JSONL, use `tools/prepare_dataset.py --input train.jsonl --output data/custom`. Target text must describe the text actually in the training image. That strict custom schema supports final 512px images, 1–8 convex quadrilaterals and up to 20 code points per region. The published corpus uses its preserved richer parent schema and should be restored through `prepare_data.py`.

The legacy `tools/prepare_glyph.py` entry is not used for this source-exclusive publication. It reports the replacement command instead of looking for removed packaging-group paths.
