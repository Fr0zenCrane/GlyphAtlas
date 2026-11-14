# Inference

Requests use image-local integer pixel coordinates on a 512 × 512 canvas. Paths are resolved relative to the request file, independently of the shell's working directory.

```json
{
  "id": "bookshop",
  "image": "input.png",
  "prompt": "A dark blue bookstore sign with warm cream lettering.",
  "regions": [
    {
      "text": "NEW STORIES",
      "polygon": [[65,181],[446,181],[446,251],[65,251]],
      "language": "English"
    }
  ],
  "seed": 42
}
```

Use a JSON list or one object per line in a `.jsonl` file for multiple images. Each ID must be unique and contain only ASCII letters, digits, underscores and hyphens. A separate process/output directory may be used for each GPU with `CUDA_VISIBLE_DEVICES`; the public interface does not start distributed workers.

## Input behavior

- Provide 1–8 convex quadrilaterals, with the four corners in clockwise or counterclockwise perimeter order. Coordinates must be within 0–511. Crossing, degenerate, tiny and fractional polygons are rejected.
- Input images must already be 512 × 512 with normalized EXIF orientation. Resizing an image requires scaling its polygons consistently. The interface never silently stretches an image.
- The native renderer fits the global glyph to each region and constructs the edit mask. It also appends the target texts to the scene description using the inherited caption template; the actual prompt is saved in `record.json`.
- Inference accepts up to 256 code points per region, with no silent 20-character truncation. Long strings may be difficult to fit and consume more Atlas tokens.
- `*` is the native caption placeholder and cannot occur literally in the prompt or target text. Newlines/control characters in target text are rejected; represent separate lines as separate regions.
- The `language` field is metadata. It does not select a translation service or a shaping adapter.
- Han characters and Latin/digit words follow the dev `zh_char_en_word_v2` policy. Other scripts use character clusters. Do not infer connected Arabic, arbitrary grapheme support, or native-reader verification from font coverage alone.

## Assets and checkpoints

The released [GlyphAtlas weights](https://huggingface.co/Fr0zencr4nE/GlyphAtlas) support atlas heights of 48, 64 and 80 pixels. From the code repository, download the model with:

```bash
python - <<'PY_DOWNLOAD'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Fr0zencr4nE/GlyphAtlas",
    local_dir="checkpoints/GlyphAtlas",
)
PY_DOWNLOAD
```

Use `--checkpoint checkpoints/GlyphAtlas/transformer` with `runtime/infer.py`; the default sampling settings are `--steps 10 --guidance 1 --atlas-mode fixed64`.

`--inference-assets` downloads frozen base components and upstream fonts. It omits the base transformer because the GlyphAtlas transformer is supplied separately, and omits training-only OCR/ODM weights. Fonts retain their own terms. Source files are first installed with `bootstrap_upstream.py`.

Inference requires a full `region-v2` export with `config.json`, `weights_manifest.json`, a safetensors index, and all referenced shards. It verifies shard hashes and strict parameter loading, including all 593,536 Atlas parameters. It records the config and index hashes separately because the inherited export manifest covers weight shards only. A LoRA file or a training directory without its `transformer/` export is not interchangeable.

Generation uses BF16 and the native sampler; `--steps` defaults to 10 and `--guidance` to 1. Guidance values must be exactly representable in BF16. No quantization, CPU offload, or small-VRAM memory guarantee is provided by this interface. GPU smoke-test hardware and results are listed in [VALIDATION.md](VALIDATION.md).

## Outputs

```text
output/
  settings.json
  checkpoint.json           Present after successful model loading
  complete.json             Written only after every request completes
  bookshop/
    result.png              Raw generated image; absent for --dry-run
    glyph.png               White glyphs on black
    mask.png                White edit regions
    record.json
```

An interrupted batch has no `complete.json`. Use a new output directory and a manifest containing the remaining requests to continue. Identical seeds are reproducible within the same pinned hardware/software setup; bitwise equivalence across GPU types is not promised.

The inherited `runtime/generate.py` remains available for the eight-GPU AnyText benchmark protocol. It writes JPEGs under that protocol and requires the official benchmark assets. It is distinct from the public PNG editing interface.
