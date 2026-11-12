"""Single-GPU scene-text editing with native GlyphAtlas conditioning."""
import argparse
import json
import math
import os
from pathlib import Path
import time
from common import ROOT, MODELS, VENDOR, sha, write
from requests_io import load_requests


def check_inference_assets():
    """Inference needs pinned source and fonts, not OCR training weights."""
    rows = json.loads((ROOT/'provenance/vendored_source.json').read_text())['files']
    assets = json.loads((ROOT/'provenance/vendor_assets.json').read_text())['files']
    rows = rows + [row for row in assets if row['path'].endswith('.ttf')]
    for row in rows:
        path = VENDOR/row['path']
        if not path.is_file() or sha(path) != row['sha256']:
            raise ValueError('Missing or changed inference dependency: ' + row['path'] +
                '; run tools/bootstrap_upstream.py and tools/download_assets.py --inference-assets')


def checkpoint_files(checkpoint):
    """Validate the export manifest and shard coverage before loading tensors."""
    checkpoint = Path(checkpoint).resolve()
    config = json.loads((checkpoint / 'config.json').read_text())
    if config.get('atlas_version') != 'region-v2':
        raise ValueError('Expected a full region-v2 GlyphAtlas transformer export, not a base model or LoRA')
    if config.get('atlas_structure', 'bound') != 'bound' or config.get('atlas_tile_size', 64) != 64:
        raise ValueError('This release supports the bound scene+glyph Atlas structure with tile size 64')
    manifest = checkpoint / 'weights_manifest.json'
    if not manifest.is_file():
        raise ValueError('Missing weights_manifest.json; use a complete transformer export')
    entries = json.loads(manifest.read_text())['files']
    names = set()
    for row in entries:
        rel = Path(row['file'])
        path = (checkpoint / rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(checkpoint):
            raise ValueError('Unsafe checkpoint manifest path')
        if row['file'] in names:
            raise ValueError('Duplicate checkpoint manifest entry')
        names.add(row['file'])
        if sha(path) != row['sha256']:
            raise ValueError('Checkpoint hash mismatch: ' + row['file'])
    index = checkpoint / 'diffusion_pytorch_model.safetensors.index.json'
    if not index.is_file():
        raise ValueError('Missing safetensors shard index')
    mapping = json.loads(index.read_text())['weight_map']
    if not set(mapping.values()) <= names or not any(n.startswith('atlas_') for n in mapping):
        raise ValueError('Incomplete transformer / Atlas shard coverage')
    return sha(manifest), mapping


def generate_audited(pipe, **kwargs):
    """Check actual backbone guidance and denoising count, restoring the hook."""
    import inspect
    import torch
    from src.flux.generate_fill import generate_fill
    namespace = inspect.unwrap(generate_fill).__globals__
    native = namespace['tranformer_forward']
    observed = []
    def traced(*args, **kw):
        guidance = kw.get('guidance')
        if guidance is None or not torch.all(guidance == kwargs['guidance_scale']).item():
            raise RuntimeError('Actual backbone guidance differs from the requested value')
        observed.append(float(guidance[0]))
        return native(*args, **kw)
    namespace['tranformer_forward'] = traced
    try:
        result = generate_fill(pipe, **kwargs)
    finally:
        namespace['tranformer_forward'] = native
    if len(observed) != kwargs['num_inference_steps']:
        raise RuntimeError('Actual denoising-call count differs from the requested steps')
    return result, dict(guidance=observed[0], denoising_forward_calls=len(observed), passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='One request, JSON list, or JSONL file')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing contents are never overwritten')
    parser.add_argument('--checkpoint', type=Path, help='Full GlyphAtlas transformer export')
    parser.add_argument('--base-model', type=Path, default=MODELS/'FLUX.1-Fill-dev-diffusers')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--guidance', type=float, default=1.)
    parser.add_argument('--atlas-mode', choices=['fixed48', 'fixed64', 'fixed80'], default='fixed64')
    parser.add_argument('--dry-run', action='store_true', help='Validate input and save conditioning previews without loading weights')
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.guidance) or args.guidance < 0:
        parser.error('steps must be positive and guidance must be finite and nonnegative')
    if not args.dry_run and not args.checkpoint:
        parser.error('--checkpoint is required for generation')
    rows = load_requests(args.input)
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a new --output directory')
    check_inference_assets()
    from inference_inputs import prepare_input
    import numpy as np
    from PIL import Image
    # Fail before reserving GPUs if the requested text cannot be rendered.
    first = prepare_input(rows[0])
    from font_support import font_path
    for row in rows:
        for region in row['regions']:
            font_path(region['text'])
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(steps=args.steps, guidance=args.guidance, atlas_mode=args.atlas_mode,
        resolution=[512, 512], atlas_policy='zh_char_en_word_v2', output='raw pipeline PNG',
        dry_run=args.dry_run, input_manifest_sha256=sha(args.input), requests=len(rows))
    write(output / 'settings.json', settings)
    cp_hash = None
    if not args.dry_run:
        import torch
        from diffusers import FluxFillPipeline
        from safetensors import safe_open
        from glyph_atlas.model import AtlasTransformer
        from glyph_atlas.resolution_data import ResolutionAtlasEncoder
        from glyph_atlas.data import make_routes
        from src.flux.condition import Condition
        from src.flux.generate_fill import generate_fill
        from pipeline_precision import StableTrainingPipeline
        device = torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Generation requires a CUDA device with BF16 support')
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 support is required')
        # Native guidance is represented in BF16; reject silently rounded settings.
        if torch.tensor(args.guidance, dtype=torch.bfloat16).item() != args.guidance:
            raise ValueError('guidance must be exactly representable in BF16 (e.g. 1, 3.5)')
        torch.set_num_threads(4)
        cp_hash, mapping = checkpoint_files(args.checkpoint)
        transformer, info = AtlasTransformer.from_pretrained(str(args.checkpoint),
            torch_dtype=torch.bfloat16, output_loading_info=True, local_files_only=True)
        if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise ValueError('Checkpoint does not strictly match AtlasTransformer: ' + str(info))
        atlas_parameters = [(name, value) for name, value in transformer.named_parameters() if name.startswith('atlas_')]
        if sum(value.numel() for _, value in atlas_parameters) != 593536:
            raise ValueError('Unexpected Atlas parameter count')
        for filename in {mapping[n] for n, _ in atlas_parameters}:
            with safe_open(Path(args.checkpoint)/filename, framework='pt', device='cpu') as tensors:
                for name, value in atlas_parameters:
                    if mapping[name] == filename and not torch.equal(value.cpu(), tensors.get_tensor(name).to(value.dtype)):
                        raise ValueError('Atlas tensor reload mismatch: ' + name)
        transformer.eval().requires_grad_(False)
        pipe = FluxFillPipeline.from_pretrained(str(args.base_model), transformer=transformer,
            torch_dtype=torch.bfloat16, local_files_only=True).to(device)
        # Fixed BF16 pipeline compute avoids Diffusers' unordered dtype inference.
        pipe.__class__ = StableTrainingPipeline
        pipe.set_progress_bar_config(disable=False)
        encoder = ResolutionAtlasEncoder(pipe, args.atlas_mode)
        import yaml
        model_config = yaml.safe_load((VENDOR/'train/config/word_512_size.yaml').read_text())['model']
        write(output/'checkpoint.json', dict(weights_manifest_sha256=cp_hash,
            config_sha256=sha(Path(args.checkpoint)/'config.json'),
            shard_index_sha256=sha(Path(args.checkpoint)/'diffusion_pytorch_model.safetensors.index.json'), atlas_parameters=593536,
            strict_loading=True, exact_atlas_reload=True, dtype=str(pipe.dtype)))
    completed = []
    for index, row in enumerate(rows):
        data = first if index == 0 else prepare_input(row)
        folder = output/row['id']
        folder.mkdir()
        Image.fromarray(np.uint8(np.clip(data['condition'], 0, 1)*255)).save(folder/'glyph.png')
        Image.fromarray(np.uint8(np.clip(data['hint'], 0, 1)*255)).save(folder/'mask.png')
        record = dict(id=row['id'], seed=row['seed'], regions=row['regions'],
            input_sha256=sha(row['image']), rendered_prompt=data['description'],
            checkpoint_manifest_sha256=cp_hash, **settings)
        if not args.dry_run:
            import random
            random.seed(row['seed']); np.random.seed(row['seed'] % 2**32)
            torch.manual_seed(row['seed']); torch.cuda.manual_seed_all(row['seed'])
            start = time.monotonic()
            with torch.inference_mode():
                atlas = encoder.encode(data['texts'][:data['n_lines']], data['positions'], policy='zh_char_en_word_v2')
                if atlas is None:
                    raise ValueError('No visible Atlas conditioning')
                atlas['routes'] = make_routes(atlas['regions'], atlas['owners'])
                condition = Condition(condition_type='word_fill',
                    condition=[data['condition'], data['hint'], data['image'][None]], position_delta=[0, 0])
                generated, forward_audit = generate_audited(pipe, prompt=[data['description']], conditions=[condition],
                    height=512, width=512, num_inference_steps=args.steps, guidance_scale=args.guidance,
                    generator=torch.Generator(device).manual_seed(row['seed']),
                    model_config=dict(model_config, atlas=atlas), default_lora=True)
                result = generated.images[0]
            result.save(folder/'result.png')
            record.update(output_sha256=sha(folder/'result.png'), seconds=time.monotonic()-start,
                atlas_tokens=atlas['features'].shape[1], atlas_plan=atlas.get('plan'), runtime_parameters=forward_audit)
        write(folder/'record.json', record)
        completed.append(row['id'])
        print(('PREPARED' if args.dry_run else 'GENERATED') + ' ' + row['id'], flush=True)
    write(output/'complete.json', dict(complete=True, dry_run=args.dry_run, requests=completed))


if __name__ == '__main__':
    main()
