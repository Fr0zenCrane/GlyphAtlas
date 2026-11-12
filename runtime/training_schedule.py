"""Authorized two-epoch schedule; optimizer and sampler are never reset."""


def stage_for_step(step, cfg):
    assert 1 <= step <= cfg['steps']
    if cfg.get('training_schedule') == 'fixed64_cosine_then_random_constant_v1':
        boundary = cfg['updates_per_epoch']
        assert cfg['steps'] == 2 * boundary and cfg['warmup_steps'] == 0
        return (1, 'fixed64') if step <= boundary else (2, 'random')
    return 1, cfg['atlas_training_mode']


def switch_encoder(model, mode, seed=42):
    """Carry the fixed64 cache across the boundary without touching native RNG."""
    from glyph_atlas.data import AtlasEncoder
    from glyph_atlas.resolution_data import ResolutionAtlasEncoder
    previous = getattr(model, 'atlas_encoder', None)
    if previous is not None and getattr(previous, 'mode', 'fixed64') == mode:
        return False
    assert mode in ('fixed64', 'random')
    encoder = (ResolutionAtlasEncoder(model.flux_pipe, mode='random', seed=seed)
               if mode == 'random' else AtlasEncoder(model.flux_pipe))
    if previous is not None:
        encoder.cache = previous.cache
    model.atlas_encoder = encoder
    return True


def checkpoint_due(step, limit, cfg):
    return (step % cfg['save_every'] == 0 or step == limit or
            (cfg.get('training_schedule') and step % cfg['updates_per_epoch'] == 0))


def verify_complete_checkpoint(folder):
    """Fail closed before advertising a resumable epoch endpoint.

    Checks presence/size and completion metadata; not a full tensor reload.
    """
    import json
    from pathlib import Path
    folder = Path(folder)
    info = json.loads((folder / 'training_state.json').read_text())
    done = json.loads((folder / 'resume_aux_complete.json').read_text())
    assert done['complete'] and done['step'] == info['step']
    assert done['ranks'] == info['world_size']
    required = ['distributed/.metadata', 'transformer/config.json',
                'transformer/weights_manifest.json',
                'transformer/diffusion_pytorch_model.safetensors.index.json']
    for rank in range(info['world_size']):
        required += [f'rng-rank{rank}.pt', f'aux-rank{rank}.pt']
    if info['checkpoint_layout']['mode'] == 'node_shard':
        required += [f'optimizer-shard{rank}.pt'
                     for rank in range(info['checkpoint_layout']['shard_size'])]
    weights = json.loads((folder / 'transformer/weights_manifest.json').read_text())
    assert weights['files'], 'Empty BF16 export'
    shards = list((folder / 'distributed').glob('*.distcp'))
    assert shards and all(p.stat().st_size > 0 for p in shards), 'Missing DCP tensors'
    from torch.distributed.checkpoint import FileSystemReader
    metadata = FileSystemReader(folder / 'distributed').read_metadata()
    assert metadata.storage_data, 'Empty DCP storage metadata'
    for storage in metadata.storage_data.values():
        shard = folder / 'distributed' / storage.relative_path
        assert shard.is_file() and shard.stat().st_size >= storage.offset + storage.length, str(shard)
    for entry in weights['files']:
        file = folder / 'transformer' / entry['file']
        assert file.is_file() and file.stat().st_size == entry['bytes'], str(file)
    for relative in required:
        file = folder / relative
        assert file.is_file() and file.stat().st_size > 0, str(file)
    return dict(step=info['step'], path=str(folder), full_resume=True,
                bf16_export=True, retain=True)
