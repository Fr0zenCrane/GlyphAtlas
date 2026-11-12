"""Validate full-release metadata for both supported restoration layouts."""


def require_full_release(ready,release):
    modern=release.get('schema')=='glyphatlas-training-release-v2'
    if modern:
        expected={'anytext':release['record_types']['full_image']['parents'],
                  'augmentation':release['record_types']['crop_views']['parents']}
    else:
        expected={'anytext':release['components']['anytext_unique']['images'],
                  'augmentation':release['components']['augmentation']['parents']}
    if not ready.get('complete') or ready.get('smoke_only'):
        raise ValueError('A complete non-smoke dataset is required')
    if ready.get('images')!=sum(expected.values()):
        raise ValueError('Full-release parent count mismatch')
    if ready.get('source_release',{}).get('revision')!=release['revision']:
        raise ValueError('Dataset revision differs from the pinned release')
    if modern:
        identity=ready.get('source_release',{})
        if identity.get('repo_id')!=release['repo_id'] or identity.get('catalog_sha256')!=release['catalog']['sha256']:
            raise ValueError('Dataset repository/catalog identity differs from the pinned release')
        sources={name:item['parents'] for name,item in release['sources'].items()}
        if ready.get('parents_by_source')!=sources or set(identity.get('sources',[]))!=set(sources):
            raise ValueError('Full release requires every published source with its exact parent count')
        if ready.get('format')!='glyph-source-indexed-v1':
            raise ValueError('The source-exclusive publication requires its source-indexed restoration')
    if ready.get('format')=='glyph-source-indexed-v1':
        if ready.get('parents_by_component')!=expected:
            raise ValueError('Full-release component counts mismatch')
    elif ready.get('format')=='glyph-combined-parents-v1':
        counts={c['path']:c['images'] for c in ready['components']}
        if counts!={'anytext_unique':expected['anytext'],'augmentation':expected['augmentation']}:
            raise ValueError('Full-release component counts mismatch')
    else:
        raise ValueError('Expected a complete combined or source-indexed release')
