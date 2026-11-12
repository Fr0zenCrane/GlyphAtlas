"""Stable source names for the published GlyphAtlas training corpus."""

SOURCE_NAMES = {
    'art': 'ArT', 'bharat_scene_text': 'Bharat Scene Text', 'coco_text': 'COCO-Text',
    'ctw': 'CTW', 'ctw1500': 'CTW1500', 'hiertext': 'HierText', 'laion': 'LAION',
    'lsvt': 'LSVT', 'mlt2019': 'MLT2019', 'mtwi2018': 'MTWI2018',
    'rctw2017': 'RCTW2017', 'rects': 'ReCTS', 'stiplar': 'STIPLAR',
    'textocr': 'TextOCR', 'totaltext': 'Total-Text', 'wukong': 'WuKong',
}
ALIASES = {
    'Art': 'art', 'BharatSceneText': 'bharat_scene_text', 'COCO_Text': 'coco_text',
    'CTW': 'ctw', 'CTW1500': 'ctw1500', 'HierText': 'hiertext', 'laion': 'laion',
    'LSVT': 'lsvt', 'MLT2019': 'mlt2019', 'mlt2019': 'mlt2019',
    'MTWI2018': 'mtwi2018', 'icdar2017rctw': 'rctw2017', 'ReCTS': 'rects',
    'STIPLAR': 'stiplar', 'TextOCR': 'textocr', 'TotalText': 'totaltext',
    **{f'wukong_{i}of5': 'wukong' for i in range(1, 6)},
}


def source_identity(parent):
    """Return exclusive source, all source tags, and original source names.

    AnyText shared parents use the first normalized source lexicographically.
    Augmentation retains its published primary source. No parent is copied to
    several training records, and no annotation/crop variant is filtered.
    """
    raw = {alias['source'] for alias in parent.get('aliases', []) if isinstance(alias, dict) and alias.get('source')}
    if 'crop_variants' in parent:
        raw.add(parent['source'])
        primary = ALIASES[parent['source']]
    else:
        if not raw:
            raise ValueError('Missing source aliases for AnyText parent')
        primary = min(ALIASES[name] for name in raw)
    tags = sorted({ALIASES[name] for name in raw})
    return primary, tags, sorted(raw)
