"""Validate portable 512px requests before importing the GPU stack."""
import json
import math
from pathlib import Path
import re
import unicodedata
from PIL import Image


def load_requests(path, *, training=False):
    path = Path(path).resolve()
    if path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
        rows = rows if isinstance(rows, list) else [rows]
    if not rows:
        raise ValueError('Input contains no requests')
    seen = set()
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - {'id', 'image', 'prompt', 'regions', 'seed'}:
            raise ValueError(f'Request {index}: unexpected fields or non-object input')
        identity = row.get('id', f'{index:06d}')
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', identity):
            raise ValueError('id must contain only ASCII letters, digits, underscores, or hyphens')
        if identity in seen:
            raise ValueError('Duplicate request id: ' + identity)
        seen.add(identity)
        image = (path.parent / row['image']).resolve()
        with Image.open(image) as im:
            if im.size != (512, 512):
                raise ValueError(f'{identity}: expected 512x512; resize image AND polygons explicitly')
            if im.getexif().get(274, 1) != 1:
                raise ValueError(f'{identity}: normalize EXIF orientation and polygon coordinates first')
            im.load()
        prompt = row.get('prompt', '')
        if not isinstance(prompt, str) or '*' in prompt:
            raise ValueError('prompt must be a string without the reserved * placeholder')
        regions = row['regions']
        if not isinstance(regions, list) or not 1 <= len(regions) <= 8:
            raise ValueError('Each request requires 1 to 8 text regions')
        clean = []
        for region in regions:
            if not isinstance(region, dict) or set(region) - {'text', 'polygon', 'language'}:
                raise ValueError('Unexpected region fields')
            text = region['text']
            if not isinstance(text, str) or not text.strip() or '*' in text:
                raise ValueError('Text must be nonempty and cannot contain reserved *')
            if any(unicodedata.category(c) in {'Cc', 'Cs'} for c in text):
                raise ValueError('Text cannot contain control characters, newlines, or surrogates')
            if len(text) > (20 if training else 256):
                raise ValueError('Text exceeds the supported length; it will not be silently truncated')
            polygon = region['polygon']
            if not isinstance(polygon, list) or len(polygon) != 4:
                raise ValueError('Provide four polygon corners in perimeter order')
            for point in polygon:
                if not isinstance(point, list) or len(point) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x != int(x) or not 0 <= x <= 511 for x in point):
                    raise ValueError('Polygon coordinates must be integers in [0, 511]')
            polygon = [[int(x) for x in point] for point in polygon]
            cross = []
            for i in range(4):
                a, b, c = polygon[i], polygon[(i+1)%4], polygon[(i+2)%4]
                cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
            if not (all(v > 0 for v in cross) or all(v < 0 for v in cross)):
                raise ValueError('Polygon must be a nondegenerate convex quadrilateral in perimeter order')
            area = abs(sum(polygon[i][0]*polygon[(i+1)%4][1]-polygon[(i+1)%4][0]*polygon[i][1] for i in range(4))) / 2
            if area < 16:
                raise ValueError('Polygon is too small to rasterize reliably')
            language = region.get('language', 'Other')
            if not isinstance(language, str) or not language:
                raise ValueError('language must be a nonempty string')
            clean.append(dict(text=text, polygon=polygon, language=language))
        seed = row.get('seed', 42)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        result.append(dict(id=identity, image=str(image), prompt=prompt, regions=clean, seed=seed))
    return result
