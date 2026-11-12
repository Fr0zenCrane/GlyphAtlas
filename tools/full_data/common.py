"""Paths and serialization; no dependency on historical experiment launchers."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('GLYPH_OMNI_ROOT', Path(__file__).resolve().parents[2])).resolve()
CODE = ROOT / 'vendor/FluxText'
sys.path.insert(0, str(CODE))
BJ = datetime.timezone(datetime.timedelta(hours=8))


def now():
    return datetime.datetime.now(BJ).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def check_vendor():
    expected = {
        'src/train/model.py': '1db2d9e4979d4f40f5ed392acfcd46b971f9b43ee0c16d29ece0f18dfc0f9587',
        'src/train/data_word.py': '3daf6c13f2f83251ccb8355140bc40b296e7e1227c76b85278586bd6a4f59c84',
    }
    for name, digest in expected.items():
        assert sha(CODE / name) == digest, f'Native source changed: {name}'
