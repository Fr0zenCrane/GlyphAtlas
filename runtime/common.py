"""Portable paths; this module never starts workers or a monitor."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('GLYPH_OMNI_ROOT', Path(__file__).resolve().parents[1])).resolve()
if ROOT != Path(__file__).resolve().parents[1]:
    raise RuntimeError('GLYPH_OMNI_ROOT must point to this checkout; unset stale workspace overrides')
os.environ.setdefault('GLYPH_OMNI_ROOT', str(ROOT))
CODE = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/FluxText'
MODELS = Path(os.environ.get('GLYPH_MODEL_ROOT', ROOT / 'assets/models')).resolve()
RUN = Path(os.environ.get('GLYPH_RUN_DIR', ROOT / 'runs/default')).resolve()
sys.path[:0] = [str(CODE / 'scripts'), str(CODE / 'scripts/reproduce_flux_text'), str(VENDOR)]

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()
