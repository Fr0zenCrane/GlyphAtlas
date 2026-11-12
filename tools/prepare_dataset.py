"""Import explicit 512px training examples without filtering or resampling."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from requests_io import load_requests


def prepare(source, destination, smoke_only=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new output directory; datasets are immutable')
    rows = load_requests(source, training=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dataset-', dir=destination.parent) as temp:
        root = Path(temp) / 'prepared'
        (root / 'images').mkdir(parents=True)
        hashes = set()
        with sqlite3.connect(root / 'dataset.sqlite') as db:
            db.execute('CREATE TABLE records(id INTEGER PRIMARY KEY, image TEXT, image_sha256 TEXT, annotation TEXT)')
            for i, row in enumerate(rows):
                raw = Path(row['image']).read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest in hashes:
                    raise ValueError('Duplicate image bytes: represent annotation variants with the prepared parent release, not duplicate parents')
                hashes.add(digest)
                rel = 'images/' + row['id'] + Path(row['image']).suffix.lower()
                (root / rel).write_bytes(raw)
                annotation = dict(img_name=rel, width=512, height=512, caption=row['prompt'],
                    annotations=[dict(r, valid=True, pos=-1) for r in row['regions']])
                db.execute('INSERT INTO records VALUES(?,?,?,?)', (i, rel, digest, json.dumps(annotation, ensure_ascii=False)))
        ready = dict(complete=True, format='glyph-custom-images-v1', images=len(rows),
            smoke_only=smoke_only, dataset_sha256=hashlib.sha256((root/'dataset.sqlite').read_bytes()).hexdigest(),
            input_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            preparation='Exact input image bytes and text; no resize, filtering, splitting, or reweighting')
        (root / 'ready.json').write_text(json.dumps(ready, indent=2)+'\n')
        root.rename(destination)
    return ready


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.smoke_only), indent=2))
