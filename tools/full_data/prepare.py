"""Full public annotations -> native-quality gates -> original image bytes.

No 100K sampling, language balancing or train/train deduplication. Materializes
only eligible images, grouped by parquet row group. No model or GPU is loaded.
An interrupted materialization can be resumed; incomplete selection is rebuilt
in a fresh output directory. --limit-per-source is a clearly marked smoke set.
"""
import argparse
import ast
from collections import Counter
import io
import json
from pathlib import Path
import sqlite3
import types

import cv2
import ijson
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from common import ROOT, CODE, check_vendor, now, sha, write


def native_gate():
    tree = ast.parse((CODE / 'src/train/data_word.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'T3DataSet')
    methods = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in ('is_vertical_screen', 'isvalid_img')]
    scope = dict(cv2=cv2, np=np)
    exec(compile(ast.Module(body=methods, type_ignores=[]), 'native_data_filter', 'exec'), scope)
    obj = types.SimpleNamespace()
    for method in methods:
        setattr(obj, method.name, types.MethodType(scope[method.name], obj))
    return obj


def benchmark_hashes():
    result = set()
    root = ROOT / 'data/raw/AnyText-benchmark/benchmark_extracted/benchmark'
    for split in ('laion_word', 'wukong_word'):
        for row in json.loads((root / split / 'test1k.json').read_text())['data_list']:
            with Image.open(root / split / 'imgs' / row['img_name']) as image:
                import hashlib
                result.add(hashlib.sha256(image.convert('RGB').tobytes()).hexdigest())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-uids', type=Path, default=Path(__file__).with_name('known_exclusions.json'))
    p.add_argument('--limit-per-source', type=int, default=0)
    args = p.parse_args()
    assert args.limit_per_source >= 0
    check_vendor()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / '.prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'ready.json').exists():
        raise RuntimeError('Dataset already complete; use it without rebuilding')
    exclusions = json.loads(args.exclude_uids.read_text())
    excluded = set(exclusions['uids'])
    recipe = dict(limit_per_source=args.limit_per_source, exclusions_sha256=sha(args.exclude_uids),
                  release_sha256=sha(ROOT / 'data/raw/FLUX-Text/release.json'))
    db_path = out / 'dataset.sqlite'
    if not db_path.exists():
        db = sqlite3.connect(db_path)
        db.executescript('''
            CREATE TABLE candidates (uid TEXT PRIMARY KEY, source TEXT, shard TEXT, rg INT,
                row_num INT, annotation TEXT, status TEXT);
            CREATE INDEX by_location ON candidates(shard,rg,row_num);
            CREATE TABLE records (id INTEGER PRIMARY KEY, uid TEXT UNIQUE, annotation TEXT,
                image TEXT, image_sha256 TEXT, rgb_sha256 TEXT);
        ''')
        write(out / 'recipe.json', recipe)
        index = sqlite3.connect(f'file:{ROOT}/data/processed/flux-text-english-v1/image_index.sqlite?mode=ro', uri=True)
        for path, size, mtime, _ in index.execute('SELECT * FROM shards'):
            stat = (ROOT / path).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (size, mtime), path
        gate, counts = native_gate(), Counter()
        release = json.loads((ROOT / 'data/raw/FLUX-Text/release.json').read_text())
        for entry in release['files']:
            path = ROOT / 'data/raw/FLUX-Text' / entry['path']
            assert sha(path) == entry['local_sha256'], path
            source, selected = path.parent.name, 0
            with path.open('rb') as stream:
                for gt in ijson.items(stream, 'data_list.item', use_float=True):
                    counts['raw_records_scanned'] += 1
                    if gt.get('wm_score', 0) > .5:
                        counts['watermark_rejected'] += 1
                        continue
                    if 'annotations' not in gt or not gate.isvalid_img(gt['annotations']):
                        counts['annotation_or_recognition_rejected'] += 1
                        continue
                    counts['native_eligible'] += 1
                    uid = f'{source}/{gt["img_name"]}'
                    if uid in excluded:
                        counts['known_benchmark_or_heldout_cluster_rejected'] += 1
                        continue
                    assert not gt.get('use_re'), f'Replacement-pair data needs explicit support: {uid}'
                    assert gt['width'] == gt['height'] == 512
                    loc = index.execute('SELECT shard,rg,row FROM images WHERE source=? AND img_name=?',
                                        (source, gt['img_name'])).fetchone()
                    assert loc is not None, uid
                    db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,NULL)',
                               (uid, source, *loc, json.dumps(gt, ensure_ascii=False)))
                    selected += 1
                    if args.limit_per_source and selected >= args.limit_per_source:
                        break
            counts['selected'] += selected
            db.commit()
            print(source, selected, dict(counts), flush=True)
        write(out / 'selection.json', dict(counts=counts, recipe=recipe, checked_beijing=now()))
    else:
        assert (out / 'selection.json').exists(), 'Interrupted selection: use a fresh output directory'
        assert json.loads((out / 'recipe.json').read_text()) == recipe, 'Resume with identical preparation options'
        db = sqlite3.connect(db_path)

    heldout = benchmark_hashes()
    next_id = db.execute('SELECT count(*) FROM records').fetchone()[0]
    for (shard,) in db.execute('SELECT DISTINCT shard FROM candidates WHERE status IS NULL').fetchall():
        file = pq.ParquetFile(ROOT / shard)
        groups = db.execute('SELECT DISTINCT rg FROM candidates WHERE shard=? AND status IS NULL ORDER BY rg', (shard,)).fetchall()
        for (rg,) in groups:
            table = file.read_row_group(rg, columns=['img_name', 'image'])
            candidates = db.execute('SELECT uid,row_num,annotation FROM candidates WHERE shard=? AND rg=? AND status IS NULL ORDER BY row_num', (shard, rg)).fetchall()
            for uid, row_num, annotation in candidates:
                item = table.slice(row_num, 1).to_pylist()[0]
                gt = json.loads(annotation)
                assert item['img_name'] == gt['img_name'], uid
                payload = item['image']['bytes']
                import hashlib
                with Image.open(io.BytesIO(payload)) as image:
                    rgb = image.convert('RGB')
                    assert rgb.size == (512, 512), uid
                    rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                if rgb_hash in heldout:
                    db.execute('UPDATE candidates SET status=? WHERE uid=?', ('exact_benchmark_pixels', uid))
                    continue
                # uid-derived path prevents basename collisions across corpora.
                name = hashlib.sha256(uid.encode()).hexdigest()
                relative = f'images/{name[:2]}/{name}.img'
                destination = out / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(payload).hexdigest()
                if destination.exists():
                    assert sha(destination) == digest, destination
                else:
                    temporary = destination.with_suffix('.tmp')
                    temporary.write_bytes(payload)
                    temporary.replace(destination)
                db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)', (next_id, uid, annotation, relative, digest, rgb_hash))
                next_id += 1
                db.execute('UPDATE candidates SET status=? WHERE uid=?', ('ready', uid))
            db.commit()
        print('materialized', shard, flush=True)
    counts = dict(db.execute('SELECT status,count(*) FROM candidates GROUP BY status'))
    assert None not in counts
    count = db.execute('SELECT count(*) FROM records').fetchone()[0]
    sources = dict(db.execute('SELECT c.source,count(*) FROM records r JOIN candidates c ON r.uid=c.uid GROUP BY c.source'))
    db.commit()
    db.close()
    write(out / 'ready.json', dict(complete=True, images=count, smoke_only=bool(args.limit_per_source),
          dataset_sha256=sha(db_path), counts=counts, source_counts=sources, checked_beijing=now(), recipe=recipe,
          language_filter='none; all scripts retained', train_train_deduplication=False,
          exclusion_scope='Exact benchmark RGB plus known benchmark/development duplicate-cluster UIDs; unknown duplicates remain possible',
          native_quality_filter=True, author_training_identities_claim=False))


if __name__ == '__main__':
    main()
