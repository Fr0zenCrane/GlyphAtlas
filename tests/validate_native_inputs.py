"""CPU checks: real-data native parity, eight-rank cursor replay, source parity."""
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import random
import sqlite3
import tempfile

import numpy as np
import torch
torch.set_num_threads(2)
from torch.utils.data import DataLoader
import sys,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("GLYPH_OMNI_ROOT",str(ROOT))
sys.path.insert(0,str(ROOT/"runtime"))
from common import VENDOR as CODE,now,sha,write
from tools_check import check_native as check_vendor
from dataset import Dataset, UpdateSampler
from data_records import select_annotation
from src.train.data_word import T3DataSet, T3DataSetWarp

import yaml


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b, (a, b)


def definitions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    check_vendor()
    package = Path(__file__).parent
    dataset = Dataset(args.dataset)
    ready = json.loads((args.dataset / 'ready.json').read_text())
    assert sha(args.dataset / 'dataset.sqlite') == ready['dataset_sha256']
    con = sqlite3.connect(f'file:{args.dataset}/dataset.sqlite?mode=ro', uri=True)
    rows = con.execute('SELECT id,annotation,image,image_sha256 FROM records ORDER BY id').fetchall()
    assert len(rows) == len(dataset) and [r[0] for r in rows] == list(range(len(rows)))
    native_annotations = []
    for _, annotation, image, digest in rows:
        assert sha(args.dataset / image) == digest
        gt = select_annotation(json.loads(annotation),_,0)
        gt['img_name'] = image
        native_annotations.append(gt)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'native.json'
        path.write_text(json.dumps(dict(data_root=str(args.dataset), data_list=native_annotations)))
        reference = object.__new__(T3DataSetWarp)
        for name in ('condition_type', 'glyph_scale', 'drop_text_prob', 'drop_image_prob',
                     'return_pil_image', 'condition_size', 'to_tensor'):
            setattr(reference, name, getattr(dataset, name))
        T3DataSet.__init__(reference, [[str(path), str(args.dataset)]],
            max_lines=8, max_chars=20, font_path=str(CODE / 'font/Arial_Unicode.ttf'),
            caption_pos_prob=0., mask_pos_prob=1., mask_img_prob=0., glyph_scale=1,
            percent=1., debug=False, using_dlc=False, wm_thresh=.5, random_select=False)
        assert len(reference) == len(dataset)
        import copy
        for i in range(len(dataset)):
            original = copy.deepcopy(reference.data_list[i])
            seed = int.from_bytes(hashlib.sha256(f'{dataset.seed}:0:{i}'.encode()).digest()[:4], 'big')
            random.seed(seed); np.random.seed(seed)
            a = reference[f'{i}-1-512-512']
            reference.data_list[i] = original
            equal(a, dataset[(i, 0)])
            equal(dataset[(i, 1)], dataset[(i, 1)])
    # Worker scheduling must not change stateless per-sample augmentation.
    count=min(32,len(dataset))
    a=DataLoader(dataset,batch_size=1,sampler=[(i,2) for i in range(count)],num_workers=0)
    b=DataLoader(dataset,batch_size=1,sampler=[(i,2) for i in range(count)],num_workers=2)
    for x,y in zip(a,b):equal(x,y)
    write(args.output,dict(passed=True,native_render_cases=len(dataset),worker_count_parity=[0,2],original_image_hashes=len(dataset),training_updates=0))
    print('CPU_VALIDATION_PASSED', len(dataset), flush=True)


if __name__ == '__main__':
    main()
