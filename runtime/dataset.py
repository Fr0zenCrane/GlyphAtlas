"""Lazy metadata, native rendering, fixed 512px, reproducible sample transforms."""
import hashlib
import json
import os
import random
import sqlite3
from pathlib import Path

import numpy as np
from PIL import ImageFont
import torch
from torch.utils.data import Sampler
import torchvision.transforms as T
from common import VENDOR as CODE
from data_records import select_record
from src.train.data_word import T3DataSetWarp
from src.train import data_word
from font_support import install_fallback
install_fallback(data_word)


class Rows:
    def __init__(self, folder, count):
        self.folder, self.count = Path(folder), count
        self.connection, self.pid = None, None
        self.epoch = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.pid != os.getpid():
            self.connection = sqlite3.connect(f'file:{self.folder}/dataset.sqlite?mode=ro', uri=True)
            self.pid = os.getpid()
        row = self.connection.execute('SELECT annotation,image FROM records WHERE id=?', (int(index),)).fetchone()
        if row is None:
            raise IndexError(index)
        gt,relative,augmentation=select_record(json.loads(row[0]),index,self.epoch,row[1])
        path=str(self.folder/relative)
        # Exactly the public load_data conversion; fresh arrays prevent in-place
        # polygon/invalid-region changes from leaking into a later presentation.
        valid = [a for a in gt['annotations'] if a['polygon'] and a['text'] and a.get('valid') is not False]
        invalid = [a for a in gt['annotations'] if a['polygon'] and (augmentation or a['text']) and a.get('valid') is False]
        return dict(img_path=path, ref_path=path, caption=gt.get('caption', '').replace('*', ' '),
                    polygons=[np.array(a['polygon'],dtype=np.int32 if augmentation else None) for a in valid],
                    invalid_polygons=[np.array(a['polygon'],dtype=np.int32 if augmentation else None) for a in invalid],
                    texts=[a['text'] for a in valid], language=[a.get('language','Other') for a in valid],
                    pos=[a.get('pos',-1) for a in valid], width=gt['width'], height=gt['height'],
                    corrected_augmentation=augmentation)


class CombinedRows:
    """Concatenate parents without copying images or giving crops extra weight."""
    def __init__(self,folder,components):
        import hashlib
        with sqlite3.connect(f'file:{Path(folder)}/dataset.sqlite?mode=ro',uri=True) as db:
            actual=db.execute('SELECT path,images,dataset_sha256,format FROM components ORDER BY path').fetchall()
        expected=sorted(tuple(c[k] for k in ['path','images','dataset_sha256','format']) for c in components)
        assert actual==expected,'Combined index and ready metadata disagree'
        assert {c['path'] for c in components}=={'anytext_unique','augmentation'} and len(components)==2
        self.epoch=0;self.children=[];self.ends=[];total=0
        for c in components:
            path=Path(folder)/c['path']
            ready=json.loads((path/'ready.json').read_text())
            assert ready['complete'] and ready['images']==c['images']
            assert ready['dataset_sha256']==c['dataset_sha256']
            h=hashlib.sha256()
            with (path/'dataset.sqlite').open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
            assert h.hexdigest()==c['dataset_sha256'],'Component database changed'
            self.children.append(Rows(path,c['images']));total+=c['images'];self.ends.append(total)
    def __len__(self):return self.ends[-1]
    def __getitem__(self,index):
        import bisect
        if not 0<=index<len(self):raise IndexError(index)
        k=bisect.bisect_right(self.ends,index);offset=self.ends[k-1] if k else 0
        child=self.children[k];child.epoch=self.epoch
        return child[index-offset]


class Dataset(T3DataSetWarp):
    def __init__(self, folder, seed=42):
        ready = json.loads((Path(folder) / 'ready.json').read_text())
        assert ready['complete'] and ready['images'] > 0
        self.seed=seed
        self.data_list=CombinedRows(folder,ready['components']) if ready.get('format')=='glyph-combined-parents-v1' else Rows(folder,ready['images'])
        self.condition_type, self.glyph_scale, self.condition_size = 'word_fill', 1, 512
        self.drop_text_prob = self.drop_image_prob = .1
        self.return_pil_image = self.debug = self.for_show = self.using_dlc = False
        self.random_select = False  # Same setting as our retained Atlas runs.
        self.max_lines, self.max_chars, self.place_holder = 8, 20, '*'
        self.caption_pos_porb, self.mask_pos_prob, self.mask_img_prob = 0., 1., 0.
        self.to_tensor, self.font = T.ToTensor(), ImageFont.truetype(str(CODE / 'font/Arial_Unicode.ttf'), 60)

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (int(key), 0)
        self.data_list.epoch = int(epoch)
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{epoch}:{index}'.encode()).digest()[:4], 'big')
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        previous_max=self.max_chars
        try:
            info=self.data_list[index]
            if info.get('corrected_augmentation'):
                self.max_chars=max(len(text) for text in info['texts'])
            random.seed(seed)
            np.random.seed(seed)
            return super().__getitem__(f'{index}-1-512-512')
        finally:
            self.max_chars=previous_max
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


class UpdateSampler(Sampler):
    """Disjoint global batches; checkpoint cursor counts consumed updates.

    Each epoch drops fewer than global_batch images, reshuffling next epoch.
    Worker prefetch never advances the serialized optimizer-update cursor.
    """
    def __init__(self, size, rank, world=8, global_batch=128, seed=1024):
        assert global_batch % world == 0 and 0 <= rank < world
        self.size, self.rank, self.world = size, rank, world
        self.global_batch, self.seed = global_batch, seed
        self.accumulation = global_batch // world
        self.updates_per_epoch, self.start_step = size // global_batch, 0
        assert self.updates_per_epoch > 0, 'Dataset must contain at least one global batch'

    def state_dict(self, step):
        return dict(start_step=step, size=self.size, world=self.world,
                    global_batch=self.global_batch, seed=self.seed)

    def load_state_dict(self, state):
        for key in ('size', 'world', 'global_batch', 'seed'):
            assert state[key] == getattr(self, key), f'Sampler mismatch: {key}'
        self.start_step = state['start_step']

    def __iter__(self):
        epoch, offset = divmod(self.start_step, self.updates_per_epoch)
        while True:
            order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch))
            for update in range(offset, self.updates_per_epoch):
                for micro in range(self.accumulation):
                    index = update*self.global_batch + micro*self.world + self.rank
                    yield int(order[index]), epoch
            epoch, offset = epoch + 1, 0
