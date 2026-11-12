import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('GLYPH_OMNI_ROOT', str(ROOT))
sys.path[:0] = [str(ROOT/'runtime'), str(ROOT/'tools')]
from requests_io import load_requests
from prepare_dataset import prepare
from infer import checkpoint_files


class PublicInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        shutil.copyfile(ROOT/'examples/input.png', self.folder/'input.png')
        self.row = json.loads((ROOT/'examples/edit.json').read_text())

    def tearDown(self):
        self.temp.cleanup()

    def load(self, rows=None, training=False):
        path = self.folder/'input.json'
        path.write_text(json.dumps(self.row if rows is None else rows))
        return load_requests(path, training=training)

    def test_rejects_ambiguous_or_lost_text(self):
        for change in [dict(id='../escape'), dict(seed=True), dict(seed=-1), dict(unknown=1),
                       dict(prompt='reserved * placeholder'), dict(regions=[]),
                       dict(regions=self.row['regions']*9)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(dict(self.row, **change))
        with self.assertRaises(ValueError):
            self.load([self.row, self.row])
        for polygon in [[[0,0],[500,500],[0,500],[500,0]],
                        [[0,0],[512,0],[511,511],[0,511]],
                        [[0,0],[1,0],[1,1],[0,1]],
                        [[0,0],[float('nan'),0],[10,10],[0,10]],
                        [[0,0],[10.5,0],[10,10],[0,10]]]:
            row=copy.deepcopy(self.row);row['regions'][0]['polygon']=polygon
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):self.load(row)
        row=copy.deepcopy(self.row);row['regions'][0]['text']='x'*21
        self.assertEqual(self.load(row)[0]['regions'][0]['text'], 'x'*21)
        with self.assertRaises(ValueError):self.load(row, training=True)

    def test_import_is_exact_and_loader_reads_it(self):
        self.load()
        destination=self.folder/'dataset'
        ready=prepare(self.folder/'input.json', destination, smoke_only=True)
        self.assertTrue(ready['smoke_only'])
        self.assertEqual((destination/'images/bookshop.png').read_bytes(), (self.folder/'input.png').read_bytes())
        from dataset import Rows
        row=Rows(destination, 1)[0]
        self.assertEqual(row['texts'], ['NEW STORIES'])
        self.assertEqual(row['polygons'][0].tolist(), self.row['regions'][0]['polygon'])
        with self.assertRaises(FileExistsError):prepare(self.folder/'input.json', destination)
        second=dict(self.row,id='duplicate')
        self.load([self.row,second])
        with self.assertRaises(ValueError):prepare(self.folder/'input.json',self.folder/'bad')
        self.assertFalse((self.folder/'bad').exists())

    def test_native_renderer_parity_and_rng_preservation(self):
        import numpy as np
        import torch
        from inference_inputs import prepare_input
        from eval.anytext_inference import T3DataSetWarp
        (self.folder/'imgs').mkdir()
        shutil.copyfile(self.folder/'input.png', self.folder/'imgs/input.png')
        reference=self.folder/'native.json'
        native=dict(img_name='input.png',caption=self.row['prompt'],annotations=[
            dict(r,valid=True,recog_valid=True,pos=-1) for r in self.row['regions']])
        reference.write_text(json.dumps(dict(data_list=[native])))
        cwd=Path.cwd()
        try:
            os.chdir(ROOT/'vendor/FluxText')
            baseline=T3DataSetWarp(str(reference),condition_size=512,target_size=512,glyph_scale=1,
                condition_type='word_fill',drop_text_prob=0,drop_image_prob=0,use_filter=False)
        finally:os.chdir(cwd)
        for seed in [0,42,719]:
            request=self.load(dict(self.row, seed=seed))[0]
            state=random.getstate();npstate=np.random.get_state()
            actual=prepare_input(request)
            self.assertEqual(state, random.getstate())
            np.testing.assert_array_equal(npstate[1], np.random.get_state()[1])
            random.seed(seed);np.random.seed(seed)
            expected=baseline[0]
            for field in ['texts','description','n_lines','language']:
                self.assertEqual(actual[field], expected[field], field)
            for field in ['image','condition','hint','positions','gly_line','attnmask']:
                np.testing.assert_array_equal(np.asarray(actual[field]), np.asarray(expected[field]), err_msg=field)
        long=dict(self.row,regions=[dict(self.row['regions'][0],text='A long string above twenty characters')])
        self.assertEqual(prepare_input(self.load(long)[0])['texts'][0],long['regions'][0]['text'])

    def test_checkpoint_missing_shards_and_path_traversal(self):
        cp=self.folder/'cp';cp.mkdir()
        (cp/'config.json').write_text(json.dumps(dict(atlas_version='region-v2')))
        (cp/'part.safetensors').write_bytes(b'fixture')
        entry=dict(file='part.safetensors',sha256=hashlib.sha256(b'fixture').hexdigest())
        (cp/'weights_manifest.json').write_text(json.dumps(dict(files=[entry])))
        index=cp/'diffusion_pytorch_model.safetensors.index.json'
        index.write_text(json.dumps(dict(weight_map={'atlas_projection.weight':'part.safetensors'})))
        checkpoint_files(cp)
        (cp/'config.json').write_text(json.dumps(dict(atlas_version='region-v2',atlas_structure='scene_only')))
        with self.assertRaises(ValueError):checkpoint_files(cp)
        (cp/'config.json').write_text(json.dumps(dict(atlas_version='region-v2')))
        index.write_text(json.dumps(dict(weight_map={'atlas_projection.weight':'missing.safetensors'})))
        with self.assertRaises(ValueError):checkpoint_files(cp)
        entry['file']='../input.png'
        (cp/'weights_manifest.json').write_text(json.dumps(dict(files=[entry])))
        with self.assertRaises(ValueError):checkpoint_files(cp)


if __name__=='__main__':unittest.main()
