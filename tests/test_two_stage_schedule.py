import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('GLYPH_OMNI_ROOT',str(ROOT))
sys.path.insert(0,str(ROOT/'runtime'))
from training_schedule import stage_for_step, switch_encoder, checkpoint_due, verify_complete_checkpoint
spec=importlib.util.spec_from_file_location('two_stage_train',ROOT/'runtime/train.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
from dataset import UpdateSampler


def config(boundary=11993):
    cfg=json.loads((ROOT/'configs/legacy/continuous_resolution_schedule.json').read_text())
    cfg.update(updates_per_epoch=boundary,steps=2*boundary)
    return cfg


class TwoStageTests(unittest.TestCase):
    def test_exact_horizon_lr_and_modes(self):
        cfg=config(); rates=[trainer.learning_rate(i,cfg) for i in range(1,23987)]
        self.assertAlmostEqual(rates[0],5e-5)
        self.assertAlmostEqual(rates[11992],5e-6)
        self.assertTrue(all(a>=b for a,b in zip(rates[:11993],rates[1:11993])))
        self.assertEqual(set(rates[11993:]),{5e-6})
        for step,want in [(1,(1,'fixed64')),(11993,(1,'fixed64')),
                          (11994,(2,'random')),(23986,(2,'random'))]:
            self.assertEqual(stage_for_step(step,cfg),want)
        self.assertEqual([i for i in range(1,23987) if checkpoint_due(i,23986,cfg)], [11993,23986])
        with self.assertRaises(AssertionError):stage_for_step(23987,cfg)

    def test_stage_boundary_sampler_continues_epoch_and_crop_index(self):
        for world in (8,32):
            for rank in range(world):
                s=UpdateSampler(397,rank,world)
                continuous=list(itertools.islice(iter(s),s.accumulation*4))
                resumed=UpdateSampler(397,rank,world)
                resumed.load_state_dict(s.state_dict(3))
                batch=list(itertools.islice(iter(resumed),s.accumulation))
                self.assertEqual(batch,continuous[3*s.accumulation:])
                self.assertEqual({epoch for _,epoch in batch},{1})

    def test_encoder_switch_and_resume_cache_keep_native_rng(self):
        model=SimpleNamespace(flux_pipe=object())
        state=torch.get_rng_state().clone(); py=random.getstate(); npstate=np.random.get_state()
        switch_encoder(model,'fixed64')
        cache=model.atlas_encoder.cache;cache['sentinel']=torch.tensor([1.])
        self.assertTrue(switch_encoder(model,'random'))
        self.assertIs(model.atlas_encoder.cache,cache)
        self.assertEqual(model.atlas_encoder.mode,'random')
        self.assertEqual(len(model.atlas_encoder.dynamic_cache),0)
        self.assertFalse(switch_encoder(model,'random'))
        self.assertTrue(torch.equal(torch.get_rng_state(),state));self.assertEqual(random.getstate(),py)
        now=np.random.get_state();self.assertEqual(now[0],npstate[0]);np.testing.assert_array_equal(now[1],npstate[1]);self.assertEqual(now[2:],npstate[2:])

    def test_adam_and_random_allocator_boundary_resume(self):
        from glyph_atlas.resolution_data import allocate
        from glyph_atlas.budget import GlyphUnit,InkExtent
        from glyph_atlas.data import FONT
        cfg=config(2)
        units=[GlyphUnit(str(i),'字',0,64,(InkExtent(20,20),),str(FONT)) for i in range(8)]
        model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters(),lr=5e-5)
        for step in (1,2):
            for group in opt.param_groups:group['lr']=trainer.learning_rate(step,cfg)
            opt.zero_grad();model(torch.ones(1,2)).square().sum().backward();opt.step()
        weights=copy.deepcopy(model.state_dict());optim=copy.deepcopy(opt.state_dict())
        restored=torch.nn.Linear(2,1);restored.load_state_dict(weights)
        ropt=torch.optim.AdamW(restored.parameters());ropt.load_state_dict(optim)
        context=(2*4+0)*32+0+1
        self.assertEqual(allocate(units,'random',context,0),allocate(units,'random',context,0))
        for m,o in [(model,opt),(restored,ropt)]:
            for group in o.param_groups:group['lr']=trainer.learning_rate(3,cfg)
            o.zero_grad();m(torch.ones(1,2)).square().sum().backward();o.step()
            self.assertTrue(all(float(v['step'])==3 for v in o.state.values()))
        for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_checkpoint_marker_does_not_hide_missing_rank_or_adam(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            info=dict(step=11993,world_size=2,checkpoint_layout=dict(mode='node_shard',shard_size=1))
            (p/'training_state.json').write_text(json.dumps(info))
            (p/'resume_aux_complete.json').write_text(json.dumps(dict(complete=True,step=11993,ranks=2)))
            for name in ['distributed/.metadata','distributed/a.distcp','transformer/config.json',
                         'transformer/diffusion_pytorch_model.safetensors.index.json',
                         'transformer/model.safetensors','rng-rank0.pt','rng-rank1.pt',
                         'aux-rank0.pt','aux-rank1.pt','optimizer-shard0.pt']:
                f=p/name;f.parent.mkdir(exist_ok=True,parents=True);f.write_bytes(b'x')
            import pickle
            with (p/'distributed/.metadata').open('wb') as f:
                pickle.dump(SimpleNamespace(storage_data={'x':SimpleNamespace(relative_path='a.distcp',offset=0,length=1)}),f)
            (p/'transformer/weights_manifest.json').write_text(json.dumps(dict(files=[dict(file='model.safetensors',bytes=1)])))
            self.assertTrue(verify_complete_checkpoint(p)['retain'])
            (p/'optimizer-shard0.pt').unlink()
            with self.assertRaises(AssertionError):verify_complete_checkpoint(p)

    def test_full_dataset_required_and_budget_cannot_expand(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            release=json.loads((ROOT/'configs/glyph_release.json').read_text())
            ready=dict(complete=True,smoke_only=False,format='glyph-combined-parents-v1',images=1535582,
                       source_release=dict(revision=release['revision']),
                       components=[dict(path='anytext_unique',images=1459957),dict(path='augmentation',images=75625)],dataset_sha256='test')
            if release.get('schema')=='glyphatlas-training-release-v2':
                ready.update(format='glyph-source-indexed-v1',parents_by_component=dict(anytext=1459957,augmentation=75625),
                    parents_by_source={name:item['parents'] for name,item in release['sources'].items()})
                ready['source_release'].update(repo_id=release['repo_id'],catalog_sha256=release['catalog']['sha256'],sources=sorted(release['sources']))
            (p/'ready.json').write_text(json.dumps(ready))
            args=SimpleNamespace(config=ROOT/'configs/legacy/continuous_resolution_schedule.json',dataset=p,epochs=2,steps=None,resume=None,allow_smoke_data=False)
            cfg,_=trainer.build_plan(args);self.assertEqual(cfg['steps'],23992)
            args.epochs=3
            with self.assertRaises(AssertionError):trainer.build_plan(args)
            args.epochs=2;ready['images']=1459957;(p/'ready.json').write_text(json.dumps(ready))
            with self.assertRaises(ValueError):trainer.build_plan(args)

if __name__=='__main__':unittest.main()
