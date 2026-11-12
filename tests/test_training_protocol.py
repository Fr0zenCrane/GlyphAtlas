"""CPU regression tests for global BS128, C4 weighting, and resume cursor."""
import importlib.util,itertools,json,os,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('GLYPH_OMNI_ROOT',str(ROOT));sys.path.insert(0,str(ROOT/'runtime'))
from dataset import UpdateSampler
from glyph_atlas.stage1_ocr import ocr_time_multiplier
spec=importlib.util.spec_from_file_location('portable_training',ROOT/'runtime/train.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)

class ProtocolTests(unittest.TestCase):
    def test_global_batch_disjoint_and_resume(self):
        for world in (8,16,32):
            samplers=[UpdateSampler(397,r,world) for r in range(world)]
            streams=[list(itertools.islice(iter(s),s.accumulation*7)) for s in samplers]
            for update in range(6):
                group=[streams[r][update*samplers[r].accumulation+j] for r in range(world) for j in range(samplers[r].accumulation)]
                self.assertEqual(len(group),128)
                self.assertEqual(len(set(group)),128)
                self.assertEqual({e for _,e in group},{update//3})
            for rank,s in enumerate(samplers):
                resumed=UpdateSampler(397,rank,world);resumed.load_state_dict(s.state_dict(4))
                actual=list(itertools.islice(iter(resumed),s.accumulation*2))
                self.assertEqual(actual,streams[rank][4*s.accumulation:6*s.accumulation])
    def test_accumulated_gradient_matches_full_batch(self):
        torch.manual_seed(42);x=torch.randn(128,3,dtype=torch.float64);y=torch.randn(128,2,dtype=torch.float64)
        t=torch.sigmoid(torch.randn(128,dtype=torch.float64))
        w=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
        full=(((x@w-y)**2).mean(1)*(.25*2*t)).mean();full.backward();expected=w.grad.clone()
        for world in (8,16,32):
            gradients=[]
            for rank in range(world):
                v=w.detach().clone().requires_grad_(True)
                for micro in range(128//world):
                    i=micro*world+rank
                    loss=((x[i]@v-y[i])**2).mean()*.25*(2*t[i])/(128//world)
                    loss.backward()
                gradients.append(v.grad)
            torch.testing.assert_close(torch.stack(gradients).mean(0),expected,rtol=1e-12,atol=1e-12)
    def test_c4_is_per_sample_and_no_sampling_change(self):
        for t in [0.,.25,.5,1.]:
            actual=ocr_time_multiplier(torch.tensor([t]),'high_normalized_native')
            self.assertEqual(float(actual),2*t)
        with self.assertRaises(AssertionError):ocr_time_multiplier(torch.tensor([.1,.9]),'high_normalized_native')
    def test_lr_endpoints(self):
        cfg=json.loads((ROOT/'configs/c4_bs128.json').read_text());cfg['steps']=12500
        self.assertAlmostEqual(training.learning_rate(1,cfg),2e-5/500)
        self.assertAlmostEqual(training.learning_rate(500,cfg),2e-5)
        self.assertAlmostEqual(training.learning_rate(12500,cfg),2e-6)
    def test_random_context_unique_across_ranks_and_accumulation(self):
        for world in (8,16,32,64,128):
            accumulation=128//world
            contexts=[(step*accumulation+micro)*world+rank+1 for step in range(3)
                      for micro in range(accumulation) for rank in range(world)]
            self.assertEqual(contexts,list(range(1,385)))
    def test_random_sizes_do_not_consume_native_rng(self):
        import random
        from glyph_atlas.resolution_data import allocate
        from glyph_atlas.budget import GlyphUnit,InkExtent
        from glyph_atlas.data import FONT
        units=[GlyphUnit(str(i),'字',0,64,(InkExtent(20,20),),str(FONT)) for i in range(200)]
        units.append(GlyphUnit('punct','。',0,64,(InkExtent(20,20),),str(FONT)))
        py_state=random.getstate();torch_state=torch.get_rng_state().clone()
        plan=allocate(units,'random',1,0)[0]
        self.assertEqual(py_state,random.getstate());self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertEqual(plan.allocations[-1].height,64)
        self.assertEqual(plan,allocate(units,'random',1,0)[0])
        self.assertNotEqual(plan,allocate(units,'random',2,0)[0])

if __name__=='__main__':unittest.main()
