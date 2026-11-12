import sys,unittest
from pathlib import Path
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from pipeline_precision import StableTrainingPipeline

class Component(torch.nn.Module):
    def __init__(self,dtype):super().__init__();self.dtype=dtype;self.device=torch.device('cpu')
class MixedPipeline:
    def __init__(self):self.transformer=Component(torch.float32);self.vae=Component(torch.bfloat16);self.order=['transformer','vae']
    @staticmethod
    def _get_signature_keys(obj):return obj.order,set()

class PrecisionTests(unittest.TestCase):
    def test_original_property_depends_on_module_order(self):
        p=MixedPipeline();a=DiffusionPipeline.dtype.fget(p)
        p.order.reverse();b=DiffusionPipeline.dtype.fget(p)
        self.assertEqual((a,b),(torch.float32,torch.bfloat16))
    def test_explicit_dtype_independent_of_component_order(self):
        p=MixedPipeline();a=StableTrainingPipeline.dtype.fget(p)
        p.order.reverse();b=StableTrainingPipeline.dtype.fget(p)
        self.assertEqual(a,torch.bfloat16);self.assertEqual(a,b)
    def test_old_dtype_choice_changes_atlas_input(self):
        pixels=torch.arange(256,dtype=torch.float32)
        bf16=(pixels.to(torch.bfloat16)/127.5-1)
        fp32=(pixels/127.5-1).to(torch.bfloat16)
        self.assertGreater(int((bf16!=fp32).sum()),0)

if __name__=='__main__':unittest.main()
