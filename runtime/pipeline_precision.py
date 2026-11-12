"""Explicit compute dtype for pipelines with FP32 FSDP masters and BF16 VAE.

Diffusers0.32.2 DiffusionPipeline.dtype chooses the first module in a set of
signature keys. A mixed-master pipeline can therefore report FP32 or BF16
depending on Python hash order. Atlas normalizes pixels *after* casting, so
this changes conditioning values. Pin compute properties to the frozen VAE.
No library patch, optimizer dtype change, or attention/softmax replacement.
"""
import torch
from diffusers import FluxFillPipeline

class StableTrainingPipeline(FluxFillPipeline):
    @property
    def dtype(self):return self.vae.dtype
    @property
    def device(self):return self.vae.device

def bind_training_precision(pipe):
    if not isinstance(pipe,FluxFillPipeline):raise TypeError('Expected native FluxFillPipeline')
    before=str(pipe.dtype)
    if pipe.vae.dtype!=torch.bfloat16:raise ValueError('This recipe requires BF16 VAE compute')
    if pipe.vae.device.type!='cuda':raise ValueError('Move VAE to its local CUDA device before binding')
    pipe.__class__=StableTrainingPipeline
    return dict(prior_inferred_dtype=before,compute_dtype=str(pipe.dtype),compute_device=str(pipe.device),
                master_parameters='FP32',rule='Pipeline compute dtype/device are explicitly the frozen VAE dtype/device')
