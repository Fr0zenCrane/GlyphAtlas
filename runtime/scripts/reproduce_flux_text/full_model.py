import os
"""Full-DiT FLUX-Text with the native forward/loss and FSDP module hooks."""
import types

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock

from src.train.model import OminiModelFIll
from src.flux import transformer as native_transformer
from src.flux.block import block_forward, single_block_forward
from src.train import model as native_training


def call_module(module=None,*args,**kwargs):
    # The official activation checkpoint invokes block_forward(self=block,...).
    if module is None:module=kwargs.pop('self')
    return module(*args,**kwargs)


class FullDiTModel(OminiModelFIll):
    def init_lora(self,lora_path,lora_config):
        assert lora_path is None
        self.transformer.requires_grad_(True)
        return []


def install_module_forward(transformer):
    for block in transformer.transformer_blocks:
        block.forward=types.MethodType(block_forward,block)
    for block in transformer.single_transformer_blocks:
        block.forward=types.MethodType(single_block_forward,block)
    transformer.forward=types.MethodType(native_transformer.tranformer_forward,transformer)
    native_transformer.block_forward=call_module
    native_transformer.single_block_forward=call_module
    native_training.tranformer_forward=call_module


def shard_transformer(model,rank):
    transformer=model.transformer
    assert not any('lora_' in name for name,_ in transformer.named_parameters())
    count=sum(p.numel() for p in transformer.parameters())
    assert count>11_000_000_000 and all(p.requires_grad for p in transformer.parameters())
    transformer.float()  # FP32 master parameters; BF16 computation below.
    install_module_forward(transformer)
    sharded=FSDP(transformer,device_id=rank,
        auto_wrap_policy=ModuleWrapPolicy({FluxTransformerBlock,FluxSingleTransformerBlock}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16),
        use_orig_params=True,sync_module_states=True,limit_all_gathers=True)
    model.transformer=sharded;model.flux_pipe.transformer=sharded
    model._device=torch.device('cuda',rank);model._dtype=torch.bfloat16
    for module in [model.flux_pipe.vae,model.flux_pipe.text_encoder,model.flux_pipe.text_encoder_2]:
        module.to(device=model.device,dtype=torch.bfloat16).requires_grad_(False).eval()
    if model.odm_loss is not None:model.odm_loss.ResNet1.to(model.device).requires_grad_(False).eval()
    return count
