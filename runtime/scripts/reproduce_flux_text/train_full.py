"""Eight-GPU full-parameter FLUX-Text, AdamW, explicit warmup/cosine schedule."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
    ShardedStateDictConfig, ShardedOptimStateDictConfig, FullStateDictConfig)
from safetensors.torch import save_file
import lightning as L
import yaml

from train import ROOT,CODE,write,sha
MODELS=Path(os.environ.get('GLYPH_MODEL_ROOT',str(ROOT/'assets/models')))
from full_model import FullDiTModel,shard_transformer
from src.data.dataloader import prepare_dataloader


def learning_rate(step,peak,warmup=500,total=12500,end_ratio=.1):
    assert 1<=step<=total and 0<warmup<total
    if step<=warmup:return peak*step/warmup
    progress=(step-warmup)/(total-warmup)
    return peak*(end_ratio+(1-end_ratio)*.5*(1+math.cos(math.pi*progress)))


def device_batch(value,device):
    if isinstance(value,torch.Tensor):return value.to(device,non_blocking=True)
    if isinstance(value,dict):return {k:device_batch(v,device) for k,v in value.items()}
    if isinstance(value,list):return [device_batch(v,device) for v in value]
    if isinstance(value,tuple):return tuple(device_batch(v,device) for v in value)
    return value


def export_transformer(model,folder):
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        weights=model.state_dict()
    if dist.get_rank()==0:
        folder.mkdir(exist_ok=True)
        config=dict(model.module.config);write(folder/'config.json',config)
        # Save BF16 inference shards; FP32 masters/optimizer remain in DCP.
        groups=[];group={};size=0
        for name,tensor in weights.items():
            assert 'lora_' not in name and '_fsdp_wrapped_module' not in name
            tensor=tensor.to(dtype=torch.bfloat16).contiguous();n=tensor.numel()*tensor.element_size()
            if group and size+n>5_000_000_000:groups.append(group);group={};size=0
            group[name]=tensor;size+=n
        if group:groups.append(group)
        mapping={};files=[];total_size=0
        for index,group in enumerate(groups,1):
            name=f'diffusion_pytorch_model-{index:05d}-of-{len(groups):05d}.safetensors'
            save_file(group,str(folder/name),metadata={'format':'pt'})
            files.append(dict(file=name,sha256=sha(folder/name),bytes=(folder/name).stat().st_size))
            for key,tensor in group.items():mapping[key]=name;total_size+=tensor.numel()*tensor.element_size()
        write(folder/'diffusion_pytorch_model.safetensors.index.json',dict(metadata={'total_size':total_size},weight_map=mapping))
        write(folder/'weights_manifest.json',dict(files=files,total_tensor_bytes=total_size,precision='BF16 inference export from FP32 master parameters'))
    del weights
    dist.barrier()


def save_checkpoint(model,optimizer,sampler,folder,step,config,export=True):
    from distributed_layout import checkpoint_layout
    layout=checkpoint_layout(model);node_shard=layout['mode']=='node_shard'
    rank=dist.get_rank();world=dist.get_world_size();destination=folder/f'checkpoint-{step:06d}';temp=folder/f'.checkpoint-{step:06d}.tmp'
    if rank==0:
        assert not destination.exists(),f'Checkpoint already exists: {destination}'
        assert not temp.exists(),f'Unfinished checkpoint exists: {temp}'
        temp.mkdir(parents=True)
    dist.barrier()
    with FSDP.state_dict_type(model,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True),ShardedOptimStateDictConfig(offload_to_cpu=True)):
        if node_shard:
            # Replicated nodes must not publish duplicate DCP shards. A fixed
            # topology uses local FP32 AdamW state keyed by the local shard rank.
            state={}
            if rank<layout['shard_size']:
                state={'model':model.state_dict()}
                dcp.save(state,checkpoint_id=temp/'distributed',process_group=model.process_group)
                torch.save(optimizer.state_dict(),temp/f'optimizer-shard{rank}.pt')
        else:
            state={'model':model.state_dict(),'optimizer':FSDP.optim_state_dict(model,optimizer)}
            dcp.save(state,checkpoint_id=temp/'distributed')
    del state
    torch.save(dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
                    sampler=sampler.state_dict(step)),temp/f'rng-rank{rank}.pt')
    dist.barrier()  # Every RNG shard must exist before rank0 can publish the directory.
    if export:export_transformer(model,temp/'transformer')
    if rank==0:
        write(temp/'training_state.json',dict(step=step,world_size=world,config=config,optimizer='AdamW',
             checkpoint_layout=layout,optimizer_storage='local_fp32_shards_v1' if node_shard else 'dcp_sharded',
             master_parameters='FP32',optimizer_moments='FP32',sample_presentations=step*config.get('global_batch',world),
             resume_limit='Model/optimizer/RNG/sampler cursor retained; prefetched worker RNG is not a byte-exact restart'))
        temp.rename(destination)
        write(folder/'checkpoint.json',dict(step=step,path=str(destination),complete=True,immutable=True))
    dist.barrier()
    return destination


def restore_checkpoint(model,optimizer,sampler,path):
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    info=json.loads((path/'training_state.json').read_text());assert info['world_size']==dist.get_world_size()
    from distributed_layout import checkpoint_layout
    layout=checkpoint_layout(model);assert info.get('checkpoint_layout',dict(mode='world_full',shard_size=dist.get_world_size(),replicas=1))==layout,'Checkpoint topology changed'
    node_shard=layout['mode']=='node_shard'
    with FSDP.state_dict_type(model,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True),ShardedOptimStateDictConfig(offload_to_cpu=True)):
        if node_shard:
            # PyTorch2.4's subgroup loaders assume global coordinator0 and its
            # dcp.load() key gather allocates WORLD-sized lists. Load only the
            # first node via the state-dict API, then broadcast matching shards.
            if dist.get_rank()<layout['shard_size']:
                state={'model':model.state_dict()}
                dcp.load_state_dict(state,storage_reader=dcp.FileSystemReader(path/'distributed'),process_group=model.process_group)
                model.load_state_dict(state['model'])
        else:
            state={'model':model.state_dict()};dcp.load(state,checkpoint_id=path/'distributed');model.load_state_dict(state['model'])
            restored=load_sharded_optimizer_state_dict(model_state_dict=state['model'],optimizer_key='optimizer',storage_reader=dcp.FileSystemReader(path/'distributed'))
            optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model,optimizer,restored['optimizer']))
    if node_shard:
        from distributed_layout import synchronize_initial_replicas
        dist.barrier();synchronize_initial_replicas(model)
        assert info['optimizer_storage']=='local_fp32_shards_v1'
        restored=torch.load(path/f'optimizer-shard{dist.get_rank(model.process_group)}.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(restored)
    rng=torch.load(path/f'rng-rank{dist.get_rank()}.pt',map_location='cpu',weights_only=False)
    torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python']);np.random.set_state(rng['numpy']);sampler.load_state_dict(rng['sampler'])
    return info['step']
