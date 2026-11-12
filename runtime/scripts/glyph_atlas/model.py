import os
"""Extra atlas tokens in all native joint-attention layers; global glyph intact."""
import ast
import importlib
import inspect
from pathlib import Path
import sys
import textwrap
import types

ROOT=Path(os.environ["GLYPH_OMNI_ROOT"])
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'vendor/FluxText'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce_flux_text'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock,FluxSingleTransformerBlock
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.checkpoint import checkpoint
from src.flux import block as native_block
from src.flux.transformer import tranformer_forward as native_forward
from full_model import FullDiTModel
from glyph_atlas.data import AtlasEncoder,make_routes


def routed_sdpa(query,key,value,model_config):
    routes=model_config.get('atlas_routes')
    if not routes:return F.scaled_dot_product_attention(query,key,value,dropout_p=0,is_causal=False)
    output=torch.empty_like(query)
    for qi,ki in routes:
        part=F.scaled_dot_product_attention(query.index_select(2,qi),key.index_select(2,ki),value.index_select(2,ki),dropout_p=0,is_causal=False)
        output.index_copy_(2,qi,part)
    return output


tree=ast.parse(textwrap.dedent(inspect.getsource(native_block.attn_forward)))
class ReplaceAttention(ast.NodeTransformer):
    changes=0
    def visit_Call(self,node):
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='F' and node.func.attr=='scaled_dot_product_attention':
            self.changes+=1
            return ast.copy_location(ast.Call(func=ast.Name(id='_routed_sdpa',ctx=ast.Load()),args=[ast.Name(id=n,ctx=ast.Load()) for n in ['query','key','value','model_config']],keywords=[]),node)
        return self.generic_visit(node)
replace=ReplaceAttention();tree=replace.visit(tree);assert replace.changes==2
ast.fix_missing_locations(tree)
ns=dict(native_block.__dict__,_routed_sdpa=routed_sdpa);exec(compile(tree,'<atlas_native_attention>','exec'),ns)
atlas_attn=ns['attn_forward']
block_ns=dict(native_block.__dict__,attn_forward=atlas_attn)
atlas_double=types.FunctionType(native_block.block_forward.__code__,block_ns,argdefs=native_block.block_forward.__defaults__)
atlas_single=types.FunctionType(native_block.single_block_forward.__code__,block_ns,argdefs=native_block.single_block_forward.__defaults__)


def call_module(module=None,*args,**kwargs):
    if module is None:module=kwargs.pop('self')
    return module(*args,**kwargs)


def atlas_forward(self,condition_latents,condition_ids,condition_type_ids,model_config=None,c_t=0,**params):
    cfg=dict(model_config or {});atlas=cfg.get('atlas')
    assert cfg.get('union_cond_attn',True) and not cfg.get('use_byt5_embeds',False) and not cfg.get('add_cond_attn',False)
    if atlas is None or not getattr(self,'atlas_enabled',True):
        cfg.pop('atlas_routes',None)
        result=native_forward(self,condition_latents=condition_latents,condition_ids=condition_ids,condition_type_ids=condition_type_ids,model_config=cfg,c_t=c_t,**params)
        # Keep new parameters in the autograd graph on dropout ranks.
        if self.training and torch.is_grad_enabled():
            anchor=sum(p.sum()*0 for p in self.atlas_projection.parameters())+sum(p.sum()*0 for p in self.atlas_geometry.parameters())
            if isinstance(result,tuple):result=(result[0]+anchor,*result[1:])
            else:result.sample=result.sample+anchor
        return result
    raw=params['hidden_states'];assert raw.shape[1:]==(1024,384) and condition_latents.shape[1:]==(1024,384)
    hidden=self.x_embedder(raw);condition=self.x_embedder(condition_latents)
    atlas_hidden=self.atlas_projection(atlas['features'])+self.atlas_geometry(atlas['geometry'])
    condition=torch.cat((condition,atlas_hidden),dim=1)
    timestep=params['timestep'].to(hidden.dtype)*1000;guidance=params.get('guidance');pooled=params['pooled_projections']
    if guidance is not None:guidance=guidance.to(hidden.dtype)*1000
    temb=self.time_text_embed(timestep,guidance,pooled) if guidance is not None else self.time_text_embed(timestep,pooled)
    cond_t=torch.ones_like(timestep)*c_t*1000
    cond_temb=self.time_text_embed(cond_t,guidance,pooled) if guidance is not None else self.time_text_embed(cond_t,pooled)
    text=self.context_embedder(params['encoder_hidden_states']);assert text.shape[1]==512
    ti,ii=params['txt_ids'],params['img_ids']
    if ti.ndim==3:ti=ti[0]
    if ii.ndim==3:ii=ii[0]
    rotary=self.pos_embed(torch.cat((ti,ii),dim=0))
    cond_rotary=self.pos_embed(torch.cat((condition_ids,atlas['ids']),dim=0))
    cfg['atlas_routes']=atlas['routes']
    self._last_atlas_shapes=dict(scene=list(hidden.shape),global_condition_tokens=1024,atlas_tokens=atlas_hidden.shape[1],routes=len(atlas['routes']))
    use_ckpt=self.training and self.gradient_checkpointing and torch.is_grad_enabled()
    for block in self.transformer_blocks:
        kw=dict(model_config=cfg,hidden_states=hidden,encoder_hidden_states=text,condition_latents=condition,temb=temb,cond_temb=cond_temb,image_rotary_emb=rotary,cond_rotary_emb=cond_rotary)
        text,hidden,condition=checkpoint(block,use_reentrant=False,**kw) if use_ckpt else block(**kw)
    hidden=torch.cat((text,hidden),dim=1)
    for block in self.single_transformer_blocks:
        kw=dict(model_config=cfg,hidden_states=hidden,condition_latents=condition,temb=temb,cond_temb=cond_temb,image_rotary_emb=rotary,cond_rotary_emb=cond_rotary)
        hidden,condition=checkpoint(block,use_reentrant=False,**kw) if use_ckpt else block(**kw)
    output=self.proj_out(self.norm_out(hidden[:,text.shape[1]:],temb))
    return (output,) if not params.get('return_dict',True) else Transformer2DModelOutput(sample=output)


def install(transformer):
    if not hasattr(transformer,'atlas_projection'):
        # Preserve baseline CPU RNG initialization sequence.
        with torch.random.fork_rng(devices=[]):
            width=transformer.x_embedder.out_features
            transformer.atlas_projection=nn.Linear(64,width,bias=False)
            transformer.atlas_geometry=nn.Sequential(nn.Linear(4,128),nn.SiLU(),nn.Linear(128,width))
            with torch.no_grad():
                transformer.atlas_projection.weight.copy_(transformer.x_embedder.weight[:,:64])
                nn.init.zeros_(transformer.atlas_geometry[-1].weight);nn.init.zeros_(transformer.atlas_geometry[-1].bias)
    for block in transformer.transformer_blocks:block.forward=types.MethodType(atlas_double,block)
    for block in transformer.single_transformer_blocks:block.forward=types.MethodType(atlas_single,block)
    transformer.forward=types.MethodType(atlas_forward,transformer);transformer.atlas_enabled=True
    transformer.register_to_config(atlas_version='region-v2',atlas_tile_size=64)
    importlib.import_module('src.train.model').tranformer_forward=call_module
    importlib.import_module('src.flux.generate_fill').tranformer_forward=call_module
    native=importlib.import_module('src.flux.transformer');native.block_forward=call_module;native.single_block_forward=call_module
    return transformer


class AtlasTransformer(FluxTransformer2DModel):
    @register_to_config
    def __init__(self,patch_size=1,in_channels=384,out_channels=64,num_layers=19,num_single_layers=38,attention_head_dim=128,num_attention_heads=24,joint_attention_dim=4096,pooled_projection_dim=768,guidance_embeds=True,axes_dims_rope=(16,56,56),atlas_version='region-v2',atlas_tile_size=64):
        assert atlas_version=='region-v2' and atlas_tile_size==64
        super().__init__(patch_size=patch_size,in_channels=in_channels,out_channels=out_channels,num_layers=num_layers,num_single_layers=num_single_layers,attention_head_dim=attention_head_dim,num_attention_heads=num_attention_heads,joint_attention_dim=joint_attention_dim,pooled_projection_dim=pooled_projection_dim,guidance_embeds=guidance_embeds,axes_dims_rope=axes_dims_rope)
        install(self)


class AtlasTrainingModel(FullDiTModel):
    def step(self,batch):
        if not hasattr(self,'atlas_encoder'):self.atlas_encoder=AtlasEncoder(self.flux_pipe)
        assert batch['image'].shape[0]==1
        count=int(batch['n_lines'][0]);texts=[v[0] for v in batch['texts'][:count]]
        visible=bool(batch['condition'].any()) and getattr(self,'atlas_condition_enabled',True)
        with torch.no_grad():
            atlas=self.atlas_encoder.encode(texts,batch['positions'],visible,getattr(self,'atlas_policy','character'),trace=batch.get('global_glyph_trace',[None])[0]) if hasattr(self.atlas_encoder,'dynamic_cache') else self.atlas_encoder.encode(texts,batch['positions'],visible,getattr(self,'atlas_policy','character'))
            if atlas is not None:atlas['routes']=make_routes(atlas['regions'],atlas['owners'])
        old_cfg=self.model_config;self.model_config=dict(old_cfg,atlas=atlas)
        self.atlas_diagnostics=dict(glyphs=(atlas.get('unit_count') or atlas['glyph_count']) if atlas else 0,tokens=atlas['features'].shape[1] if atlas else 0,routes=len(atlas['routes']) if atlas else 0,dropout=not visible)
        try:return super().step(batch)
        finally:self.model_config=old_cfg


def shard(model,rank,layout=None):
    transformer=install(model.transformer);transformer.requires_grad_(True)
    counts={'base':0,'atlas':0}
    for name,p in transformer.named_parameters():
        assert 'lora_' not in name
        counts['atlas' if name.startswith('atlas_') else 'base']+=p.numel()
    assert counts=={'base':11902391360,'atlas':593536},counts
    transformer.float()
    # Read-only initialization probes for zero-update parity diagnostics.
    import hashlib,json
    probes={};atlas_hash=hashlib.sha256()
    for name,p in transformer.named_parameters():
        flat=p.detach().reshape(-1);probes[name]=[list(p.shape),flat[[0,flat.numel()//2,flat.numel()-1]].float().cpu().tolist()]
        if name.startswith('atlas_'):atlas_hash.update(name.encode());atlas_hash.update(flat.cpu().numpy().tobytes())
    model.initial_parameter_probe_sha256=hashlib.sha256(json.dumps(probes,sort_keys=True).encode()).hexdigest()
    model.initial_atlas_sha256=atlas_hash.hexdigest()
    wrapped=FSDP(transformer,device_id=rank,process_group=layout['shard_group'] if layout else None,auto_wrap_policy=ModuleWrapPolicy({FluxTransformerBlock,FluxSingleTransformerBlock}),sharding_strategy=ShardingStrategy.FULL_SHARD,mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16),use_orig_params=True,sync_module_states=True,limit_all_gathers=True)
    if layout:
        from distributed_layout import attach_layout,synchronize_initial_replicas
        attach_layout(wrapped,layout);synchronize_initial_replicas(wrapped)
    model.transformer=model.flux_pipe.transformer=wrapped;model._device=torch.device('cuda',rank);model._dtype=torch.bfloat16
    for module in [model.flux_pipe.vae,model.flux_pipe.text_encoder,model.flux_pipe.text_encoder_2]:module.to(model.device,dtype=torch.bfloat16).requires_grad_(False).eval()
    model.odm_loss.ResNet1.to(model.device).requires_grad_(False).eval()
    return counts
