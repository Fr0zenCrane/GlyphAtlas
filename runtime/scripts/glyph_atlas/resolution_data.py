"""Shared random-resolution training and four deterministic inference policies.

Retains the source C4 surface tokenizer. Random choices use a private per-step
generator; native torch/Python/numpy noise, time and data RNG are untouched.
"""
import json, math, random, unicodedata
from collections import Counter, OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from glyph_atlas.data import AtlasEncoder, char_word_units, tile_for, line_strip_tiles, FONT
from glyph_atlas.budget import GlyphUnit, InkExtent, Allocation, BudgetPlan, BudgetTooSmall, plan_atlas, render_unit

MODES=('random','fixed48','fixed64','fixed80','dynamic')

def punctuation(text):
    base=[c for c in text if not unicodedata.category(c).startswith('M')]
    return bool(base) and all(unicodedata.category(c).startswith('P') for c in base)

def allocate(units,mode,step=None,rank=0,seed=42):
    if mode not in MODES:raise ValueError(mode)
    if mode=='dynamic':
        protected=[u.key for u in units if punctuation(u.text)]
        try:return plan_atlas(units,buckets=(256,512),fixed64_keys=protected),False
        except BudgetTooSmall as err:
            return plan_atlas(units,budget=16*math.ceil(err.minimum/16),fixed64_keys=protected),True
    if mode=='random':
        if step is None or step<1 or not 0<=rank<8:raise ValueError('Random encoder requires optimizer step and rank')
        rng=random.Random((seed<<32)+step*8+rank)
    heights=[];why=[]
    for u in units:
        if mode=='random':
            if punctuation(u.text):h=64;reason='punctuation_fixed64'
            else:
                x=rng.random();h=48 if x<.2 else 64 if x<.9 else 80;reason='random_resolution'
        else:h=int(mode.removeprefix('fixed'));reason=mode
        heights.append(h);why.append(reason)
    allocations=tuple(Allocation(u.key,u.owner,h,u.shape(h)[0],u.cost(h),u.global_min_px,reason) for u,h,reason in zip(units,heights,why))
    total=sum(a.tokens for a in allocations)
    return BudgetPlan(total,sum(u.cost(64) for u in units),total,allocations,(),()),False

class ResolutionAtlasEncoder(AtlasEncoder):
    def __init__(self,pipe,mode='random',seed=42):
        super().__init__(pipe)
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.seed=seed;self.step=None;self.rank=0
        self.dynamic_cache=OrderedDict();self.last_plan=None

    def set_context(self,step,rank):self.step=int(step);self.rank=int(rank)

    @torch.no_grad()
    def encode(self,texts,positions,visible=True,policy='zh_char_en_word_v2',trace=None):
        if not visible:
            self.last_plan=None;return None
        assert policy=='zh_char_en_word_v2'
        if isinstance(trace,str):trace=json.loads(trace)
        if self.mode=='dynamic' and (trace is None or len(trace)!=len(texts)):raise ValueError('Dynamic policy needs measured global glyph trace')
        units=[];orders=[];lengths=[];render_modes=[]
        for owner,text in enumerate(texts):
            record=trace[owner] if trace else None
            if record:assert record['text']==text and len(record['ink'])==len(text)
            offset=0;pieces=char_word_units(text)
            for order,(surface,render_mode) in enumerate(pieces):
                measures=record['ink'][offset:offset+len(surface)] if record else [[0,0]]*len(surface);offset+=len(surface)
                if surface.isspace():continue
                width=64 if render_mode=='char' else len(line_strip_tiles(surface))*64
                from font_support import font_path
                units.append(GlyphUnit(f'{owner}:{order}',surface,owner,width,tuple(InkExtent(*v) for c,v in zip(surface,measures) if not c.isspace()),font_path(surface)))
                orders.append(order);lengths.append(len(pieces));render_modes.append(render_mode)
        if not units:self.last_plan=None;return None
        plan,overflow=allocate(units,self.mode,self.step,self.rank,self.seed)
        self.last_plan=dict(mode=self.mode,seed=self.seed,optimizer_step=self.step,rank=self.rank,
                            budget=plan.budget,tokens=plan.used_tokens,baseline_tokens=plan.baseline_tokens,
                            overflow=overflow,heights=dict(Counter(a.height for a in plan.allocations)),
                            ordinary_heights=dict(Counter(a.height for u,a in zip(units,plan.allocations) if not punctuation(u.text))),
                            punctuation_units=sum(punctuation(u.text) for u in units),
                            punctuation_heights=dict(Counter(a.height for u,a in zip(units,plan.allocations) if punctuation(u.text))),
                            unit_heights=[a.height for a in plan.allocations],allocation_digest=__import__('hashlib').sha256(json.dumps([(u.key,u.text,a.height) for u,a in zip(units,plan.allocations)],ensure_ascii=False).encode()).hexdigest(),
                            policy='random_resolution_v1_source_tokenizer_punct64')
        if all(a.height==64 for a in plan.allocations):
            out=super().encode(texts,positions,visible,policy)
            out.update(plan=self.last_plan,unit_count=len(units),policy=self.last_plan['policy'])
            return out
        device=self.pipe.device;dtype=self.pipe.dtype;features=[];owners=[];ids=[];geometry=[];offsets=Counter()
        for u,a,order,length,render_mode in zip(units,plan.allocations,orders,lengths,render_modes):
            if a.height==64:
                arrays=[tile_for(u.text)] if render_mode=='char' else line_strip_tiles(u.text)
            else:arrays=[render_unit(u,a)]
            key=(u.text,render_mode,u.font_path,u.width64,a.height,a.width,'resolution_aug_v1')
            if key not in self.dynamic_cache:
                fs=[]
                for tile_idx,arr in enumerate(arrays):
                    legacy_key=(render_mode,u.text,tile_idx)
                    if a.height==64 and legacy_key in self.cache:fs.append(self.cache[legacy_key]);continue
                    tensor=torch.from_numpy(np.asarray(arr).copy()).to(device=device,dtype=dtype)[None,None].expand(-1,3,-1,-1)/127.5-1
                    z=self.pipe.vae.encode(tensor).latent_dist.mean;z=(z-self.pipe.vae.config.shift_factor)*self.pipe.vae.config.scaling_factor
                    f=self.pipe._pack_latents(z,*z.shape).squeeze(0).cpu();fs.append(f)
                    if a.height==64:self.cache[legacy_key]=f
                self.dynamic_cache[key]=torch.cat(fs);self.dynamic_cache.move_to_end(key)
                while len(self.dynamic_cache)>8192:self.dynamic_cache.popitem(last=False)
            f=self.dynamic_cache[key];assert f.shape==(a.tokens,64)
            features.append(f.to(device=device,dtype=dtype));owners.extend([u.owner]*a.tokens)
            gh,gw=a.height//16,a.width//16
            grid=[(y,x) for tile in range(gw//4) for y in range(4) for x in range(tile*4,tile*4+4)] if a.height==64 else [(y,x) for y in range(gh) for x in range(gw)]
            assert len(grid)==a.tokens
            for y,x in grid:
                ids.append([1,u.owner*4+4*y/gh,offsets[u.owner]+x])
                geometry.append([(x+.5)/gw,(y+.5)/gh,order/max(1,length),length/20])
            offsets[u.owner]+=gw
        if isinstance(positions,(tuple,list)):masks=torch.stack([torch.as_tensor(p,device=device).squeeze() for p in positions[:len(texts)]])
        else:masks=positions.to(device).reshape(len(texts),512,512)
        regions=F.avg_pool2d(masks[:,None].float(),16,16).flatten(1).T>0
        return dict(features=torch.cat(features)[None],owners=torch.tensor(owners,device=device),ids=torch.tensor(ids,device=device,dtype=torch.float32),geometry=torch.tensor(geometry,device=device,dtype=torch.float32)[None].to(dtype),regions=regions,glyph_count=sum(u.width64//64 for u in units),unit_count=len(units),policy=self.last_plan['policy'],plan=self.last_plan)
