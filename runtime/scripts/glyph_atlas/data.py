import os
"""Atlas content only; existing text/polygon pairs provide region ownership."""
from functools import lru_cache
from pathlib import Path
import unicodedata
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from font_support import font_path
import torch
import torch.nn.functional as F

ROOT=Path(os.environ["GLYPH_OMNI_ROOT"])
FONT=ROOT/'vendor/FluxText/font/Arial_Unicode.ttf'


def clusters(text):
    # Combining marks stay with their preceding character. Current task is
    # Chinese/Latin; original global glyph rendering still handles full lines.
    out=[]
    for ch in text:
        if unicodedata.combining(ch) and out:out[-1]+=ch
        else:out.append(ch)
    return out


def is_han(cluster):
    return any('\u3400'<=ch<='\u9fff' for ch in cluster)


def is_latin_or_digit(cluster):
    for ch in cluster:
        if ch.isdigit() or unicodedata.combining(ch):continue
        if 'LATIN' not in unicodedata.name(ch,''):return False
    return bool(cluster)


def char_word_units(text):
    """Deterministic surface units: Han characters and Latin/digit words.

    Apostrophes and hyphens remain inside a word only when surrounded by
    Latin/digit clusters. Whitespace is retained for exact reconstruction but
    is not encoded. Other scripts and punctuation remain individual clusters.
    """
    source=clusters(text);out=[];i=0;connectors={"'",'’','-'}
    while i<len(source):
        current=source[i]
        if current.isspace():out.append((current,'space'));i+=1;continue
        if is_han(current):out.append((current,'char'));i+=1;continue
        if is_latin_or_digit(current):
            pieces=[current];i+=1
            while i<len(source):
                if is_latin_or_digit(source[i]):pieces.append(source[i]);i+=1;continue
                if source[i] in connectors and i+1<len(source) and is_latin_or_digit(source[i+1]):
                    pieces.extend((source[i],source[i+1]));i+=2;continue
                break
            out.append((''.join(pieces),'word'));continue
        out.append((current,'char'));i+=1
    assert ''.join(surface for surface,_ in out)==text
    return out

def strip_tiles(text):
    chars=clusters(text)
    if len(chars)==1:return [tile_for(text)]
    if all('\u3400'<=c<='\u9fff' for c in chars):return [tile_for(c) for c in chars]
    base=ImageFont.truetype(font_path(text),56);left,top,right,bottom=base.getbbox(text)
    size=max(1,int(56*56/max(1,bottom-top)));font=base.font_variant(size=size);left,top,right,bottom=font.getbbox(text)
    while bottom-top>56:
        size-=1;font=base.font_variant(size=size);left,top,right,bottom=font.getbbox(text)
    width=max(64,64*int(np.ceil((right-left+8)/64)));image=Image.new('L',(width,64),0)
    ImageDraw.Draw(image).text(((width-(right-left))/2-left,(64-(bottom-top))/2-top),text,font=font,fill=255)
    return [np.asarray(image.crop((x,0,x+64,64))).copy() for x in range(0,width,64)]


def line_strip_tiles(text):
    """Render one complete text line, regardless of language, then tile it.

    Tiling is only a storage/VAE packing detail. All returned tiles keep the
    same line owner and are never treated as character units.
    """
    base=ImageFont.truetype(font_path(text),56);left,top,right,bottom=base.getbbox(text)
    size=max(1,int(56*56/max(1,bottom-top)));font=base.font_variant(size=size);left,top,right,bottom=font.getbbox(text)
    while bottom-top>56:
        size-=1;font=base.font_variant(size=size);left,top,right,bottom=font.getbbox(text)
    width=max(64,64*int(np.ceil((right-left+8)/64)));image=Image.new('L',(width,64),0)
    ImageDraw.Draw(image).text(((width-(right-left))/2-left,(64-(bottom-top))/2-top),text,font=font,fill=255)
    return [np.asarray(image.crop((x,0,x+64,64))).copy() for x in range(0,width,64)]

def tile_for(text):
    base=ImageFont.truetype(font_path(text),56)
    left,top,right,bottom=base.getbbox(text)
    ratio=min(56/max(1,right-left),56/max(1,bottom-top))
    font=base.font_variant(size=max(1,int(56*ratio)))
    left,top,right,bottom=font.getbbox(text)
    image=Image.new('L',(64,64),0)
    ImageDraw.Draw(image).text(((64-(right-left))/2-left,(64-(bottom-top))/2-top),text,font=font,fill=255)
    return np.asarray(image).copy()


def layout(texts):
    entries=[]
    for owner,text in enumerate(texts):
        group=clusters(text)
        for order,cluster in enumerate(group):
            if cluster.isspace():continue
            entries.append(dict(owner=owner,order=order,length=len(group),text=cluster))
    return entries


class AtlasEncoder:
    def __init__(self,pipe):self.pipe=pipe;self.cache={}

    @torch.no_grad()
    def encode(self,texts,positions,visible=True,policy='character'):
        entries=[]
        for owner,text in enumerate(texts):
            if policy=='zh_line_en_line':units=[(text,'line')]
            elif policy=='zh_char_en_word_v2':units=char_word_units(text)
            elif policy=='character':units=[(part,'char') for part in clusters(text)]
            else:raise ValueError(f'Unsupported active atlas policy: {policy}')
            line_token_offset=0
            for order,(surface,render_mode) in enumerate(units):
                if surface.isspace():continue
                tiles=tile_for(surface)[None] if render_mode=='char' else line_strip_tiles(surface)
                for tile_idx,_ in enumerate(tiles):
                    entries.append(dict(owner=owner,order=order,length=len(units),surface=surface,tile_idx=tile_idx,tile_count=len(tiles),render_mode=render_mode,line_token_offset=line_token_offset))
                line_token_offset+=len(tiles)*4
        device=self.pipe.device;dtype=self.pipe.dtype
        if not visible or not entries:return None
        needed=[]
        for e in entries:
            key=(e['render_mode'],e['surface'],e['tile_idx'])
            if key not in self.cache:needed.append(key)
        for start in range(0,len(needed),16):
            keys=needed[start:start+16]
            arrays=np.stack([(tile_for(k[1])[None] if k[0]=='char' else line_strip_tiles(k[1]))[k[2]] for k in keys])
            tensor=torch.from_numpy(arrays).to(device=device,dtype=dtype)[:,None].expand(-1,3,-1,-1)/127.5-1
            z=self.pipe.vae.encode(tensor).latent_dist.mean
            z=(z-self.pipe.vae.config.shift_factor)*self.pipe.vae.config.scaling_factor
            tokens=self.pipe._pack_latents(z,*z.shape).detach()
            for key,v in zip(keys,tokens):self.cache[key]=v.cpu()
        features=torch.stack([self.cache[(e['render_mode'],e['surface'],e['tile_idx'])] for e in entries]).to(device=device,dtype=dtype).reshape(1,-1,64)
        owners=torch.tensor([e['owner'] for e in entries],device=device).repeat_interleave(16)
        ids=[];geometry=[]
        for e in entries:
            for y in range(4):
                for x in range(4):
                    id_x=e['line_token_offset']+e['tile_idx']*4+x
                    ids.append([1,e['owner']*4+y,id_x])
                    if e['render_mode'] in ('line','word'):
                        u=(e['tile_idx']*4+x+.5)/max(1,e['tile_count']*4)
                        geometry.append([u,(y+.5)/4,e['order']/max(1,e['length']),e['length']/20])
                    else:
                        geometry.append([(x+.5)/4,(y+.5)/4,e['order']/max(1,e['length']),e['length']/20])
        if isinstance(positions,(tuple,list)):
            masks=torch.stack([torch.as_tensor(p,device=device).squeeze() for p in positions[:len(texts)]])
        else:masks=positions.to(device).reshape(len(texts),512,512)
        region_map=F.avg_pool2d(masks[:,None].float(),16,16).flatten(1).T>0
        assert region_map.shape==(1024,len(texts))
        units=len({(e['owner'],e['order']) for e in entries})
        return dict(features=features,owners=owners,ids=torch.tensor(ids,device=device,dtype=torch.float32),geometry=torch.tensor(geometry,device=device,dtype=dtype)[None],regions=region_map,glyph_count=len(entries),unit_count=units,entries=entries,cache_entries=len(self.cache),policy=policy)



def make_routes(regions,owners,text_tokens=512,scene_tokens=1024):
    """Query partitions implementing exact block attention, without dense L² masks."""
    device=owners.device;K=regions.shape[1];S=scene_tokens;T=text_tokens;base=T+2*S
    assert regions.shape==(S,K) and K<=8
    cpu_r=regions.detach().cpu().numpy();cpu_owner=owners.detach().cpu().numpy()
    signatures=(cpu_r.astype(np.int64)*(1<<np.arange(K))[None]).sum(1)
    base_keys=np.arange(base);routes=[]
    for sig in sorted(set(signatures.tolist())|{0}):
        where=np.flatnonzero(signatures==sig)
        query=np.r_[np.arange(T) if sig==0 else np.empty(0,np.int64),T+where,T+S+where]
        if not len(query):continue
        chosen=np.flatnonzero((sig & (1<<cpu_owner))!=0)
        keys=np.r_[base_keys,base+chosen]
        routes.append((torch.as_tensor(query,device=device,dtype=torch.long),torch.as_tensor(keys,device=device,dtype=torch.long)))
    for i in sorted(set(cpu_owner.tolist())):
        query=base+np.flatnonzero(cpu_owner==i);local=np.flatnonzero(cpu_r[:,i])
        keys=np.r_[np.arange(T),T+local,T+S+local,query]
        routes.append((torch.as_tensor(query,device=device,dtype=torch.long),torch.as_tensor(keys,device=device,dtype=torch.long)))
    return routes
