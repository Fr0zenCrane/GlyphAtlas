import os
"""Frozen character-Atlas encoder used by the original character-Atlas run."""
from functools import lru_cache
from pathlib import Path
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


ROOT = Path(os.environ["GLYPH_OMNI_ROOT"])
FONT = ROOT / "vendor/FluxText/font/Arial_Unicode.ttf"


def clusters(text):
    out = []
    for ch in text:
        if unicodedata.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


@lru_cache(maxsize=8192)
def tile_for(text):
    base = ImageFont.truetype(str(FONT), 56)
    left, top, right, bottom = base.getbbox(text)
    ratio = min(56 / max(1, right - left), 56 / max(1, bottom - top))
    font = base.font_variant(size=max(1, int(56 * ratio)))
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(image).text(((64 - (right - left)) / 2 - left, (64 - (bottom - top)) / 2 - top), text, font=font, fill=255)
    return np.asarray(image).copy()


def layout(texts):
    entries = []
    for owner, text in enumerate(texts):
        group = clusters(text)
        for order, cluster in enumerate(group):
            if cluster.isspace():
                continue
            entries.append(dict(owner=owner, order=order, length=len(group), text=cluster))
    return entries


class AtlasEncoder:
    def __init__(self, pipe):
        self.pipe = pipe
        self.cache = {}

    @torch.no_grad()
    def encode(self, texts, positions, visible=True):
        entries = layout(texts)
        device = self.pipe.device
        dtype = self.pipe.dtype
        if not visible or not entries:
            return None
        needed = list(dict.fromkeys(e["text"] for e in entries if e["text"] not in self.cache))
        for start in range(0, len(needed), 16):
            words = needed[start:start + 16]
            arrays = np.stack([tile_for(word) for word in words])
            tensor = torch.from_numpy(arrays).to(device=device, dtype=dtype)[:, None].expand(-1, 3, -1, -1) / 127.5 - 1
            latent = self.pipe.vae.encode(tensor).latent_dist.mean
            latent = (latent - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            tokens = self.pipe._pack_latents(latent, *latent.shape).detach()
            for word, value in zip(words, tokens):
                self.cache[word] = value.cpu()
        features = torch.stack([self.cache[e["text"]] for e in entries]).to(device=device, dtype=dtype).reshape(1, -1, 64)
        owners = torch.tensor([e["owner"] for e in entries], device=device).repeat_interleave(16)
        ids = []
        geometry = []
        for entry in entries:
            for y in range(4):
                for x in range(4):
                    ids.append([1, entry["owner"] * 4 + y, entry["order"] * 4 + x])
                    geometry.append([(x + .5) / 4, (y + .5) / 4, entry["order"] / 20, entry["length"] / 20])
        if isinstance(positions, (tuple, list)):
            masks = torch.stack([torch.as_tensor(p, device=device).squeeze() for p in positions[:len(texts)]])
        else:
            masks = positions.to(device).reshape(len(texts), 512, 512)
        region_map = F.avg_pool2d(masks[:, None].float(), 16, 16).flatten(1).T > 0
        assert region_map.shape == (1024, len(texts))
        return dict(
            features=features, owners=owners, ids=torch.tensor(ids, device=device, dtype=torch.float32),
            geometry=torch.tensor(geometry, device=device, dtype=dtype)[None], regions=region_map,
            glyph_count=len(entries), entries=entries, cache_entries=len(self.cache),
        )


def make_routes(regions, owners, text_tokens=512, scene_tokens=1024):
    device = owners.device
    count = regions.shape[1]
    scene = scene_tokens
    text = text_tokens
    base = text + 2 * scene
    assert regions.shape == (scene, count) and count <= 8
    cpu_regions = regions.detach().cpu().numpy()
    cpu_owner = owners.detach().cpu().numpy()
    signatures = (cpu_regions.astype(np.int64) * (1 << np.arange(count))[None]).sum(1)
    base_keys = np.arange(base)
    routes = []
    for signature in sorted(set(signatures.tolist()) | {0}):
        where = np.flatnonzero(signatures == signature)
        query = np.r_[np.arange(text) if signature == 0 else np.empty(0, np.int64), text + where, text + scene + where]
        if not len(query):
            continue
        chosen = np.flatnonzero((signature & (1 << cpu_owner)) != 0)
        keys = np.r_[base_keys, base + chosen]
        routes.append((torch.as_tensor(query, device=device, dtype=torch.long), torch.as_tensor(keys, device=device, dtype=torch.long)))
    for owner in sorted(set(cpu_owner.tolist())):
        query = base + np.flatnonzero(cpu_owner == owner)
        local = np.flatnonzero(cpu_regions[:, owner])
        keys = np.r_[np.arange(text), text + local, text + scene + local, query]
        routes.append((torch.as_tensor(query, device=device, dtype=torch.long), torch.as_tensor(keys, device=device, dtype=torch.long)))
    return routes
