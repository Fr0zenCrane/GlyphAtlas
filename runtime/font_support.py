"""Deterministic whole-string fallback; never replace or drop a GT character."""
from pathlib import Path
from functools import lru_cache,wraps
import unicodedata
import json,hashlib
from PIL import ImageFont
from fontTools.ttLib import TTFont
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'vendor/FluxText/font/Arial_Unicode.ttf'
NAMES=['DejaVuSans.ttf','NotoSansBengali[wdth,wght].ttf','NotoSansMeeteiMayek[wght].ttf',
       'NotoSansSymbols2-Regular.ttf','NotoSansMath-Regular.ttf','NotoSansSymbols[wght].ttf','FreeSerif.ttf']
@lru_cache(None)
def cmap(path):
    p=Path(path)
    if p.parent==ROOT/'assets/fonts':
        entries=json.loads((ROOT/'provenance/fallback_fonts.json').read_text())['files']
        expected=next(x['sha256'] for x in entries if x['file']==str(p.relative_to(ROOT)))
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Font asset hash mismatch: '+p.name)
    with TTFont(path) as font:return set(font.getBestCmap())
@lru_cache(65536)
def font_path(text):
    required={ord(c) for c in text if not c.isspace() and unicodedata.category(c)[0]!='C' and not 0xfe00<=ord(c)<=0xfe0f}
    for path in [BASE]+[ROOT/'assets/fonts'/n for n in NAMES]:
        if required<=cmap(str(path)):return str(path)
    raise ValueError('No shipped font covers the full GT string: '+repr(text))
def font_for_text(font,text):
    path=font_path(text)
    return font if str(font.path)==path else ImageFont.truetype(path,font.size)
def install_fallback(module):
    for name in ['draw_glyph','draw_glyph2']:
        original=getattr(module,name)
        if getattr(original,'_glyph_font_fallback',False):continue
        def wrap(fn):
            @wraps(fn)
            def render(font,text,*args,**kwargs):return fn(font_for_text(font,text),text,*args,**kwargs)
            render._glyph_font_fallback=True
            return render
        setattr(module,name,wrap(original))
