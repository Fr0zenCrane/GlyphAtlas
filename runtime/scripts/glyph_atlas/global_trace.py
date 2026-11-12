"""Trace the unchanged native global-glyph raster, with renderer-advance ownership.

No polygon/character-count size proxy. Native glyph pixels are unchanged. For
LTR nonjoining scripts, final ink pixels are assigned to character cells from
actual font advances (cell ownership can be approximate at touching glyphs).
Joining/RTL/combining strings are explicitly unresolved and get conservative
zero extents, never a false 'large/clear' classification.
"""
import ast, contextlib, inspect, json, textwrap, threading, unicodedata
import numpy as np
from PIL import Image
from glyph_atlas.budget import measure_global_ink
_STATE=threading.local()

@contextlib.contextmanager
def capture():
    previous=getattr(_STATE,'records',None);records=[];_STATE.records=records
    try:yield records
    finally:_STATE.records=previous

def _record(state):
    records=getattr(_STATE,'records',None)
    if records is None:return
    original=state['_atlas_original_text'];rendered=state['text'];font=state['new_font']
    result=state['img'];base=dict(text=original,rendered_text=rendered,canvas=list(result.shape[:2]),
        font_path=str(font.path),font_size=font.size,angle=float(state['angle']))
    unsupported=any(unicodedata.combining(c) or unicodedata.bidirectional(c) in ('R','AL','AN') or
        any(x in unicodedata.name(c,'') for x in ('DEVANAGARI','BENGALI','THAI','TAMIL','TELUGU','MEETEI')) for c in original)
    if unsupported:
        base.update(method='unresolved_shaping_conservative',ink=[[0.,0.] for c in original]);records.append(base);return
    if ''.join(original.split())!=''.join(rendered.split()):raise ValueError('Global trace text correspondence changed')
    h,w=result.shape[:2]
    if (h,w)!=(512,512):raise ValueError('Trace requires final512 global canvas; apply actual final transform first')
    cells=np.zeros((h,w),np.int32);nonspace=[]
    if not state['vert']:
        x0=state['rect'][0][0]-state['text_width']//2
        boundaries=[0]+[int(round(x0+font.getlength(rendered[:i]))) for i in range(1,len(rendered))]+[w]
        for i,c in enumerate(rendered):
            lo,hi=sorted((max(0,min(w,boundaries[i])),max(0,min(w,boundaries[i+1]))))
            cells[:,lo:hi]=i+1
            if not c.isspace():nonspace.append(i+1)
    else:
        y=float(min(state['box'][:,1]));bounds=[int(round(y))]
        for c in rendered:
            _,_,_,bottom=font.getbbox(c);y+=bottom;bounds.append(int(round(y)))
        bounds[0]=0;bounds[-1]=h
        for i,c in enumerate(rendered):
            lo,hi=max(0,min(h,bounds[i])),max(0,min(h,bounds[i+1]));cells[lo:hi,:]=i+1
            if not c.isspace():nonspace.append(i+1)
    layer=Image.fromarray(cells,mode='I').rotate(state['angle'],expand=1,center=tuple(state['rect'][0]),resample=Image.Resampling.NEAREST)
    assigned=Image.new('I',(w,h));assigned.paste(layer,(state['x_offset'],state['y_offset']));assigned=np.asarray(assigned)
    ink=np.asarray(result).squeeze()>0
    measured=[]
    for label in nonspace:
        mask=ink&(assigned==label)
        e=measure_global_ink([mask],baseline_degrees=-state['angle'])[0]
        measured.append([e.width,e.height])
    it=iter(measured);extents=[[0.,0.] if c.isspace() else next(it) for c in original]
    assert next(it,None) is None
    base.update(method='native_raster_font_advance_cells',ink=extents,
                assigned_ink_pixels=int(np.count_nonzero(ink&(assigned>0))),total_ink_pixels=int(ink.sum()))
    records.append(base)

def install_renderer(module):
    fn=module.draw_glyph2
    if getattr(fn,'_atlas_traced',False):return
    fallback=getattr(fn,'_glyph_font_fallback',False)
    fn=inspect.unwrap(fn)
    tree=ast.parse(textwrap.dedent(inspect.getsource(fn)));f=tree.body[0]
    f.body.insert(0,ast.parse('_atlas_original_text = text').body[0])
    if fallback:f.body.insert(0,ast.parse('font = _atlas_font(font,text)').body[0])
    class Inject(ast.NodeTransformer):
        def visit_Return(self,node):
            return [ast.parse('_atlas_record(locals())').body[0],node]
    tree=Inject().visit(tree);ast.fix_missing_locations(tree)
    from font_support import font_for_text
    ns=dict(fn.__globals__,_atlas_record=_record,_atlas_font=font_for_text);exec(compile(tree,'<native_glyph_trace>','exec'),ns)
    wrapped=ns[fn.__name__];wrapped._atlas_traced=True;module.draw_glyph2=wrapped

class TraceDatasetMixin:
    def __getitem__(self,item):
        with capture() as records:sample=super().__getitem__(item)
        n=int(sample['n_lines']);assert len(records)==n,(len(records),n)
        assert [r['text'] for r in records]==sample['texts'][:n]
        sample['global_glyph_trace']=json.dumps(records,ensure_ascii=False)
        return sample
