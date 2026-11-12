"""Stage1 OCR-only supervision: native time, Flow, Mask, reconstruction unchanged."""
import ast,hashlib,inspect,textwrap
import torch
from glyph_atlas.supervision import Teacher

def ocr_time_multiplier(t, kind):
    if kind=='none':return torch.ones_like(t.float()).mean()
    if kind=='high_normalized_native':
        assert t.numel()==1, 'C4 weighting is per sample; this recipe requires microbatch1'
        return 2*t.detach().float().mean()
    raise ValueError(kind)

class OCRStage1:
    def __init__(self,device,weight,time_weight='none'):
        assert weight in (.25,.05)
        assert time_weight in ('none','high_normalized_native')
        self.time_weight=time_weight;self.time_multiplier=None
        self.teacher=Teacher(device,'valid_crops_only');self.weight=weight;self.last_losses={};self.diagnostics={}
    def loss(self,pred,gt,mask,batch,t):
        ctc,raw,diag=self.teacher.components(pred,gt,batch)
        assert float(ctc.detach())==0 and torch.isfinite(raw)
        self.time_multiplier=ocr_time_multiplier(t,self.time_weight)
        weighted=raw*self.weight*self.time_multiplier;zero=pred.sum()*0
        self.last_losses=dict(loss_odm=zero,loss_ctc=zero,loss_ocr_feature_raw=raw,loss_ocr_feature=weighted)
        self.diagnostics=diag
        return weighted
    def clear(self):self.last_losses={};self.diagnostics={}

def install():
    from src.train import model as native
    original=native.OminiModelFIll.step
    assert not getattr(original,'_stage1_ocr_only',False) and not getattr(original,'_training_shift',False)
    source=textwrap.dedent(inspect.getsource(original));changed=source
    replacements=[('if self.odm_loss is not None:', 'if self.supervision is not None:'),
                  ('odm_loss = self.odm_loss.loss(image_pred, imgs, mask_image)', 'odm_loss = self.supervision.loss(image_pred, imgs, mask_image, batch, t)'),
                  ("res['loss_odm'] = odm_loss", "res.update(self.supervision.last_losses)"),
                  ('t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))','t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))\n        self._stage1_t = t.detach().clone()')]
    for old,new in replacements:
        assert changed.count(old)==1,old
        changed=changed.replace(old,new)
    reverse=changed
    for old,new in reversed(replacements):reverse=reverse.replace(new,old)
    assert reverse==source
    ns={};exec(compile(ast.parse(changed),str(__file__)+':native_stage1_ocr','exec'),native.__dict__,ns)
    patched=ns['step'];patched._stage1_ocr_only=True;native.OminiModelFIll.step=patched
    return dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),patched_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                replacements=replacements,time_sampling_unchanged=True,flow_mask_reconstruction_unchanged=True,no_vendor_file_edit=True)
