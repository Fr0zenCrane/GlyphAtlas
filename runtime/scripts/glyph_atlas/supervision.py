"""Isolated, frozen-teacher OCR / spatial / noise-time supervision.

No trainable parameters, no sampling changes, and no mutation of the native
Flow/Mask implementation. Coefficients are calibrated on training data only.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(os.environ['GLYPH_OMNI_ROOT'])
RUN=Path(os.environ.get('GLYPH_RUN_DIR',str(ROOT/'runs/default')))
VENDOR=ROOT/'vendor/FluxText'

def position_mask(p):
    p=p.float()
    if p.ndim==3:p=p.unsqueeze(0)
    if p.shape[-1]==1:p=p.permute(0,3,1,2)
    assert p.ndim==4 and p.shape[:2]==(1,1),p.shape
    return p

def time_weight(t, kind):
    if kind=='none':return torch.ones_like(t)
    if kind=='low':return 1-t
    if kind=='low_squared':return (1-t).square()
    if kind=='tail':return torch.where(t<=.8,torch.ones_like(t),torch.full_like(t,.25))
    if kind=='high':return t
    raise ValueError(kind)

def normalized_error(error,mask):
    """Linear occupancy weighting; channel-mean convention unchanged."""
    area=mask.sum()
    return (error*mask).sum()/(error.shape[1]*area.clamp_min(1e-12))

class Teacher:
    def __init__(self,device,gate_policy='legacy'):
        assert gate_policy in ('legacy','valid_crops_only')
        self.gate_policy=gate_policy
        sys.path.insert(0,str(VENDOR/'eval'))
        from recognizer import create_predictor,TextRecognizer,crop_image
        from easydict import EasyDict
        with torch.random.fork_rng(devices=[device.index]):
            predictor=create_predictor(str(VENDOR/'eval/ocr_weights/ppv3_rec.pth')).float().eval().requires_grad_(False).to(device)
        self.rec=TextRecognizer(EasyDict(rec_image_shape='3, 48, 320',rec_batch_num=8,rec_char_dict_path=str(VENDOR/'eval/ocr_weights/ppocr_keys_v1.txt'),use_fp16=False),predictor)
        self.crop=crop_image;self.device=device
        self.cache=RUN/'data_audit/teacher_gates';self.cache.mkdir(parents=True,exist_ok=True)
    @staticmethod
    def norm(s):return ''.join(c for c in s.casefold() if c.isalnum())
    def components(self,pred,gt,batch):
        count=int(batch['n_lines'][0]);zero=pred.sum()*0
        if not count:return zero,zero,dict(ocr_lines=0,ocr_eligible=0)
        pc=[];gc=[];texts=[];keys=[]
        for i in range(count):
            mask=(position_mask(batch['positions'][i])[0,0].detach().cpu().numpy()*255).astype(np.uint8)
            if self.gate_policy=='valid_crops_only' and not mask.any():continue
            text=batch['texts'][i][0]
            a=self.crop(((pred[0].float()+1)*127.5).clamp(0,255),mask)
            b=self.crop(((gt[0].float()+1)*127.5).clamp(0,255),mask)
            if min(a.shape[-2:])<1 or min(b.shape[-2:])<1:continue
            pc.append(a);gc.append(b);texts.append(text)
            h=hashlib.sha256();h.update(text.encode());h.update(mask.tobytes());h.update(b.detach().round().byte().cpu().numpy().tobytes());keys.append(h.hexdigest())
        if not pc:return zero,zero,dict(ocr_lines=count,ocr_eligible=0)
        with torch.autocast('cuda',enabled=False):
            with torch.no_grad():gl,gf=self.rec.pred_imglist(gc,norm=True)
            # This repository's CTC head returns unnormalized logits even in eval.
            pl,pf=self.rec.pred_imglist(pc,norm=True)
            if self.gate_policy=='valid_crops_only':
                # Feature MSE requires valid paired crops, not a valid CTC transcript.
                # No confidence/exact-match/vocabulary/length/alignment rejection.
                assert pf.shape==gf.shape and torch.isfinite(pf).all() and torch.isfinite(gf).all()
                feat=F.mse_loss(pf.float(),gf.float())
                assert torch.isfinite(feat)
                han=sum(any('\u3400'<=c<='\u9fff' for c in text) for text in texts)
                return zero,feat,dict(ocr_candidate_lines=count,ocr_lines=len(pc),ocr_eligible=len(pc),ocr_invalid_crop_lines=count-len(pc),ocr_reliability_rejected=0,ocr_eligible_han=han,ocr_han_lines=han,ocr_gt_mean_confidence=0.,ocr_gt_confidence_measured=False)
            eligible=[];confidence=[];T=pl.shape[1]
            for i,text in enumerate(texts):
                probs=gl[i].softmax(-1);ids,loc=self.rec.decode(probs)
                decoded=self.rec.get_text(ids);conf=float(probs[loc,ids].mean()) if len(loc) else 0.
                repeats=sum(a==b for a,b in zip(text,text[1:]));known=all(c in self.rec.char2id and self.rec.char2id[c]!=0 for c in text)
                # Conservatively reject native max_chars boundary; no truncated transcript supervision.
                ok=bool(0<len(text)<20 and known and len(text)+repeats<=T and self.norm(decoded)==self.norm(text) and conf>=.9)
                entry={'eligible':ok,'text':text,'gt_prediction':decoded,'confidence':conf,'input_length':T,'known_vocabulary':known,'native_length_boundary_rejected':len(text)>=20}
                path=self.cache/(keys[i]+'.json')
                if path.exists():
                    old=json.loads(path.read_text());assert old['text']==text and old['eligible']==ok,(keys[i],old,entry)
                else:
                    temp=path.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(json.dumps(entry,ensure_ascii=False));temp.replace(path)
                if ok:eligible.append(i)
                confidence.append(conf)
            if eligible:
                logits=pl[eligible].float();target=[];lens=[]
                for i in eligible:target.extend(self.rec.char2id[c] for c in texts[i]);lens.append(len(texts[i]))
                ctc=F.ctc_loss(logits.log_softmax(-1).transpose(0,1),torch.tensor(target,device=pred.device,dtype=torch.long),torch.full((len(eligible),),T,dtype=torch.long,device=pred.device),torch.tensor(lens,dtype=torch.long,device=pred.device),blank=0,reduction='none',zero_infinity=False).div(T).mean()
                feat=F.mse_loss(pf[eligible].float(),gf[eligible].float())
            else:ctc=zero;feat=zero
        assert torch.isfinite(ctc) and torch.isfinite(feat)
        return ctc,feat,dict(ocr_lines=len(pc),ocr_eligible=len(eligible),ocr_eligible_han=sum(any('\u3400'<=c<='\u9fff' for c in texts[i]) for i in eligible),ocr_han_lines=sum(any('\u3400'<=c<='\u9fff' for c in text) for text in texts),ocr_gt_mean_confidence=sum(confidence)/len(confidence))

class Supervision:
    def __init__(self,native,variant,device,coefficients=None,calibrating=False):
        self.native=native;self.variant=variant;self.coefficients=coefficients or {};self.calibrating=calibrating
        self.teacher=Teacher(device,variant.get('gt_gate','legacy')) if calibrating or variant['ocr']!='none' else None
        self.last_losses={};self.diagnostics={};self.components={};self.prediction=None
    def spatial_components(self,pred,gt,mask,batch):
        pf,_=self.native.ResNet1(pred.type(self.native.dtype))
        with torch.no_grad():gf,_=self.native.ResNet1(gt.type(self.native.dtype))
        result={k:pred.sum()*0 for k in ['union','lines','lines_area']};missing=0;count=int(batch['n_lines'][0])
        for p,g in zip(pf[:4],gf[:4]):
            error=(p.float()-g.float()).square();size=p.shape[-2:];union=F.interpolate(mask.float(),size=size,mode='nearest')
            result['union']=result['union']+normalized_error(error,union)
            for kind,mode in [('lines','nearest'),('lines_area','area')]:
                values=[]
                for i in range(count):
                    m=F.interpolate(position_mask(batch['positions'][i]),size=size,mode=mode)
                    if float(m.sum())>0:values.append(normalized_error(error,m))
                    elif kind=='lines':missing+=1
                if values:result[kind]=result[kind]+torch.stack(values).mean()
        return result,missing
    def loss(self,pred,gt,mask,batch,velocity,t,noise):
        if self.calibrating:
            # Isolate the calibrated output variable from FSDP's backward hooks.
            # DiT is run under no_grad; only frozen VAE/teachers build a graph.
            with torch.enable_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                velocity=velocity.detach().clone().requires_grad_(True)
                latents=self.pipe._unpack_latents(noise.detach()-velocity,pred.shape[-2],pred.shape[-1],self.pipe.vae_scale_factor)
                latents=latents/self.pipe.vae.config.scaling_factor+self.pipe.vae.config.shift_factor
                pred=self.pipe.vae.decode(latents,return_dict=False)[0]
                return self._loss_impl(pred,gt,mask,batch,velocity,t)
        return self._loss_impl(pred,gt,mask,batch,velocity,t)
    def _loss_impl(self,pred,gt,mask,batch,velocity,t):
        base=self.native.loss(pred,gt,mask)
        zero=pred.sum()*0;components={'base':base,'ctc':zero,'features':zero}
        diag={};spatial=self.variant['spatial'];time=self.variant['time'];ocr=self.variant['ocr']
        if self.calibrating or spatial!='none':
            new,missing=self.spatial_components(pred,gt,mask,batch);components.update(new);diag['nearest_empty_line_scales']=missing
        if self.teacher:
            components['ctc'],components['features'],d=self.teacher.components(pred,gt,batch);diag.update(d)
        if self.calibrating:
            weighted=base;ctc=zero;features=zero
        else:
            raw=base if spatial=='none' else components[spatial]
            weighted=raw*float(self.coefficients.get('odm_scale',1.))
            if time!='none':weighted=weighted*time_weight(t,time).mean()/float(self.coefficients['time_mean'])
            ctc=components['ctc']*float(self.coefficients.get('ctc_scale',0.))
            features=components['features']*float(self.coefficients.get('feature_scale',0.))
        self.last_losses={'loss_odm':weighted,'loss_odm_raw':base/30,'loss_odm_reference':base,'loss_ctc':ctc,'loss_ocr_feature':features,'loss_ctc_raw':components['ctc'],'loss_ocr_feature_raw':components['features']}
        if spatial!='none' and not self.calibrating:self.last_losses['loss_odm_spatial_raw']=components[spatial]
        self.diagnostics=diag;self.components=components;self.prediction=velocity;self.t=t.detach()
        return weighted+ctc+features
    def clear_graph(self):
        self.components={};self.prediction=None;self.last_losses={}
