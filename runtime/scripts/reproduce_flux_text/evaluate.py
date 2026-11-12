"""Eight-rank complete AnyText scoring using the published FLUX-Text functions.

Preflight checks metric assets, preprocessing and device placement on all eight
GPUs. It is not a trained-model evaluation. The full scorer requires an explicit
sample count and a complete, provenance-bound generation manifest.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(os.environ["GLYPH_OMNI_ROOT"])
CODE=ROOT/'vendor/FluxText'
BENCH=Path(os.environ.get('GLYPH_BENCHMARK_ROOT',str(ROOT/'data/raw/AnyText-benchmark/benchmark_extracted/benchmark')))
DEFAULT_OUT=Path(os.environ.get('FLUX_TEXT_RUN_DIR',str(ROOT/'outputs/flux-text-reproduction'))).resolve()/'evaluation'
OCR=ROOT/'data/cache/modelscope/models/damo/cv_convnextTiny_ocr-recognition-general_damo'
os.environ.setdefault('TORCH_HOME',str(ROOT/'data/cache/home/.cache/torch'))
sys.path.insert(0,str(CODE/'eval'))
sys.path.insert(0,str(CODE))

import cv2
from easydict import EasyDict
import Levenshtein
import numpy as np
from PIL import Image,ImageFont
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from recognizer import TextRecognizer,crop_image
from dataset_util import load
from t3_dataset import draw_glyph,draw_glyph2,get_caption_pos


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def official_functions(file,names,namespace):
    """Preserve actual source function bodies instead of reimplementing them."""
    tree=ast.parse(file.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),namespace)
    return namespace


def load_ocr_annotations(input_path):
    """Adapt optional caption-position metadata without rewriting frozen inputs.

    Native T3DataSet uses -1 when position labels are absent. The public OCR
    loader requires the key, although get_item uses caption position prob=0
    and OCR cropping depends only on the unchanged polygons.
    """
    content = load(input_path)
    for row in content['data_list']:
        for annotation in row.get('annotations', []):
            annotation.setdefault('pos', -1)
    return content


def ocr_functions():
    ns=dict(cv2=cv2,np=np,torch=torch,random=random,math=math,Levenshtein=Levenshtein,
            load=load_ocr_annotations,draw_glyph=draw_glyph,draw_glyph2=draw_glyph2,get_caption_pos=get_caption_pos,
            PLACE_HOLDER='*',max_chars=20,max_lines=20,
            font=ImageFont.truetype(str(CODE/'font/Arial_Unicode.ttf'),size=60))
    return official_functions(CODE/'eval/eval_dgocr.py',
                              ['draw_pos','load_data','get_item','get_ld','pre_process'],ns)


def recognizer(split):
    dictionary='ppocr_keys_v1.txt' if split=='wukong_word' else 'en_dict.txt'
    return TextRecognizer(EasyDict(rec_image_shape='3, 48, 320',
        rec_char_dict_path=str(CODE/'eval/ocr_weights'/dictionary),rec_batch_num=1,use_fp16=False),None)


def ocr_inputs(path,item,funcs):
    img=cv2.imread(str(path))
    assert img is not None,path
    if img.shape[:2]!=(512,512):img=cv2.resize(img,(512,512))
    tensor=torch.from_numpy(img).permute(2,0,1).float()
    crops=[crop_image(tensor,(item['positions'][k]*255.).astype(np.uint8)) for k in range(item['n_lines'])]
    return funcs['pre_process'](crops,'3, 48, 320') if crops else []


def score_strings(pred,gt,rec,funcs):
    gt_ids=[rec.char2id.get(c,len(rec.chars)-1) for c in gt]
    pred_ids=[rec.char2id.get(c,len(rec.chars)-1) for c in pred]
    return int(pred==gt),float(funcs['get_ld'](pred_ids,gt_ids))


def metric_models(rank):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import lpips
    from pytorch_fid.inception import InceptionV3
    predictor=pipeline(Tasks.ocr_recognition,model=str(OCR),device=f'gpu:{rank}')
    perceptual=lpips.LPIPS(net='alex').eval().to(f'cuda:{rank}')
    inception=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(f'cuda:{rank}')
    return predictor,perceptual,inception


def assets():
    paths=[CODE/'eval/eval_dgocr.py',CODE/'eval/recognizer.py',CODE/'eval/eval_lpips.py',
           CODE/'eval/t3_dataset.py',CODE/'eval/ocr_weights/ppocr_keys_v1.txt',CODE/'eval/ocr_weights/en_dict.txt',
           OCR/'configuration.json',OCR/'pytorch_model.pt',OCR/'vocab.txt',
           Path(os.environ['TORCH_HOME'])/'hub/checkpoints/pt_inception-2015-12-05-6726825d.pth',
           Path(os.environ['TORCH_HOME'])/'hub/checkpoints/alexnet-owt-7be5be79.pth']
    return [{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]


def preflight(rank,out,models,funcs,lpips_prep):
    predictor,perceptual,inception=models
    from pytorch_fid.fid_score import get_activations
    split='laion_word' if rank%2==0 else 'wukong_word'
    rows=funcs['load_data'](str(BENCH/split/'test1k.json'))
    index=rank//2;item=funcs['get_item'](rows,index);path=BENCH/split/'imgs'/item['img_name']
    crop=ocr_inputs(path,item,funcs)[0]
    rec=recognizer(split)
    with torch.inference_mode():
        prediction=predictor(crop)['text'][0]
        im=lpips_prep(str(path)).cuda(rank)
        equal=float(perceptual(im,im).item())
    activations=get_activations([str(path)],inception,batch_size=1,dims=2048,device=f'cuda:{rank}',num_workers=0)
    assert abs(equal)<1e-7 and activations.shape==(1,2048) and np.isfinite(activations).all()
    assert next(predictor.model.parameters()).device.index==rank
    identical,ned=score_strings(item['texts'][0],item['texts'][0],rec,funcs)
    assert identical==1 and ned==1
    result=dict(rank=rank,split=split,input_index=index,ocr_device=str(next(predictor.model.parameters()).device),
                ocr_inference_completed=isinstance(prediction,str),lpips_identical_image=equal,
                inception_shape=list(activations.shape),inception_finite=True,
                official_exact_match_self_check=True,formal_model_evaluation=False)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from glyph_atlas.supervision import Teacher
    teacher=Teacher(torch.device('cuda',rank))
    with torch.inference_mode():
        rgb=torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1).float().cuda(rank)
        teacher_crop=crop_image(rgb,(item['positions'][0]*255.).astype(np.uint8))
        logits,features=teacher.rec.pred_imglist([teacher_crop],norm=True)
        assert torch.isfinite(logits).all() and logits.shape[-1]==6625
        perceptual.spatial=True
        spatial=perceptual(im,im)
        perceptual.spatial=False
        assert abs(float(spatial.max()))<1e-6
    result.update(secondary_teacher_finite=True,spatial_lpips_identical_passed=True,primary_is_independent_dgocr=True)
    gathered=[None]*8;dist.all_gather_object(gathered,result)
    if rank==0:
        # Unequal-size shards must not be averaged as local means.
        all_values=np.arange(37*11,dtype=np.float64).reshape(37,11)/100
        parts=[all_values[i::8] for i in range(8)]
        joined=np.concatenate(parts)
        assert np.allclose(np.cov(joined,rowvar=False),np.cov(all_values,rowvar=False),rtol=1e-12,atol=1e-12)
        write(out/'preflight.json',dict(passed=True,ranks=gathered,world_size=8,assets=assets(),
            unequal_shard_feature_aggregation_checked=True,training_updates=0,benchmark_scores_produced=False))
        print('EVALUATION_PREFLIGHT_PASSED',flush=True)


def full_score(args,rank,models,funcs,lpips_prep):
    from pytorch_fid.fid_score import get_activations,calculate_frechet_distance
    predictor,perceptual,inception=models
    split=args.split;out=args.output/split
    benchmark_path=BENCH/split/'test1k.json'
    rows=funcs['load_data'](str(benchmark_path))
    count=len(rows);assert count==args.expect_images
    generated=json.loads(args.generation_manifest.read_text())
    assert generated['benchmark_sha256']==sha(benchmark_path)
    assert generated['samples_per_image']==args.samples_per_image
    assert generated['checkpoint_sha256'] and generated['training_run_id']
    records=generated['records']
    by_id={(r['source_index'],r['sample_index']):r for r in records}
    expected={(i,j) for i in range(count) for j in range(args.samples_per_image)}
    assert len(records)==len(by_id)==len(expected) and set(by_id)==expected
    expected_names={f"{row['img_name'].split('.')[0]}_{j}.jpg" for row in rows for j in range(args.samples_per_image)}
    assert {p.name for p in args.img_dir.glob('*.jpg')}==expected_names, 'Missing or extra generated JPGs'
    for i,j in expected:
        r=by_id[(i,j)]
        assert r['source_img_name']==rows[i]['img_name']
        assert r['output_name']==rows[i]['img_name'].split('.')[0]+f'_{j}.jpg'
        path=args.img_dir/r['output_name']
        assert path.resolve().parent==args.img_dir.resolve() and path.exists(),path
        if i%8==rank:assert sha(path)==r['output_sha256'],path
    dist.barrier()
    rec=recognizer(split);results=[];pred_files=[];gt_files=[]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from glyph_atlas.supervision import Teacher
    teacher=Teacher(torch.device('cuda',rank))
    char_counts=json.loads((ROOT/'provenance/character_counts.json').read_text())
    for i in range(rank,len(rows),8):
        item=funcs['get_item'](rows,i);gt=BENCH/split/'imgs'/item['img_name'];gt_files.append(str(gt))
        for j in range(args.samples_per_image):
            path=args.img_dir/by_id[(i,j)]['output_name'];pred_files.append(str(path))
            crops=ocr_inputs(path,item,funcs)
            with torch.inference_mode():
                strings=[predictor(im)['text'][0] for im in crops]
                lp=float(perceptual(lpips_prep(str(gt)).cuda(rank),lpips_prep(str(path)).cuda(rank)).item())
            with torch.inference_mode():
                rgb=torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1).float().cuda(rank)
                teacher_crops=[crop_image(rgb,(item['positions'][k]*255.).astype(np.uint8)) for k in range(item['n_lines'])]
                if teacher_crops:
                    logits,_=teacher.rec.pred_imglist(teacher_crops,norm=True)
                    teacher_strings=[teacher.rec.get_text(teacher.rec.decode(x.softmax(-1))[0]) for x in logits]
                else:teacher_strings=[]
                perceptual.spatial=True
                spatial=perceptual(lpips_prep(str(gt)).cuda(rank),lpips_prep(str(path)).cuda(rank))
                perceptual.spatial=False
                union=np.clip(sum(item['positions'][:item['n_lines']]),0,1) if item['n_lines'] else np.zeros((512,512,1))
                mask=torch.from_numpy(union).permute(2,0,1).unsqueeze(0).float().cuda(rank)
                mask=torch.nn.functional.interpolate(mask,size=spatial.shape[-2:],mode='area')
                fg=float((spatial*mask).sum()/mask.sum().clamp_min(1e-8));bg=float((spatial*(1-mask)).sum()/(1-mask).sum().clamp_min(1e-8))
            lines=[]
            for k,pred in enumerate(strings):
                acc,ned=score_strings(pred,item['texts'][k],rec,funcs)
                text=item['texts'][k];poly=np.asarray(item['polygons'][k],np.float32);_,dims,_=cv2.minAreaRect(poly);ops=Levenshtein.editops(text,pred)
                lines.append(dict(line=k,gt=text,prediction=pred,exact=acc,ned=ned,teacher_prediction=teacher_strings[k],teacher_exact=int(teacher_strings[k]==text),han=any('\u3400'<=c<='\u9fff' for c in text),small_text=min(dims)<16,long_text=len(text)>=12,crowded=item['n_lines']>=6,rare_character=any(c.isalnum() and char_counts.get(c,0)<=20 for c in text),insertions=sum(o[0]=='insert' for o in ops),deletions=sum(o[0]=='delete' for o in ops),substitutions=sum(o[0]=='replace' for o in ops)))
            results.append(dict(source_index=i,sample_index=j,img_name=item['img_name'],lpips=lp,foreground_lpips=fg,background_lpips=bg,lines=lines))
        if len(results)%25==0:print('SCORING',rank,split,len(results),flush=True)
    # Each rank saves full float64 feature arrays; rank 0 concatenates all samples.
    # Do not average local FIDs or covariances. GT is represented once per source.
    gt_feats=get_activations(gt_files,inception,batch_size=25,dims=2048,device=f'cuda:{rank}',num_workers=0)
    pred_feats=get_activations(pred_files,inception,batch_size=25,dims=2048,device=f'cuda:{rank}',num_workers=0)
    assert np.isfinite(gt_feats).all() and np.isfinite(pred_feats).all()
    write(out/f'rank{rank}.json',results)
    np.savez(out/f'rank{rank}_fid.npz',gt=gt_feats,pred=pred_feats)
    dist.barrier()
    if rank==0:
        all_results=sum([json.loads((out/f'rank{r}.json').read_text()) for r in range(8)],[])
        assert len(all_results)==len(expected)
        assert {(r['source_index'],r['sample_index']) for r in all_results}==expected
        expected_lines=sum(min(len(r['texts']),20) for r in rows)*args.samples_per_image
        lines=[line for r in all_results for line in r['lines']]
        assert len(lines)==expected_lines
        gt=np.concatenate([np.load(out/f'rank{r}_fid.npz')['gt'] for r in range(8)])
        pred=np.concatenate([np.load(out/f'rank{r}_fid.npz')['pred'] for r in range(8)])
        assert gt.shape==(count,2048) and pred.shape==(len(expected),2048)
        fid=float(calculate_frechet_distance(gt.mean(0),np.cov(gt,rowvar=False),pred.mean(0),np.cov(pred,rowvar=False)))
        result=dict(split=split,images=count,outputs=len(expected),text_lines=len(lines),world_size=8,
            sen_acc=float(np.mean([x['exact'] for x in lines])),ned=float(np.mean([x['ned'] for x in lines])),
            lpips=float(torch.tensor([x['lpips'] for x in all_results],dtype=torch.float32).mean()),fid=fid,
            source_code='AMAP-ML/FluxText d406544',generation_manifest_sha256=sha(args.generation_manifest),
            assets=assets(),metric_scope='Official released full-image FID/LPIPS; official line OCR. Paper wording differs.',
            adaptation='Disjoint eight-rank sharding; explicit source pairing and global feature aggregation; original score functions retained.')
        result.update(foreground_lpips=float(np.mean([x['foreground_lpips'] for x in all_results])),background_lpips=float(np.mean([x['background_lpips'] for x in all_results])),teacher_sen_acc=float(np.mean([x['teacher_exact'] for x in lines])),secondary_ocr='PP-OCRv3 used for training; primary is independently weighted DuGuang ConvNeXt OCR',strata={k:dict(text_lines=sum(x[k] for x in lines),sen_acc=float(np.mean([x['exact'] for x in lines if x[k]])) if any(x[k] for x in lines) else None) for k in ['han','small_text','long_text','crowded','rare_character']},character_errors={k:sum(x[k] for x in lines) for k in ['insertions','deletions','substitutions']})
        write(out/'metrics.json',result);print('BENCHMARK_SCORED',json.dumps(result),flush=True)
    dist.barrier()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--expect-images',type=int,default=1000)
    parser.add_argument('--split',choices=['laion_word','wukong_word'])
    parser.add_argument('--img-dir',type=Path)
    parser.add_argument('--generation-manifest',type=Path)
    parser.add_argument('--samples-per-image',type=int,choices=[1,4])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if not args.preflight:
        assert all([args.split,args.img_dir,args.generation_manifest,args.samples_per_image]),'Full evaluation requires explicit protocol and generation provenance'
    assert int(os.environ['WORLD_SIZE'])==8
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.split:(args.output/args.split).mkdir(exist_ok=True)
    funcs=ocr_functions()
    lpips_prep=official_functions(CODE/'eval/eval_lpips.py',['preprocess_image'],
                                  {'Image':Image,'transforms':transforms})['preprocess_image']
    models=metric_models(rank)
    if args.preflight:preflight(rank,args.output,models,funcs,lpips_prep)
    else:full_score(args,rank,models,funcs,lpips_prep)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
