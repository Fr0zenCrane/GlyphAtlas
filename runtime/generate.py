"""Full2K original benchmark protocol with added regional atlas conditioning."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("GLYPH_OMNI_ROOT",str(Path(__file__).resolve().parents[1]))
ROOT=Path(os.environ["GLYPH_OMNI_ROOT"]);sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'));sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts/reproduce_flux_text'))
MODELS=Path(os.environ.get('GLYPH_MODEL_ROOT',str(ROOT/'assets/models')))
import numpy as np
import torch
import torch.distributed as dist
import yaml
from diffusers import FluxFillPipeline
from safetensors import safe_open
from generate_benchmark import BenchmarkDataset,BENCH,CODE,sha,write
from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from glyph_atlas.model import AtlasTransformer
from glyph_atlas.data import AtlasEncoder as PolicyAtlasEncoder,make_routes as policy_make_routes
from glyph_atlas.resolution_data import ResolutionAtlasEncoder
from glyph_atlas.global_trace import TraceDatasetMixin,install_renderer
from glyph_atlas.character_data_legacy import AtlasEncoder as CharacterAtlasEncoder,make_routes as character_make_routes


def generate_checked(pipe,audit_path,**kwargs):
    """Audit actual first-image backbone guidance and denoising-call count per rank."""
    if getattr(pipe,'_inference_parameters_checked',False):return generate_fill(pipe,**kwargs)
    expected_guidance=kwargs['guidance_scale'];expected_steps=kwargs['num_inference_steps'];observed=[]
    import inspect
    namespace=inspect.unwrap(generate_fill).__globals__;native=namespace['tranformer_forward']
    def traced(*args,**kw):
        guidance=kw.get('guidance')
        assert guidance is not None and torch.all(guidance==expected_guidance).item(),'Actual backbone guidance mismatch'
        observed.append(float(guidance[0]));return native(*args,**kw)
    namespace['tranformer_forward']=traced
    try:result=generate_fill(pipe,**kwargs)
    finally:namespace['tranformer_forward']=native
    assert len(observed)==expected_steps,(len(observed),expected_steps)
    write(audit_path,dict(passed=True,rank=dist.get_rank(),guidance=expected_guidance,denoising_forward_calls=len(observed),expected_steps=expected_steps,training_updates=0))
    pipe._inference_parameters_checked=True
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--training-run-id',required=True);p.add_argument('--limit-per-language',type=int,default=1000);p.add_argument('--steps',type=int,choices=[10,20],default=10);p.add_argument('--guidance',type=float,choices=[1.0,3.5],default=1.0);p.add_argument('--atlas-policy',choices=['character','zh_line_en_line','zh_char_en_word_v2'],default='zh_char_en_word_v2')
    p.add_argument('--atlas-mode',choices=['fixed48','fixed64','fixed80','dynamic'],default='fixed64')
    args=p.parse_args();cp=args.checkpoint.resolve();out=args.output.resolve();rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==8
    assert 8<=args.limit_per_language<=1000
    torch.cuda.set_device(rank);torch.set_num_threads(4);dist.init_process_group('nccl',timeout=datetime.timedelta(hours=2));os.chdir(CODE)
    if rank==0:
        for item in json.loads((cp/'weights_manifest.json').read_text())['files']:assert sha(cp/item['file'])==item['sha256']
    dist.barrier()
    assert args.atlas_policy=='zh_char_en_word_v2' or args.atlas_mode=='fixed64'
    protocol=dict(steps=args.steps,guidance=args.guidance,seed=42,atlas_mode=args.atlas_mode,punctuation_height=64 if args.atlas_mode=='dynamic' else int(args.atlas_mode[5:]),atlas_policy=args.atlas_policy,checkpoint_sha256=sha(cp/'weights_manifest.json'),images_per_language=args.limit_per_language)
    if rank==0:
        receipt=out/'inference_settings.json'
        if receipt.exists():assert json.loads(receipt.read_text())==protocol,'Inference configuration mismatch'
        else:
            assert not list(out.glob('generated/*/*.jpg')),'Unbound existing images'
            write(receipt,protocol)
    dist.barrier()
    transformer,loading=AtlasTransformer.from_pretrained(str(cp),torch_dtype=torch.bfloat16,output_loading_info=True)
    assert not loading.get('missing_keys') and not loading.get('unexpected_keys') and not loading.get('mismatched_keys'),loading
    parameters=[(n,p) for n,p in transformer.named_parameters() if n.startswith('atlas_')];assert sum(p.numel() for n,p in parameters)==593536
    mapping=json.loads((cp/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    for filename in set(mapping[n] for n,p in parameters):
        with safe_open(cp/filename,framework='pt',device='cpu') as weights:
            for n,p in parameters:
                if mapping[n]==filename:assert torch.equal(p.detach().cpu(),weights.get_tensor(n)),n
    device=torch.device('cuda',rank);transformer.eval()
    pipe=FluxFillPipeline.from_pretrained(str(MODELS/'FLUX.1-Fill-dev-diffusers'),transformer=transformer,torch_dtype=torch.bfloat16).to(device);pipe.set_progress_bar_config(disable=True)
    encoder=ResolutionAtlasEncoder(pipe,args.atlas_mode) if args.atlas_policy=='zh_char_en_word_v2' else (CharacterAtlasEncoder if args.atlas_policy=='character' else PolicyAtlasEncoder)(pipe)
    Dataset=BenchmarkDataset
    import importlib
    from font_support import install_fallback
    render_module=importlib.import_module(BenchmarkDataset.__mro__[2].__module__)
    install_fallback(render_module)
    if args.atlas_mode=='dynamic':
        install_renderer(render_module)
        class TracedBenchmark(TraceDatasetMixin,BenchmarkDataset):pass
        Dataset=TracedBenchmark
    route_builder=character_make_routes if args.atlas_policy=='character' else policy_make_routes
    cfg=yaml.safe_load((MODELS/'FLUX-Text-official/model_512/config.yaml').read_text())['model'];cp_hash=sha(cp/'weights_manifest.json')
    if rank==0:write(out/'load_validation.json',dict(passed=True,new_parameter_count=593536,strict_loading=True,exact_new_tensor_reload=True,checkpoint_sha256=cp_hash))
    for split in ['laion_word','wukong_word']:
        folder=out/'generated'/split;folder.mkdir(parents=True,exist_ok=True);original=BENCH/split/'test1k.json';source=json.loads(original.read_text())['data_list']
        dataset=Dataset(str(original),condition_size=512,target_size=512,glyph_scale=1,condition_type='word_fill',drop_text_prob=0,drop_image_prob=0,use_filter=False);records=[]
        for index in range(rank,args.limit_per_language,8):
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            data=dataset[index];name=source[index]['img_name'].split('.')[0]+'_0.jpg';destination=folder/name;meta=folder/(name+'.json')
            if meta.exists() and destination.exists():
                record=json.loads(meta.read_text());assert record['checkpoint_sha256']==cp_hash and record['output_sha256']==sha(destination) and record['inference_steps']==args.steps and record['guidance']==args.guidance;records.append(record);continue
            start=time.monotonic()
            if args.atlas_policy=='character':
                atlas=encoder.encode(data['texts'][:data['n_lines']],data['positions'])
            else:
                kw=dict(policy=args.atlas_policy)
                if isinstance(encoder,ResolutionAtlasEncoder):kw['trace']=data.get('global_glyph_trace')
                atlas=encoder.encode(data['texts'][:data['n_lines']],data['positions'],**kw)
            if atlas is not None:atlas['routes']=route_builder(atlas['regions'],atlas['owners'])
            condition=Condition(condition_type='word_fill',condition=[data['condition'],data['hint'],data['image'][None]],position_delta=[0,0])
            with torch.inference_mode():
                image=generate_checked(pipe,audit_path=out/f'runtime-parameters-rank{rank}.json',prompt=[data['description']],conditions=[condition],height=512,width=512,num_inference_steps=args.steps,guidance_scale=args.guidance,generator=torch.Generator(device).manual_seed(42),model_config=dict(cfg,atlas=atlas),default_lora=True).images[0]
            image.save(destination)
            record=dict(source_index=index,sample_index=0,source_img_name=source[index]['img_name'],output_name=name,output_sha256=sha(destination),checkpoint_sha256=cp_hash,seconds=time.monotonic()-start,prompt=data['description'],seed=42,inference_steps=args.steps,guidance=args.guidance,
                        atlas_mode=args.atlas_mode,atlas_plan=atlas.get('plan') if atlas else None,atlas_glyphs=atlas['glyph_count'] if atlas else 0,atlas_tokens=atlas['features'].shape[1] if atlas else 0,atlas_version='region-v2')
            write(meta,record);records.append(record);print('ATLAS_GENERATED',split,index,round(record['seconds'],2),flush=True)
        write(folder/f'rank{rank}_manifest.json',records);dist.barrier()
        if rank==0:
            combined=sum([json.loads((folder/f'rank{i}_manifest.json').read_text()) for i in range(8)],[]);combined.sort(key=lambda r:r['source_index'])
            assert [r['source_index'] for r in combined]==list(range(args.limit_per_language))
            write(folder/'generation_manifest.json',dict(benchmark_sha256=sha(original),samples_per_image=1,checkpoint_sha256=cp_hash,checkpoint_kind='full_transformer_plus_region_atlas',training_run_id=args.training_run_id,records=combined,full_benchmark=args.limit_per_language==1000,
                  generation_config=dict(steps=args.steps,guidance=args.guidance,seed=42,world_size=8,resolution=[512,512],atlas_version='region-v2',atlas_policy=args.atlas_policy,protocol='Original global glyph/scene/mask/prompt and sampler retained, plus polygon-routed atlas; one JPG per source; no character polygons')))
        dist.barrier()
    if rank==0:write(out/'generation_complete.json',dict(complete=True,images=2*args.limit_per_language,world_size=8,checkpoint_sha256=cp_hash,steps=args.steps,guidance=args.guidance))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
