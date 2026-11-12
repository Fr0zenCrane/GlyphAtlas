"""Fresh-base full-DiT Atlas + PP-OCRv3, global BS128 throughout.

Only this new run's own full checkpoints may be resumed. No historical
experiment initialization. Two-stage execution requires its explicit pinned config.
"""
import argparse
import copy
import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import time
from common import ROOT, CODE, VENDOR, MODELS, now, read, write, sha
from training_schedule import stage_for_step, switch_encoder, checkpoint_due, verify_complete_checkpoint


def learning_rate(step, cfg):
    total, warmup = cfg['steps'], cfg['warmup_steps']
    if cfg.get('training_schedule'):
        stage, _ = stage_for_step(step, cfg)
        if stage == 2:return cfg['stage2_lr']
        total = cfg['updates_per_epoch']
    assert 1 <= step <= total and 0 <= warmup < total
    if step <= warmup:
        return cfg['lr']*step/warmup
    fraction = (step-warmup)/(total-warmup) if warmup else (step-1)/max(1,total-1)
    return cfg['end_lr'] + .5*(cfg['lr']-cfg['end_lr'])*(1+math.cos(math.pi*fraction))


def build_plan(args):
    cfg = read(args.config)
    cfg.setdefault('atlas_training_mode','fixed64')
    cfg.setdefault('fsdp_mode','auto');cfg.setdefault('replica_bucket_mb',64)
    assert isinstance(cfg['replica_bucket_mb'],int) and 1<=cfg['replica_bucket_mb']<=1024
    assert cfg['atlas_training_mode'] in ('fixed64','random','two_stage')
    if cfg['atlas_training_mode']=='two_stage':
        assert cfg.get('training_schedule')=='fixed64_cosine_then_random_constant_v1'
        assert cfg['lr']==5e-5 and cfg['end_lr']==cfg['stage2_lr']==5e-6 and cfg['warmup_steps']==0
    else:assert not cfg.get('training_schedule')
    assert cfg['global_batch']==128 and cfg['micro_batch']==1
    assert cfg['ocr_time_weight'] in ('none','high_normalized_native')
    assert cfg['ocr_weight']==.25 and cfg['odm_weight']==cfg['ctc_weight']==0
    assert cfg['flow_weight']==cfg['mask_weight']==cfg['training_guidance']==cfg['training_time_shift']==1
    ready = read(args.dataset/'ready.json')
    assert ready['complete']
    assert not ready['smoke_only'] or args.allow_smoke_data, 'Smoke data requires explicit flag'
    if cfg.get('require_all_data'):
        release=read(ROOT/'configs/glyph_release.json')
        from dataset_contract import require_full_release
        require_full_release(ready,release)
    world=int(os.environ.get('WORLD_SIZE',8))
    from distributed_layout import layout_spec
    physical_local=int(os.environ.get('LOCAL_WORLD_SIZE',min(world,8)))
    virtual_local=getattr(args,'validation_local_size',None)
    if virtual_local:
        assert args.validate_only and world==physical_local and 1<virtual_local<world and world%virtual_local==0,'Virtual nodes are only allowed for single-host zero-update validation'
    topology=layout_spec(world,virtual_local or physical_local,cfg['fsdp_mode'])
    assert 128%world==0 and world>=8, 'Use at least eight GPUs; world size must divide 128'
    updates_per_epoch=ready['images']//128
    assert updates_per_epoch>0
    steps=args.steps if args.steps is not None else args.epochs*updates_per_epoch
    assert steps>0 and 0<=cfg['warmup_steps']<steps
    if cfg.get('training_schedule'):assert steps==2*updates_per_epoch, 'Exactly two epochs authorized'
    import hashlib
    code_hash=hashlib.sha256()
    for path in sorted(CODE.rglob('*.py')):
        code_hash.update(str(path.relative_to(CODE)).encode());code_hash.update(path.read_bytes())
    cfg.update(steps=steps,world_size=world,accumulation=128//world,
        dataset_sha256=ready['dataset_sha256'],dataset_images=ready['images'],
        updates_per_epoch=updates_per_epoch,dropped_images_per_epoch=ready['images']%128,
        seed=42,data_seed=1024,format_version='atlas-ocr-bs128-v3',distributed_layout=topology,
        dataset_release=ready.get('source_release'),
        font_assets_manifest_sha256=sha(ROOT/'provenance/fallback_fonts.json'),
        pipeline_compute_dtype='bfloat16_from_frozen_vae',
        runtime_code_sha256=code_hash.hexdigest(),
        source='Fresh black-forest-labs/FLUX.1-Fill-dev; no experimental checkpoint',
        optimizer=dict(type='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01,fused=True),
        resolution=[512,512],atlas_policy='zh_char_en_word_v2',gradient_clip_norm=.5)
    if args.resume:
        previous=read(args.resume/'training_state.json')
        assert previous['config']==cfg, 'Resume must preserve this run recipe, world size, data, and LR horizon'
        assert previous['config']['format_version']=='atlas-ocr-bs128-v3'
        assert previous['step']<steps
        assert read(args.resume/'resume_aux_complete.json')['complete']
        verify_complete_checkpoint(args.resume)
    return cfg, ready


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--steps',type=int,help='Explicit total optimizer-update horizon')
    budget.add_argument('--epochs',type=int,help='Compute updates from final admitted dataset size')
    parser.add_argument('--resume',type=Path,help='Only a full checkpoint from this BS128 runner')
    parser.add_argument('--stop-after',type=int,help='Bounded segment; absolute step, unchanged LR horizon')
    parser.add_argument('--allow-smoke-data',action='store_true')
    parser.add_argument('--validate-only',action='store_true',help='One BS128 accumulated backward, ZERO updates')
    parser.add_argument('--validation-local-size',type=int,help='ZERO-UPDATE ONLY: emulate node groups on one physical host; not a network benchmark')
    parser.add_argument('--validation-deterministic',action='store_true',help='ZERO-UPDATE ONLY: deterministic backend diagnostic; never changes production defaults')
    parser.add_argument('--validation-reduction-reference',action='store_true',help='ZERO-UPDATE ONLY: compare every real gradient bucket against an independent WORLD reduction')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    assert not args.validation_deterministic or args.validate_only
    assert not args.validation_reduction_reference or args.validate_only
    for key in ('config','dataset','output','resume'):
        value=getattr(args,key)
        if value is not None:setattr(args,key,value.resolve())
    cfg,ready=build_plan(args)
    if args.dry_run:
        print(json.dumps(cfg,ensure_ascii=False,indent=2));return
    assert not (args.output/'metrics.jsonl').exists(), 'Use a NEW output directory for every launch/segment'
    assert not (args.output/'STOP').exists()
    os.environ['GLYPH_RUN_DIR']=str(args.output)
    from tools_check import check_native
    check_native()
    import numpy as np
    import torch
    if args.validation_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    import yaml
    from dataset import Dataset,UpdateSampler
    from train_full import device_batch,save_checkpoint,restore_checkpoint
    from glyph_atlas.model import AtlasTrainingModel,shard
    from glyph_atlas.data import AtlasEncoder
    from glyph_atlas.stage1_ocr import OCRStage1,install
    world,rank,local=int(os.environ['WORLD_SIZE']),int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(4)
    if cfg.get('required_world_size'):
        assert world==cfg['required_world_size'] and int(os.environ['LOCAL_WORLD_SIZE'])==cfg['required_local_world_size'], 'Authorized hardware: four nodes of eight GPUs'
        assert cfg['required_gpu_substring'] in torch.cuda.get_device_name(local), 'This scaling config is authorized for H200'
    dist.init_process_group('nccl',timeout=datetime.timedelta(hours=2))
    from distributed_layout import create_layout,synchronize_replica_gradients,verify_rank_agreement,clip_global_grad_norm
    consistency=verify_rank_agreement(cfg,ready,args.dataset)
    layout=create_layout(cfg['fsdp_mode'],local_size=args.validation_local_size,validate_hosts=not args.validation_local_size)
    layout['spec']['virtual_node_validation']=bool(args.validation_local_size)
    if rank==0:
        assert sha(args.dataset/'dataset.sqlite')==ready['dataset_sha256']
        import fcntl
        args.output.mkdir(parents=True,exist_ok=True)
        lock=(args.output/'.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(args.output/'config.json',cfg)
        write(args.output/'distributed_layout.json',layout['spec'])
        write(args.output/'cluster_consistency.json',consistency)
        import subprocess,importlib.metadata
        revision=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
        dirty=subprocess.run(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True)
        write(args.output/'run_provenance.json',dict(git_commit=revision.stdout.strip() if revision.returncode==0 else None,
            tracked_changes=bool(dirty.stdout),runtime_code_sha256=cfg['runtime_code_sha256'],
            packages={name:importlib.metadata.version(name) for name in ['torch','torchvision','diffusers','transformers']},
            data_release=ready.get('source_release'),dataset_sha256=ready['dataset_sha256'],
            vendor_manifest_sha256=sha(ROOT/'provenance/vendored_source.json'),checked_beijing=now()))
    dist.barrier()
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    os.chdir(VENDOR)
    native=yaml.safe_load((MODELS/'FLUX-Text-official/model_512/config.yaml').read_text())
    odm=copy.deepcopy(native['train']['odm_loss'])
    odm.update(modelpath=str(MODELS/'FLUX-Text-official/epoch_100.pt'),**{f'w_loss_{i}':1 for i in range(1,5)})
    # Upstream constructs ODM during initialization; it is removed before any training loss.
    model=AtlasTrainingModel(flux_pipe_id=str(MODELS/'FLUX.1-Fill-dev-diffusers'),device='cpu',
        dtype=torch.bfloat16,lora_config={},model_config=native['model'],optimizer_config=None,
        gradient_checkpointing=True,odm_loss_config=odm)
    model.atlas_policy=cfg['atlas_policy']
    assert shard(model,local,layout)==dict(base=11902391360,atlas=593536)
    from pipeline_precision import bind_training_precision
    precision=bind_training_precision(model.flux_pipe)
    write(args.output/f'pipeline-precision-rank{rank}.json',precision)
    assert sum(isinstance(m,FSDP) for m in model.transformer.modules())==58
    model.transformer.train()
    optimizer=torch.optim.AdamW(model.transformer.parameters(),lr=cfg['lr'],betas=(.9,.999),
        eps=1e-8,weight_decay=.01,fused=True)
    dataset=Dataset(args.dataset,seed=cfg['seed'])
    sampler=UpdateSampler(len(dataset),rank,world,cfg['global_batch'],cfg['data_seed'])
    step=restore_checkpoint(model.transformer,optimizer,sampler,args.resume) if args.resume else 0
    _, active_mode=stage_for_step(max(1,step),cfg)
    switch_encoder(model,active_mode,cfg['seed'])
    if args.resume:
        assert all(float(v['step'])==step and v['exp_avg'].dtype==torch.float32 and
                   v['exp_avg_sq'].dtype==torch.float32 for v in optimizer.state.values())
        aux=torch.load(args.resume/f'aux-rank{rank}.pt',map_location='cpu',weights_only=False)
        assert aux['step']==step
        model.atlas_encoder.cache=aux['atlas_cache']
        assert aux.get('atlas_mode',active_mode)==active_mode
        if active_mode=='random':model.atlas_encoder.dynamic_cache=aux['dynamic_cache']
        del aux
    else:
        assert not optimizer.state
    # OCR initialization preserves all model RNG streams; GT branch is no_grad, generated branch differentiable.
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    model.supervision=OCRStage1(model.device,.25,cfg['ocr_time_weight'])
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    model.odm_loss=None;gc.collect();torch.cuda.empty_cache();patch=install()
    if rank==0:write(args.output/'native_patch.json',patch)
    loader=DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=cfg['workers'],pin_memory=True,
        generator=torch.Generator().manual_seed(cfg['seed']+rank),prefetch_factor=2 if cfg['workers'] else None)
    iterator=iter(loader);initial_step=step;limit=args.stop_after or cfg['steps']
    assert step<limit<=cfg['steps']
    started=time.monotonic();validation_inputs=[]
    def tensor_digest(tensor):
        import hashlib
        value=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return hashlib.sha256(value.numpy().tobytes()).hexdigest()
    epoch_checkpoints={}
    if args.resume and (args.resume.parent/'epoch_checkpoints.json').is_file():
        epoch_checkpoints=read(args.resume.parent/'epoch_checkpoints.json')
    if args.resume and step%cfg['updates_per_epoch']==0:
        epoch_checkpoints[str(step//cfg['updates_per_epoch'])]=verify_complete_checkpoint(args.resume)
    while step<limit:
        stage, active_mode=stage_for_step(step+1,cfg)
        changed=switch_encoder(model,active_mode,cfg['seed'])
        if changed and rank==0:
            write(args.output/f'stage{stage}_transition.json',dict(step_before_update=step,next_step=step+1,atlas_mode=active_mode,optimizer_reset=False,sampler_reset=False,checked_beijing=now()))
        assert not (args.output/'STOP').exists(), 'Stop requested; resume last complete checkpoint'
        optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(step+1,cfg)
        for group in optimizer.param_groups:group['lr']=lr
        tick=time.monotonic();sums={}
        # FSDP averages within its shard group each microbatch in FP32.
        # Average the accumulated shards across replicas once below: global128.
        for micro in range(cfg['accumulation']):
            if active_mode=='random':
                # Unique presentation context across all ranks/microbatches;
                # same context after resume, and no native RNG consumption.
                model.atlas_encoder.set_context((step*cfg['accumulation']+micro)*world+rank+1,0)
            batch=device_batch(next(iterator),model.device)
            if args.validate_only:
                validation_inputs.append(dict(micro=micro,image=tensor_digest(batch['image']),condition=tensor_digest(batch['condition']),hint=tensor_digest(batch['hint']),texts=batch['texts'],cpu_rng=tensor_digest(torch.get_rng_state()),cuda_rng=tensor_digest(torch.cuda.get_rng_state())))
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.step(batch)
            assert torch.isfinite(loss)
            assert torch.allclose(loss,model.res['loss_sd']+model.res['loss_mask']+model.res['loss_ocr_feature'],atol=1e-6,rtol=1e-5)
            assert model.odm_loss is None and float(model.res['loss_ctc'])==0 and float(model.res['loss_odm'])==0
            assert torch.equal(model.res['loss_ocr_feature'],model.res['loss_ocr_feature_raw']*.25*model.supervision.time_multiplier)
            if args.validate_only:validation_inputs[-1].update(time=model._stage1_t.detach().cpu().tolist(),loss={k:float(v.detach()) for k,v in model.res.items()},atlas=dict(model.atlas_diagnostics))
            (loss/cfg['accumulation']).backward()
            for key,value in model.res.items():sums[key]=sums.get(key,0.)+float(value.detach())/cfg['accumulation']
            del loss,batch
            model.res={};model.supervision.clear()
        communication=synchronize_replica_gradients(model.transformer,cfg['replica_bucket_mb'],args.validation_reduction_reference)
        # Clip only after the complete global mean; shard group norm is sufficient
        # because corresponding gradients now agree on every node.
        norm=clip_global_grad_norm(model.transformer,.5)
        assert torch.isfinite(norm) and norm>0
        if step==initial_step:
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.transformer.parameters())
            write(args.output/f'first-gradient-rank{rank}.json',dict(passed=True,optimizer_updates=0,
                global_batch=128,accumulation=cfg['accumulation'],step_before_update=step,
                communication=communication,
                grad_norm=float(norm),checked_beijing=now()))
            dist.barrier()
        if args.validate_only:
            write(args.output/f'validation-inputs-rank{rank}.json',dict(initial_parameter_probe_sha256=model.initial_parameter_probe_sha256,initial_atlas_sha256=model.initial_atlas_sha256,inputs=validation_inputs,precision=precision,deterministic_diagnostic=args.validation_deterministic))
            # Full-model diagnostic fingerprint, no updates. Aggregate in the
            # shard group so replicated nodes are not counted more than once.
            fingerprint=torch.zeros(3,device=model.device,dtype=torch.float64)
            for parameter in model.transformer.parameters():
                if parameter.grad is None or not parameter.grad.numel():continue
                g=parameter.grad.detach().double()
                fingerprint+=torch.stack([g.sum(),g.square().sum(),g.abs().sum()])
            dist.all_reduce(fingerprint,group=model.transformer.process_group)
            metric_keys=sorted(sums);metric_values=torch.tensor([sums[k] for k in metric_keys],dtype=torch.float64,device=model.device)
            dist.all_reduce(metric_values);metric_values/=world;torch.cuda.synchronize()
            if rank==0:write(args.output/'validation.json',dict(passed=True,new_training_updates=0,
                source='official base' if args.resume is None else 'new BS128 run checkpoint',
                global_batch=128,world_size=world,accumulation=cfg['accumulation'],checkpoint_written=False,
                distributed_layout=layout['spec'],communication=communication,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                global_loss=dict(zip(metric_keys,metric_values.tolist())),
                deterministic_diagnostic=args.validation_deterministic,
                clipped_gradient_sum_sumsq_abs=fingerprint.tolist(),first_global_batch_seconds=time.monotonic()-tick,
                checked_beijing=now()))
            break
        optimizer.step();step+=1
        keys=sorted(sums);values=torch.tensor([sums[k] for k in keys],dtype=torch.float64,device=model.device)
        dist.all_reduce(values);values/=world;torch.cuda.synchronize()
        record=dict(step=step,target_steps=cfg['steps'],stage=stage,atlas_mode=active_mode,epoch=step/cfg['updates_per_epoch'],lr=lr,loss=dict(zip(keys,values.tolist())),
            grad_norm=float(norm),global_samples=128,samples_seen=step*128,
            communication=communication,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            compute_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,checked_beijing=now())
        if rank==0:
            with (args.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            write(args.output/'status.json',dict(state='training',**record));print('OPTIMIZER_STEP',json.dumps(record),flush=True)
        optimizer.zero_grad(set_to_none=True)
        if checkpoint_due(step,limit,cfg):
            import shutil
            gc.collect();torch.cuda.empty_cache()
            if rank==0:assert shutil.disk_usage(args.output).free>250_000_000_000, 'Need 250 GB checkpoint headroom'
            dist.barrier()
            saved=save_checkpoint(model.transformer,optimizer,sampler,args.output,step,cfg,export=True)
            auxiliary=dict(step=step,atlas_mode=active_mode,atlas_cache=model.atlas_encoder.cache,
                resolution_context=dict(next_global_update=step,seed=cfg['seed'],formula='(step*accumulation+micro)*world+rank+1; private allocator RNG'))
            if active_mode=='random':auxiliary['dynamic_cache']=model.atlas_encoder.dynamic_cache
            torch.save(auxiliary,saved/f'aux-rank{rank}.pt')
            dist.barrier()
            if rank==0:write(saved/'resume_aux_complete.json',dict(complete=True,step=step,ranks=world,
                sampler='consumed BS128 optimizer updates; stateless per-sample transforms',atlas_cache=True))
            dist.barrier()
            if rank==0:
                complete_checkpoint=verify_complete_checkpoint(saved)
                if step%cfg['updates_per_epoch']==0:
                    epoch_checkpoints[str(step//cfg['updates_per_epoch'])]=complete_checkpoint
                    write(args.output/'epoch_checkpoints.json',epoch_checkpoints)
            dist.barrier()
    if not args.validate_only and rank==0:
        if cfg.get('training_schedule') and step==cfg['steps']:
            assert set(epoch_checkpoints)=={'1','2'}, 'Both retained epoch endpoints are required'
            for endpoint in epoch_checkpoints.values():verify_complete_checkpoint(Path(endpoint['path']))
        write(args.output/'completed.json',dict(complete=True,step=step,
            new_updates=step-initial_step,epoch_checkpoints=epoch_checkpoints,checked_beijing=now()))
        write(args.output/'status.json',dict(state='complete' if step==cfg['steps'] else 'segment_complete',**record))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
