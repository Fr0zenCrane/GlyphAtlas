"""Node-local FSDP plus one FP32 replica-gradient mean per optimizer update.

All-gather and reduce-scatter stay inside a node. FSDP still reduces each
microbatch in FP32; only accumulated gradient shards cross nodes. No no_sync(),
full-gradient accumulation, private FSDP hooks, or loss/precision changes.
"""
import os,socket,hashlib,json
import torch
import torch.distributed as dist

def verify_rank_agreement(config,ready,dataset):
    """Abort together before model initialization on version/data mismatches."""
    import importlib.metadata
    from pathlib import Path
    from common import sha
    identity=dict(config=config,data_ready=ready,packages={name:importlib.metadata.version(name) for name in
        ['torch','torchvision','diffusers','transformers','numpy','Pillow','fonttools']},
        cuda_runtime=torch.version.cuda,nccl_version=torch.cuda.nccl.version(),gpu=torch.cuda.get_device_name())
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    error=None
    if int(os.environ['LOCAL_RANK'])==0:
        try:
            dataset=Path(dataset)
            assert sha(dataset/'dataset.sqlite')==ready['dataset_sha256'],'Dataset database hash differs'
            for c in ready.get('components',[]):
                assert sha(dataset/c['path']/'dataset.sqlite')==c['dataset_sha256'],c['path']+' database hash differs'
        except Exception as e:error=type(e).__name__+': '+str(e)
    record=dict(rank=dist.get_rank(),host=socket.gethostname(),identity_sha256=digest,local_error=error)
    records=[None]*dist.get_world_size();dist.all_gather_object(records,record)
    if any(r['local_error'] for r in records):raise ValueError('Per-node input verification failed: '+str(records))
    if len({r['identity_sha256'] for r in records})!=1:raise ValueError('Nodes disagree on runtime code, config, data release, environment or GPU type: '+str(records))
    return dict(passed=True,ranks=records,packages=identity['packages'],gpu=identity['gpu'],
        scope='Runtime code/config/release/ready/environment agreement; dataset and component database hashes on each physical node. Image hashes are verified during data import.')

def layout_spec(world,local_size,mode='auto'):
    if world<1 or local_size<1 or world%local_size:raise ValueError('Require homogeneous whole nodes')
    if mode=='auto':mode='node_shard' if world>local_size else 'world_full'
    if mode not in ('node_shard','world_full'):raise ValueError('Unknown sharding mode')
    shard_size=local_size if mode=='node_shard' else world
    replicas=world//shard_size
    return dict(mode=mode,world_size=world,local_world_size=local_size,shard_size=shard_size,replicas=replicas,
        shard_ranks=[list(range(n*shard_size,(n+1)*shard_size)) for n in range(replicas)],
        replica_ranks=[[n*shard_size+i for n in range(replicas)] for i in range(shard_size)],
        gradient_policy='FP32 local reduce-scatter each micro; replica mean once before clipping')

def create_layout(mode='auto',local_size=None,validate_hosts=True):
    world=dist.get_world_size();rank=dist.get_rank()
    local_size=int(os.environ['LOCAL_WORLD_SIZE']) if local_size is None else int(local_size)
    spec=layout_spec(world,local_size,mode)
    if validate_hosts:
        own=dict(rank=rank,host=socket.gethostname(),local_rank=int(os.environ['LOCAL_RANK']),local_size=int(os.environ['LOCAL_WORLD_SIZE']))
        all_hosts=[None]*world;dist.all_gather_object(all_hosts,own)
        if any(v['local_size']!=local_size or v['local_rank']!=i%local_size for i,v in enumerate(all_hosts)):
            raise ValueError('Rank placement differs from homogeneous torchrun node layout')
        hosts=[{v['host'] for v in all_hosts[i:i+local_size]} for i in range(0,world,local_size)]
        if any(len(h)!=1 for h in hosts) or len(set().union(*hosts))!=len(hosts):
            raise ValueError('Node-local groups do not match physical hosts')
        spec['hosts']=[next(iter(h)) for h in hosts]
    shard_group=dist.group.WORLD;replica_group=None
    if spec['mode']=='node_shard':
        for ranks in spec['shard_ranks']:
            group=dist.new_group(ranks)
            if rank in ranks:shard_group=group
        if spec['replicas']>1:
            for ranks in spec['replica_ranks']:
                group=dist.new_group(ranks)
                if rank in ranks:replica_group=group
    return dict(spec=spec,shard_group=shard_group,replica_group=replica_group,
                shard_rank=rank%spec['shard_size'],replica_index=rank//spec['shard_size'])

def attach_layout(model,layout):
    model._atlas_layout=layout

def checkpoint_layout(model):
    layout=getattr(model,'_atlas_layout',None)
    if layout is None or layout['spec']['mode']=='world_full':return dict(mode='world_full',shard_size=dist.get_world_size(),replicas=1)
    return {k:layout['spec'][k] for k in ['mode','shard_size','replicas']}

@torch.no_grad()
def synchronize_initial_replicas(model):
    layout=getattr(model,'_atlas_layout',None)
    if layout is None or layout['replica_group'] is None:return
    group=layout['replica_group'];spec=layout['spec']
    signature=[(name,list(p.shape),str(p.dtype),p.requires_grad) for name,p in model.named_parameters()]
    digest=hashlib.sha256(json.dumps(signature).encode()).hexdigest();checks=[None]*spec['replicas']
    dist.all_gather_object(checks,digest,group=group)
    if len(set(checks))!=1:raise ValueError('Replica parameter shards disagree')
    src=layout['shard_rank']  # Corresponding shard on the first node.
    for p in model.parameters():
        if p.numel():dist.broadcast(p,src=src,group=group)
    for b in model.buffers():
        if b.numel():dist.broadcast(b,src=src,group=group)

@torch.no_grad()
def synchronize_replica_gradients(model,bucket_mb=64,verify_reference=False):
    layout=getattr(model,'_atlas_layout',None)
    if layout is None or layout['replica_group'] is None:return dict(replica_all_reduce_calls=0,gradient_payload_bytes=0)
    group=layout['replica_group'];replicas=layout['spec']['replicas']
    params=[p for p in model.parameters() if p.requires_grad and p.numel()]
    device=next(model.parameters()).device
    # Some conditions may leave a parameter unused on one replica. Average a
    # zero contribution there, but preserve None globally for AdamW semantics.
    present=torch.tensor([int(p.grad is not None) for p in params],device=device,dtype=torch.int32)
    dist.all_reduce(present,op=dist.ReduceOp.MAX,group=group)
    active=[]
    for p,used in zip(params,present.tolist()):
        if not used:continue
        if p.grad is None:p.grad=torch.zeros_like(p)
        if p.grad.dtype!=torch.float32 or not p.grad.is_contiguous():raise ValueError('Expected contiguous FP32 sharded gradients')
        active.append(p.grad.view(-1))
    limit=max(1,int(bucket_mb*1024**2)//4);pieces=[];size=0;calls=1;elements=0;reference_error=0.
    if verify_reference:
        count=(sum(v.numel() for v in active)+limit-1)//limit;counts=[None]*dist.get_world_size()
        dist.all_gather_object(counts,count)
        if len(set(counts))!=1:raise ValueError('Reference diagnostic needs equal bucket counts; use a larger bucket size for this test')
    def flush():
        nonlocal pieces,size,calls,elements,reference_error
        if not pieces:return
        bucket=torch.cat(pieces)
        if verify_reference:
            # Each local shard gets a separate lane. A WORLD sum of sparse
            # lanes is an independent reference for the replica-only mean,
            # using the identical real backward gradients, not a second run.
            reference=torch.zeros((layout['spec']['shard_size'],limit),device=device,dtype=torch.float32)
            reference[layout['shard_rank'],:size].copy_(bucket)
            dist.all_reduce(reference);reference.div_(replicas)
            expected=reference[layout['shard_rank'],:size]
        dist.all_reduce(bucket,group=group);bucket.div_(replicas)
        if verify_reference:
            reference_error=max(reference_error,float((bucket-expected).abs().max()))
            torch.testing.assert_close(bucket,expected,rtol=2e-6,atol=1e-8)
        offset=0
        for view in pieces:
            n=view.numel();view.copy_(bucket[offset:offset+n]);offset+=n
        calls+=1;elements+=size;pieces=[];size=0
    for gradient in active:
        offset=0
        while offset<gradient.numel():
            n=min(limit-size,gradient.numel()-offset);pieces.append(gradient[offset:offset+n]);size+=n;offset+=n
            if size==limit:flush()
    flush()
    return dict(replica_all_reduce_calls=calls,gradient_payload_bytes=elements*4,
                presence_payload_bytes=present.numel()*4,replicas=replicas,bucket_mb=bucket_mb,
                independent_world_reference_checked=verify_reference,
                reference_gradient_elements_checked=elements if verify_reference else 0,
                reference_max_absolute_error=reference_error if verify_reference else None)

@torch.no_grad()
def clip_global_grad_norm(model,max_norm):
    layout=getattr(model,'_atlas_layout',None)
    if layout is None or layout['replica_group'] is None:return model.clip_grad_norm_(max_norm)
    # FSDP1 iterates a set of original parameters when summing norms. Different
    # processes may sum in different orders. Use one canonical norm so matching
    # replica shards get bit-identical clipping, avoiding replica weight drift.
    norm=model.clip_grad_norm_(float('inf'))
    dist.broadcast(norm,src=0)
    if not torch.isfinite(norm):raise FloatingPointError('Non-finite global gradient norm')
    coefficient=(float(max_norm)/(norm+1e-6)).clamp(max=1.)
    for p in model.parameters():
        if p.grad is not None:p.grad.mul_(coefficient.to(p.grad.dtype))
    return norm
