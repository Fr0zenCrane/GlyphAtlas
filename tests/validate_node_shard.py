"""Real eight-GPU numerical/checkpoint tests; only a tiny model is updated.

Two logical nodes of four GPUs on one host exercise replica communication.
This is not a measurement of physical multi-node/H200 networking.
"""
import os,sys,argparse,copy,json,gc,contextlib,inspect,collections
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from common import write
from distributed_layout import create_layout,attach_layout,synchronize_initial_replicas,synchronize_replica_gradients,clip_global_grad_norm
from dataset import UpdateSampler
from train_full import save_checkpoint,restore_checkpoint

class Block(torch.nn.Module):
    def __init__(self):super().__init__();self.lin=torch.nn.Linear(24,24)
    def forward(self,x):return torch.nn.functional.silu(self.lin(x))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input=torch.nn.Linear(12,24);self.block=Block();self.output=torch.nn.Linear(24,3);self.conditional=torch.nn.Linear(24,3);self.unused=torch.nn.Parameter(torch.ones(5))
    def forward(self,x,condition):
        h=self.block(self.input(x));out=self.output(h)
        if condition:out=out+self.conditional(h)
        return out

def gather(model,shards):
    local={name:dict(parameter=p.detach().float().reshape(-1).cpu(),gradient=p.grad.detach().float().reshape(-1).cpu() if p.grad is not None else None) for name,p in model.named_parameters()}
    all_ranks=[None]*dist.get_world_size();dist.all_gather_object(all_ranks,local)
    result=[]
    for start in range(0,len(all_ranks),shards):
        group=all_ranks[start:start+shards];full={}
        for name in local:
            values=[r[name] for r in group];full[name]=dict(parameter=torch.cat([v['parameter'] for v in values]))
            full[name]['gradient']=None if all(v['gradient'] is None for v in values) else torch.cat([v['gradient'] if v['gradient'] is not None else torch.zeros_like(v['parameter']) for v in values])
        result.append(full)
    # All logical nodes must have the same complete parameters and gradients.
    for other in result[1:]:
        for name,fields in result[0].items():
            for key,want in fields.items():
                actual=other[name][key]
                if want is None:assert actual is None,(name,key)
                else:torch.testing.assert_close(actual,want,rtol=0,atol=0)
    return result[0]

def run_case(mode,dtype,initial,x,y,output,two_stage=False):
    rank=dist.get_rank();world=dist.get_world_size();local=int(os.environ['LOCAL_RANK'])
    layout=create_layout(mode,4 if mode=='node_shard' else 8,validate_hosts=False)
    net=Model();net.load_state_dict(initial);net.cuda(local)
    model=FSDP(net,process_group=layout['shard_group'],device_id=local,use_orig_params=True,sync_module_states=True,
        auto_wrap_policy=ModuleWrapPolicy({Block}),mixed_precision=MixedPrecision(param_dtype=dtype,reduce_dtype=torch.float32,buffer_dtype=dtype))
    attach_layout(model,layout);synchronize_initial_replicas(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)
    sampler=UpdateSampler(256 if two_stage else 1024,rank,world);history=[];comm=[];collectives=[]
    cfg=dict(global_batch=128)
    if two_stage:
        import importlib.util
        spec=importlib.util.spec_from_file_location('schedule_trainer',ROOT/'runtime/train.py')
        training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
        cfg=json.loads((ROOT/'configs/legacy/continuous_resolution_schedule.json').read_text())
        cfg.update(updates_per_epoch=2,steps=4)
    originals={name:getattr(dist,name) for name in ['all_gather_into_tensor','reduce_scatter_tensor','all_reduce']}
    for name,fn in originals.items():
        signature=inspect.signature(fn)
        def wrapper(*args,_fn=fn,_name=name,_signature=signature,**kwargs):
            pg=_signature.bind_partial(*args,**kwargs).arguments.get('group');collectives.append(dict(operation=_name,group_size=dist.get_world_size(pg)))
            return _fn(*args,**kwargs)
        setattr(dist,name,wrapper)
    try:
        for step in range(3):
            if two_stage:
                for group in optimizer.param_groups:group['lr']=training.learning_rate(step+1,cfg)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(128//world):
                i=micro*world+rank
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
                    prediction=model(x[step,i:i+1],i%world<4)
                    loss=(prediction.float()-y[step,i:i+1]).square().mean()/(128//world)
                loss.backward()
            stats=synchronize_replica_gradients(model,bucket_mb=1);comm.append(stats)
            norm=clip_global_grad_norm(model,.5);assert torch.isfinite(norm)
            gradient=gather(model,layout['spec']['shard_size'])
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            parameters=gather(model,layout['spec']['shard_size'])
            history.append(dict(norm=float(norm),gradient=gradient,parameters=parameters))
            if step==1 and mode=='node_shard':
                expected=[p.detach().clone() for p in model.parameters()]
                moments=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
                folder=output/('checkpoint_'+str(dtype).split('.')[-1]);folder.mkdir(parents=True,exist_ok=True);dist.barrier()
                cp=save_checkpoint(model,optimizer,sampler,folder,2,cfg,export=False)
                with torch.no_grad():
                    for p in model.parameters():p.add_(3)
                for state in optimizer.state.values():
                    for key in ['exp_avg','exp_avg_sq']:state[key].zero_()
                torch.manual_seed(999)
                assert restore_checkpoint(model,optimizer,sampler,cp)==2
                assert torch.equal(torch.get_rng_state(),rng) and torch.equal(torch.cuda.get_rng_state(),cuda_rng)
                for p,want in zip(model.parameters(),expected):torch.testing.assert_close(p,want,rtol=0,atol=0)
                actual=optimizer.state_dict();assert actual['param_groups']==moments['param_groups']
                for key,s in actual['state'].items():
                    for field,t in s.items():torch.testing.assert_close(t,moments['state'][key][field],rtol=0,atol=0)
                assert sampler.start_step==2
                if two_stage:
                    assert next(iter(sampler))[1]==1
                    assert training.stage_for_step(3,cfg)==(2,'random')
                    assert all(group['lr']==5e-6 for group in optimizer.param_groups)
    finally:
        for name,fn in originals.items():setattr(dist,name,fn)
    if mode=='node_shard':
        assert all(c['group_size']==4 for c in collectives if c['operation'] in ('all_gather_into_tensor','reduce_scatter_tensor')),collections.Counter((c['operation'],c['group_size']) for c in collectives)
        assert all(c['replica_all_reduce_calls']==2 for c in comm),comm
    del model,optimizer,net;gc.collect();torch.cuda.empty_cache();dist.barrier()
    return history,comm

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--two-stage-test',action='store_true');a=ap.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2);dist.init_process_group('nccl')
    assert dist.get_world_size()==8;torch.manual_seed(42);initial=copy.deepcopy(Model().state_dict())
    generator=torch.Generator().manual_seed(987);x=torch.randn(3,128,12,generator=generator).cuda();y=torch.randn(3,128,3,generator=generator).cuda();receipts=[]
    for dtype in [torch.float32,torch.bfloat16]:
        baseline,_=run_case('world_full',dtype,initial,x,y,a.output,a.two_stage_test)
        optimized,comm=run_case('node_shard',dtype,initial,x,y,a.output,a.two_stage_test)
        max_grad=max_weight=max_norm=0.
        for ref,actual in zip(baseline,optimized):
            max_norm=max(max_norm,abs(ref['norm']-actual['norm']))
            for phase,field in [('gradient','gradient'),('parameters','parameter')]:
                for name,value in ref[phase].items():
                    want=value[field];got=actual[phase][name][field]
                    if want is None:assert got is None,(phase,name)
                    else:
                        torch.testing.assert_close(got,want,rtol=2e-5,atol=2e-7)
                        error=float((got-want).abs().max())
                        if phase=='gradient':max_grad=max(max_grad,error)
                        else:max_weight=max(max_weight,error)
        receipts.append(dict(dtype=str(dtype),passed=True,max_gradient_abs_error=max_grad,max_parameter_abs_error=max_weight,max_norm_abs_error=max_norm,communication=comm,optimizer='fused AdamW',optimizer_resume_and_rng_exact=True,updates=3))
    write(a.output/f'rank{dist.get_rank()}.json',dict(passed=True,tests=receipts,flux_optimizer_updates=0))
    dist.barrier()
    if dist.get_rank()==0:write(a.output/'validation.json',dict(passed=True,two_stage_boundary_resume=a.two_stage_test,world_size=8,virtual_nodes=2,shards_per_virtual_node=4,tests=receipts,flux_optimizer_updates=0,physical_multinode_tested=False,scope='Nested FSDP, FP32/BF16, global128, replica-only conditional parameter, globally unused parameter, clipping, AdamW, checkpoint restore then next update'))
    dist.destroy_process_group()
if __name__=='__main__':main()
