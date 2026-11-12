"""Eight-rank SMALL synthetic-model checkpoint test, never updates FLUX weights."""
import argparse,json,os,sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('GLYPH_OMNI_ROOT',str(ROOT))
sys.path.insert(0,str(ROOT/'runtime'))
from common import write
from train_full import save_checkpoint,restore_checkpoint
from dataset import UpdateSampler

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    world,rank=int(os.environ['WORLD_SIZE']),int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl');torch.manual_seed(42)
    model=FSDP(torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.SiLU(),torch.nn.Linear(32,4)).cuda(),use_orig_params=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,fused=True)
    model(torch.randn(8,16,device='cuda')).square().mean().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    expected=[p.detach().clone() for p in model.parameters()]
    expected_moments={p:{k:v.clone() if torch.is_tensor(v) else v for k,v in s.items()} for p,s in optimizer.state.items()}
    sampler=UpdateSampler(1000,rank,world);out=a.output.resolve()
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    dist.barrier();rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    cp=save_checkpoint(model,optimizer,sampler,out,1,dict(global_batch=128),export=False)
    with torch.no_grad():
        for parameter in model.parameters():parameter.add_(42)
    for state in optimizer.state.values():
        for key in ('exp_avg','exp_avg_sq'):state[key].zero_()
    torch.manual_seed(123)
    assert restore_checkpoint(model,optimizer,sampler,cp)==1
    assert torch.equal(torch.get_rng_state(),rng) and torch.equal(torch.cuda.get_rng_state(),cuda_rng)
    for actual,want in zip(model.parameters(),expected):assert torch.equal(actual,want)
    for parameter,state in optimizer.state.items():
        for key in ('step','exp_avg','exp_avg_sq'):assert torch.equal(state[key].cpu(),expected_moments[parameter][key].cpu())
    assert sampler.start_step==1
    write(out/f'rank{rank}.json',dict(passed=True,model='tiny synthetic MLP',flux_optimizer_updates=0,fp32_adam=True,rng_exact=True,bs128_cursor=True))
    dist.barrier()
    if rank==0:write(out/'validation.json',dict(passed=True,ranks=world,model='tiny synthetic MLP',flux_optimizer_updates=0,scope='Shared FSDP DCP save/restore helper and BS128 sampler; not a full FLUX checkpoint roundtrip'))
    dist.destroy_process_group()

if __name__=='__main__':main()
