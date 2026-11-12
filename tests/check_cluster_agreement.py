"""Distributed fail-closed tests: mismatched config and missing data, no updates."""
import argparse,json,os,sys
from pathlib import Path
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from common import write
from distributed_layout import verify_rank_agreement

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    ready=json.loads((a.dataset/'ready.json').read_text());cfg=dict(test='same_release_and_code',global_batch=128)
    verify_rank_agreement(cfg,ready,a.dataset)
    altered=dict(cfg)
    if dist.get_rank()==dist.get_world_size()-1:altered['global_batch']=64
    rejected=0
    for config,folder in [(altered,a.dataset),(cfg,a.dataset/'missing-on-leader')]:
        try:verify_rank_agreement(config,ready,folder)
        except ValueError:rejected+=1
    assert rejected==2
    dist.barrier()
    if dist.get_rank()==0:write(a.output,dict(passed=True,ranks=dist.get_world_size(),identical_inputs_accepted=True,mismatched_rank_config_rejected=True,missing_node_data_rejected=True,training_updates=0))
    dist.destroy_process_group()
if __name__=='__main__':main()
