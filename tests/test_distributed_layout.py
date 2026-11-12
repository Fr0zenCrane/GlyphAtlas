import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from distributed_layout import layout_spec

class LayoutTests(unittest.TestCase):
    def test_4x8_groups(self):
        s=layout_spec(32,8)
        self.assertEqual(s['mode'],'node_shard');self.assertEqual(s['replicas'],4)
        self.assertEqual(s['shard_ranks'],[list(range(i,i+8)) for i in [0,8,16,24]])
        self.assertEqual(s['replica_ranks'][3],[3,11,19,27])
        for groups in [s['shard_ranks'],s['replica_ranks']]:self.assertEqual(sorted(x for g in groups for x in g),list(range(32)))
    def test_single_node_stays_original(self):
        s=layout_spec(8,8);self.assertEqual(s['mode'],'world_full');self.assertEqual(s['replicas'],1)
    def test_invalid_topology_rejected(self):
        for args in [(30,8),(0,8),(8,0),(32,8,'wrong')]:
            with self.assertRaises(ValueError):layout_spec(*args)
    def test_fp64_gradient_mean_4x8_accum4(self):
        g=torch.Generator().manual_seed(7);x=torch.randn(128,7,generator=g,dtype=torch.float64);y=torch.randn(128,3,generator=g,dtype=torch.float64)
        w=torch.randn(7,3,generator=g,dtype=torch.float64,requires_grad=True)
        ((x@w-y)**2).mean().backward();expected=w.grad
        node_gradients=[]
        for node in range(4):
            rank_gradients=[]
            for local in range(8):
                v=w.detach().clone().requires_grad_(True)
                for micro in range(4):
                    idx=micro*32+node*8+local;(((x[idx]@v-y[idx])**2).mean()/4).backward()
                rank_gradients.append(v.grad)
            node_gradients.append(torch.stack(rank_gradients).mean(0))
        torch.testing.assert_close(torch.stack(node_gradients).mean(0),expected,rtol=1e-12,atol=1e-12)
if __name__=='__main__':unittest.main()
