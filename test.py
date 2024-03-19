import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collective import scatter, broadcast, reduce, setup_process_group

t1 = torch.arange(0, 20)
t1 = t1.view(4, 5)
print(t1)

# scatter 
tlist = scatter(t1, dim_split=0, world_size=4)
print(f"tlist length: {len(tlist)} \n {tlist}")
print(f"tlist[0] shape: {tlist[0].shape}")
"""
# broadcast
t3 = torch.arange(0, 4).view(2, 2)
print(f"t3: \n{t3}")
tlist3 = broadcast(t3, world_size=4)
print(f"tlist3.len: {len(tlist3)}\n{tlist3}")
print(f"tlist3[0]: {tlist3[0].shape}")
"""
import os
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # init proc group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def clean():
    dist.destroy_process_group()

def demo(rank, world_size):
    setup(rank, world_size)
    tensor = torch.ones(3, 3).to(rank)

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size), 
             nprocs=world_size,
             join=True)