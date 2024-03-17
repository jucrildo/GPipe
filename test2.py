import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
"""
def scatter(tensor, dim_split, world_size):
    rank = dist.get_rank()
    #world_size = dist.get_world_size()
    assert tensor.shape[dim_split] % world_size == 0 
    tensor_list = torch.chunk(tensor, world_size, dim=dim_split)
    return tensor_list, tensor_list[rank]

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    print(f"Worker: {rank} started")
    t = torch.ones(4, 4)
    tlist, trank = scatter(t, 0, world_size=world_size)
    print(f"Worker: {rank} output: {trank}")
"""
def run(rank, size, tensor):
    if rank == 0:
        tensor = tensor + 1
        dist.send(tensor=tensor, dst=1)
    else:
        dist.recv(tensor=tensor, src=0)
    print(f"Rank: {rank} has data {tensor[0]}")

def scatter(tensor, rank, world_size, dim):
    assert tensor[dim] % world_size == 0
    t = torch.chunk(tensor, world_size, dim=dim)
    return t[rank]

def init_proc(rank, world_size, func, backend="gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    tensor = torch.ones(4, 4)
    func(rank, world_size, tensor)

if __name__ == "__main__":
    world_sz = 4 # 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_sz):
        p = mp.Process(target=init_proc, args=(rank, world_sz, scatter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

#world_sz = 4
#rank = 0
#mp.spawn(setup, args=(world_sz,), nprocs=world_sz, join=True)
