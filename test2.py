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
def run(rank, world_size):
    tensor = torch.ones(1)
    print(f"tensor before: {tensor}")
    group = dist.new_group([0, 1]) # list of ranks
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"Rank: {rank} has data {tensor}")

def scatter(world_size, dim):
    rank = dist.get_rank()
    #torch.cuda.set_device(rank)
    output = torch.zeros(1)
    print(f"before: {rank}: {output}\n")
    if rank == 0:
        inputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
        inputs = output.split(inputs, dim=dim, split_size_or_sections=1)
        dist.scatter(output, scatter_list=list(inputs), src=rank)
    else:
        dist.scatter(output, src=rank)
    print(f"after {rank}: {output}\n")

def init_proc(rank, world_size, func, backend="gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    func(world_size, dim=0)#, tensor)

if __name__ == "__main__":
    world_sz = 4
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
