import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collective import scatter2

"""
    Great tutorial on blog https://blog.roboflow.com/collective-communication-distributed-systems-pytorch/
    to learn how to manage processes and ranks, and make the collective communication functions work. Now,
    implement those functions from scratch.
"""

def init_process(rank: int, size: int, fn, backend="gloo"):
    """Initialize distributed environment"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    fn(rank, size)

def do_reduce(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    # sending all tensors to rank 0 and sum them
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    # only rank 0 will have 4
    print(f"rank[{rank}] data = {tensor[0]}")

def do_scatter(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.empty(1)
    # sending all tensors from rank 0 to the others
    if rank == 0:
        tensor_list = [torch.tensor([i + 1], dtype=torch.float32) for i in range(size)] # tensor_list = [tensor(1), tensor(2), tensor(3), tensor(4)]
        dist.scatter(tensor, scatter_list=tensor_list, src=0, group=group)
    else:
        dist.scatter(tensor, scatter_list=[], src=0, group=group)
    print(f"rank[{rank}] data = {tensor[0]}")

def do_gather(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.tensor([rank], dtype=torch.float32)
    if rank == 0:
        tensor_list = [torch.empty(1) for i in range(size)]
        dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    else:
        dist.gather(tensor, gather_list=[], dst=0, group=group)
    if rank == 0:
        print(f"rank[{rank}] data = {tensor_list}")

def do_broadcast(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    if rank == 0:
        tensor = torch.tensor([rank], dtype=torch.float32)
    else:
        tensor = torch.empty(1)
    dist.broadcast(tensor, src=0, group=group)
    # all ranks have tensor([0.]) from rank 0
    print(f"rank[{rank}] data = {tensor}")

def hello(rank: int, size: int):
    # each process will call this functions
    print(f"rank[{rank}] say hi!")

if __name__ == "__main__":
    size = 6
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, scatter2))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()