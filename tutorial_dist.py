import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
#from collective import scatter2

"""
    Great tutorial on blog https://blog.roboflow.com/collective-communication-distributed-systems-pytorch/
    to learn how to manage processes and ranks, and make the collective communication functions work. Now,
    implement those functions from scratch.
"""

def init_process(rank: int, size: int, backend="gloo"):
    """Initialize distributed environment"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    #fn(rank, size)

def gather(tensor, root_rank):
    gathered_tensors = [torch.empty(1) for _ in range(dist.get_world_size())]
    dist.gather(tensor, gathered_tensors, root_rank)
    if dist.get_rank() == root_rank:
        return gathered_tensors[root_rank]



if __name__ == "__main__":
    world_size = 4
    init_process(rank=dist.get_rank(), size=world_size)

    # tensor to be gathered
    tensor = torch.tensor([1], dtype=torch.float32)

    # lauch multiple processes
    processes = [mp.Process(target=gather, args=(tensor, 0)) for _ in range(world_size)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    if dist.get_rank() == 0:
        print(f"gathered tensor: {gathered_tensor}")



"""
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
"""