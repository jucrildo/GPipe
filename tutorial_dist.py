import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim


#from collective import scatter2

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}")
    setup(rank, world_size)
    
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)







"""
    Great tutorial on blog https://blog.roboflow.com/collective-communication-distributed-systems-pytorch/
    to learn how to manage processes and ranks, and make the collective communication functions work. Now,
    implement those functions from scratch.
"""

"""

def do_gather(rank, world_size):
    group = dist.new_group(list(range(world_size)))
    tensor = torch.tensor([10], dtype=torch.float32)
    if rank == 0:
        tensor_list = [torch.empty(1) for i in range(world_size)]
        dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    else:
        dist.gather(tensor, gather_list=[], dst=0, group=group)

    if rank == 0:
        print(f"rank[{rank}] data = {tensor_list}")
"""
"""
def init_process(rank: int, size: int, fn, backend="gloo"):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23335"
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_gather))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    

    # lauch multiple processes
    processes = [mp.Process(target=gather, args=(tensor, 0)) for _ in range(world_size)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    if dist.get_rank() == 0:
        print(f"gathered tensor: {gathered_tensor}")
"""


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