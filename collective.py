# import os
import numpy
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp

from typing import Iterable, List, Optional, Tuple, Union
#from helper import _squeeze_list, _rank_in_group, _check_tensors_dtype, _check_tensors_size

# --- Collective communication "operations" ---
# scatter still has some issues
def scatter(tensor: torch.tensor, 
            rank: int,
            world_size: int
) -> torch.tensor:
    """
    Scatters a tensor on root rank(0) to all other ranks.
    Args:
        rank: rank that called the operation
        world_size: number of processes
    """
    group = dist.new_group(list(range(world_size))) # create group with all processes
    t = torch.empty(1)
    if rank == 0:
        tensor_list = list(torch.chunk(tensor, world_size))
        dist.scatter(t, scatter_list=tensor_list, src=0, group=group)
    else:
        dist.scatter(t, scatter_list=[], src=0, group=group)
    print(f"rank[{rank}] data = {t[0]}")
    #return tensor_list


def reduce(tensor: torch.tensor, 
           rank: int, 
           world_size: int, 
           op: torch.distributed.ReduceOp = ReduceOp.SUM
) -> torch.tensor:
    """
    Gathers the tensors and reduce them using an Operation such as SUM, placing the result on root rank.
    Args:
        rank: rank that called the process
        world_size: number of processes
        op: operation to be performed during reduce
    """
    print(f"rank{rank}(before) tensor: {tensor}")
    group = dist.new_group(list(range(world_size))) # create a group with all the processes
    # each call sends the current rank tensor to rank 0. This sums up the tensor on each call to the current tensor on rank 0
    work = dist.reduce(tensor, dst=0, op=op, group=group)
    # only rank 0 accumulates tensors 
    if rank == 0:
        print(f"rank[{rank}](after) data = {tensor}")
        return tensor, work
    return tensor


def gather(tensor: torch.tensor, 
           rank: int, 
           world_size: int
) -> torch.tensor:
    """
    Collect tensors from each device and gathers/concatenate them into root rank.
    Args:
        tensor: tensor of each process
        rank: each rank of the process group
        world_size: number of processes
    """
    print(f"rank[{rank}](before) tensor: {tensor}")
    group = dist.new_group(list(range(world_size))) # create a group with all the processes
    if rank == 0:
        # create an empty list which we'll use to hold the gathered values
        tensor_list = [torch.empty(1) for i in range(world_size)]
        # send all tensors from root rank(0) to the others
        work = dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    else:
        work = dist.gather(tensor, gather_list=[], dst=0, group=group)
    
    if rank == 0:
        print(f"rank[{rank}] data = {tensor_list}")
        return tensor_list, work
    return work


def broadcast(tensor: torch.tensor, 
              rank: int, 
              world_size: int
) -> torch.tensor:
    """
    Copies the tensor data from root rank and broadcast them across all processes.
    Args:
        tensor: tensor to be copied
        rank: process rank
        world_size: number of processes in process group
    """
    group = dist.new_group(list(range(world_size))) # create a group with all the processes
    print(f"rank[{rank}](before) tensor: {tensor}")
    if rank != 0:
        tensor = torch.empty(1)
    # sending all tensors to the others
    work = dist.broadcast(tensor, src=0, group=group)
    # all ranks will have the tensor from root rank(0)
    print(f"rank[{rank}] data = {tensor}")
    return tensor, work


def all_reduce(tensor: torch.tensor, 
               rank: int, 
               world_size: int, 
               op: torch.distributed.ReduceOp = ReduceOp.SUM
) -> torch.tensor:
    """
    Gathers the tensors and reduce them using an Operation such as SUM, placing the result on all ranks.
    Args:
        tensor: tensor of each rank to be reduced
        rank: rank of current process
        world_size: number of processes in the group
        op: operation to use in reduce
    """
    group = dist.new_group(list(range(world_size)))
    print(f"rank[{rank}](before) tensor: {tensor}")
    work = dist.all_reduce(tensor, op=op, group=group)
    print(f"rank[{rank}](after) tensor: {tensor}")
    return tensor


def all_gather(tensor: torch.tensor,
               rank: int, 
               world_size: int
) -> torch.tensor:
    """
    Collect tensors from each device and gathers/concatenate them into all ranks.
    Args:
        tensor: tensor to be gather of each rank
        rank: rank of the current process
        world_size: number of processes in the group
    """
    group = dist.new_group(list(range(world_size)))
    print(f"rank[{rank}](before) data = {tensor}")
    # create empty list to hold gathered values
    tensor_list = [torch.empty(1) for i in range(world_size)]
    # sending all tensors to the others
    work = dist.all_gather(tensor_list, tensor, group=group)
    # all ranks will have the same tensor list
    print(f"rank[{rank}](after) data = {tensor_list}")
    return tensor_list, work

def reduce_scatter(tensor: torch.tensor,
                   rank: int,
                   world_size: int,
                   op: torch.distributed.ReduceOp = ReduceOp.SUM
) -> torch.tensor:
    """
    Performs reduce, split the results, and then scatters them to each device
    Args:
        tensor: tensor of each rank to reduce
        rank: rank of the current process
        world_size: number of processes
        op: operation to apply during reduce. default = SUM
    """
    group = dist.new_group(list(range(world_size)))
    work = dist.reduce(tensor, dst=0, op=op, group=group)
    t = torch.empty(1)
    if rank == 0:
        tensor_list = list(torch.chunk(tensor, world_size))
        dist.scatter(t, scatter_list=tensor_list, src=0, group=group)
    else:
        dist.scatter(t, scatter_list=[], src=0, group=group)
    print(f"rank[{rank}] data = {t[0]}")


    reduced_tensor = reduce(tensor, rank, world_size)
    scattered_tensor = scatter(reduced_tensor, rank, world_size)


def barrier(rank: int, world_size: int):
    dist.barrier()




# TODO:
# scatter - OK
# reduce  - OK
# gather  - OK
# broadcast - OK
# all_reduce - OK
# all_gather - OK
# reduce_scatter
# barrier 
# ring all_reduce

# OBS:    
    # Questions:
    # 1.) Do I need to pass the rank and local_rank to perform scatter?
