# import os
import numpy
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp

from typing import Iterable, List, Optional, Tuple, Union
from helper import _squeeze_list, _rank_in_group, _check_tensors_dtype, _check_tensors_size

# Collective communication "operations"
def scatter(tensor: torch.tensor, rank: int, world_size: int) -> None:
    """
    Scatters a tensor on rank 0 to all other ranks.
    Args:
        rank: rank that called the operation
        world_size: number of processes
    """
    group = dist.new_group(list(range(world_size))) # create group with all processes
    t = torch.empty(1)
    if rank == 0:
        tensor_list = list(torch.chunk(tensor, world_size))
        print(f"tensor_list: {tensor_list}, {tensor_list[0].dtype}")
        dist.scatter(t, scatter_list=tensor_list, src=0, group=group)
    else:
        dist.scatter(t, scatter_list=[], src=0, group=group)
    print(f"rank[{rank}] data = {t}")

def reduce(tensor: torch.tensor, 
           rank: int, 
           world_size: int, 
           op: torch.distributed.ReduceOp = ReduceOp.SUM) -> None:
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
    dist.reduce(tensor, dst=0, op=op, group=group)
    # only rank 0 accumulated tensors 
    print(f"rank[{rank}](after) data = {tensor}")

def gather(tensor: torch.tensor, rank: int, world_size: int) -> None:
    """
    Collect tensors from each device and gathers/concatenate them into root rank.
    Args:
        tensor: tensor of each process
        rank: each rank of the process group
        world_size: number of processes
    """
    group = dist.new_group(list(range(world_size))) # create a group with all the processes
    #tensor_gathers = torch.tensor([rank], dtype=torch.float32)
    print(f"rank[{rank}](before) tensor: {tensor}")
    # sending all tensors from rank 0 to others
    #if rank == 0:
        # create an empty list which we'll use to hold the gathered values
    tensor_list = [torch.empty(i) for i in range(world_size)]
    dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    #else:
        #dist.gather(tensor, gather_list=[], dst=0, group=group)

    # only rank 0 will have the tensors from the other processes
    # tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])
    if rank == 0:
        print(f"rank[{rank}](after) data = {tensor_list}")


def broadcast(tensor: torch.tensor, rank: int, world_size: int) -> None:
    """
    Apply broadcast operation
    """
    group = dist.new_group(list(range(world_size))) # create a group with all the processes
    print(f"rank[{rank}](before) tensor: {tensor}")
    if rank == 0:
        tensor = torch.tensor(10, dtype=torch.float32)
        #tensor = torch.tensor([rank], dtype=torch.float32)
    else:
        tensor = torch.empty(1)
        # sending all tensors to the others
    dist.broadcast(tensor, src=0, group=group)
    # all ranks will have tensor([0.]) from rank 0
    print(f"rank[{rank}] data = {tensor}")






# OBS:
    # Questions:
    # 1.) Do I need to pass the rank and local_rank to perform scatter?
    #
    #if not _check_tensors_dtype(tensor_list, tensor_dtype=tensor_list[0].dtype):
    #    raise TypeError("Tensors dtypes are different")
    #
    #if not _check_tensors_size(tensor_list):
    #    raise Exception("Tensors are not the same size!")
    #
    #