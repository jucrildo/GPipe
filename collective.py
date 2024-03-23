# import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp

from typing import Iterable, List, Optional, Tuple, Union
from helper import _squeeze_list, _rank_in_group, _check_tensors_dtype, _check_tensors_size

# Collective communication "operations"
def do_scatter(rank: int, world_size: int) -> None:
    """
    Apply scatter operation
    Args:
        rank: rank that called the operation
        world_size: number of processes
    """
    group = dist.new_group(list(range(world_size))) # create group with all processes
    tensor = torch.empty(1)
    if rank == 0:
        tensor_list = [torch.tensor([i+1], dtype=torch.float32) for i in range(world_size)]
        dist.scatter(tensor, scatter_list=tensor_list, src=0, group=group)
    else:
        dist.scatter(tensor, scatter_list=[], src=0, group=group)
    print(f"rank[{rank}] data = {tensor[0]}")


def reduce(rank: int, world_size: int, op: ReduceOp = ReduceOp.SUM) -> None:
    """
    Apply reduce operation
    """
    group = dist.new_group(list(range(world_size))) # create a group with all the processes
    tensor = torch.ones(1)
    # sending all tensors to rank 0 and doing the operation on it
    dist.reduce(tensor, dst=0, op=op, group=group)
    # only rank 0 will have 4 tensors
    print(f"rank[{rank}] data = {tensor[0]}")


def gather(rank: int, world_size: int) -> None:
    """
    Apply gather operation
    """
    group = dist.new_group(list(range(world_size))) # create a group with all the processes
    tensor = torch.tensor([rank], dtype=torch.float32)
    # sending all tensors from rank 0 to others
    if rank == 0:
        # create an empty list which we'll use to hold the gathered values
        tensor_list = [torch.empty(1) for i in range(world_size)]
        dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    else:
        dist.gather(tensor, gather_list=[], dst=0, group=group)
    # only rank 0 will have the tensors from the other processes
    # tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])
    if rank == 0:
        print(f"rank[{rank}] data = {tensor_list}")


def broadcast(rank: int, world_size: int) -> None:
    """
    Apply broadcast operation
    """
    group = dist.new_group(list(range(world_size))) # create a group with all the processes
    if rank == 0:
        tensor = torch.tensor([rank], dtype=torch.float32)
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