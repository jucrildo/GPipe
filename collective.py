import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp

from typing import Iterable, List, Optional, Tuple, Union
from helper import _squeeze_list, _rank_in_group

# type ProcessType = dict[int | str, set[str]] # like typedef in C

# 1st: initialize process group
def setup_process_group(backend:str, rank: int, world_size: int) -> None:
    """
    Args:
        rank: identifier of each process
        world_size: total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost" # IP address of the machine running rank0
    os.environ["MASTER_PORT"] = "23500"

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # initializes the default distributed process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

# after finishes distributed training
def exit_process_group():
    dist.destroy_process_group()

# Collective communication "operations"
def scatter(
    tensor: torch.tensor,
    dim_split: int = 0, 
    #rank: int = 0, 
    world_size: int = 1,
) -> List[torch.tensor]:
    """
    Scatters the tensor across all devices
    Args:
        tensor: tensor to scatter #output tensor
        dim_split: which dimension to split the tensor
        rank: rank that called ("rank" is an identifier of each process)
        world_size: total number of processes
    """
    # Pipegoose scatter returns only 1 tensor. The one that is scattered to local_rank. "tensor_list[rank]"  
    rank = dist.get_rank() # process ID
    local_rank = torch.cuda.current_device() # GPU id
    world_size = dist.get_world_size()
    if dim_split >= len(tensor.shape) or dim_split < 0:
        raise ValueError("Unavailable dimmension")
    
    if world_size == 1: # there's only one process
        return tensor
    
    assert tensor.shape[dim_split] % world_size == 0 
    tensor_list = _squeeze_list(torch.chunk(tensor, world_size, dim=dim_split)) # torch.chunk() returns list of Tensors

    return tensor_list #, tensor_list[rank]

# still need a lot of changes. Now that I understood how ranks and processes groups work, remake this.
def broadcast(tensor: torch.tensor, world_size: int = 1) -> List[torch.tensor]:
    """
    Copies the data across all devices.
    Args:
        tensor: tensor to copy
        world_size: total number of processes to be copies
    """
    if world_size == 1:
        return tensor
    
    rank = dist.get_rank()

    tensor_list = []
    for _ in range(world_size):
        tensor_list.append(tensor)
    
    return tensor_list
    
# remake it now that I understand how ranks and process groups work
# When I call reduce, should reduce into the first process of the process group?
def reduce(
    tensor_list: List[torch.tensor],
    target_rank: int,
    process_group: List[int],
    op: ReduceOp = ReduceOp.SUM,
) -> torch.tensor:
    """
    Reduces the tensors with an operation into one rank.
    Args:
        tensor_list: tensor_list made by appending all tensors from all ranks 
        target_rank: the destination rank to reduce the tensors to
        process_group: process group
        op: operation used to reduce the tensors
    """
    if not isinstance(tensor_list):
        raise TypeError("Must be a list of tensors")

    if not _rank_in_group(target_rank, process_group):
        raise Exception("target_rank is not in process_group")

    tensor = [t.op for t in tensor_list]
    process_group[target_rank] = tensor

    return tensor


if __name__ == "__main__":
    world_size = torch.cuda.device_count() # how many GPUs there's on the machine
    # takes a function and spawns that across all of our processes in the process group
    mp.spawn(args=(world_size), nprocs=world_size)
    pass


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