import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from typing import Iterable, List, Optional, Tuple, Union
from helper import _check_tensors_dtype, _check_tensors_size, _squeeze_list

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
    #source_rank: int = 0, 
    world_size: int = 1,
    #process_grp: Optional[int] = None,
    ) -> List[torch.tensor]: #Tuple[torch.tensor]:
    """
    Scatters the tensor across all devices
    Args:
        tensor: tensor to scatter #output tensor
        dim_split: which dimension to split the tensor
        source_rank: rank that called ("rank" is an identifier of each process)
        world_size: total number of processes
    """
    # Pipegoose scatter returns only 1 tensor. The one that is scattered to local_rank. "tensor_list[rank]"
    #
    #if not _check_tensors_dtype(tensor_list, tensor_dtype=tensor_list[0].dtype):
    #    raise TypeError("Tensors dtypes are different")
    #
    #if not _check_tensors_size(tensor_list):
    #    raise Exception("Tensors are not the same size!")  
    
    if dim_split >= len(tensor.shape) or dim_split < 0:
        raise ValueError("Unavailable dimmension")

    if world_size == 1: # there's only one process
        return tensor
    
    assert tensor.shape[dim_split] % world_size == 0 

    #rank = dist.get_rank()
    #world_size = dist.get_world_size()
    tensor_list = _squeeze_list(torch.chunk(tensor, world_size, dim=dim_split)) # torch.chunk() returns list of Tensors
    #tensor_list = _squeeze_list(tensor_list)

    return tensor_list




if __name__ == "__main__":
    world_size = torch.cuda.device_count() # how many GPUs there's on the machine
    # takes a function and spawns that across all of our processes in the process group
    mp.spawn(args=(world_size), nprocs=world_size)
    pass