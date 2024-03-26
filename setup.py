import os
import torch
import torch.distributed as dist

# 1st: initialize process group
def setup_group(backend: str, rank: int, world_size: int, tensor: torch.tensor, function) -> None:
    """
    Args:
        backend: chosen backend ("nccl", "gloo")
        rank: identifier of each process
        world_size: total number of processes
        fn: distributed function that will be called?
    """
    os.environ["MASTER_ADDR"] = "localhost" # IP of the machine running rank0. Or could also use 127.0.0.1
    os.environ["MASTER_PORT"] = "29500" # port of the process
    # initializes the default distributed process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    function(tensor, rank, world_size) # in the tutorial, we're calling the function 

# after finishes distributed training
def destroy_group():
    dist.destroy_process_group()