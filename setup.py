import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 1st: initialize process group
def setup_process_group(backend: str, rank: int, world_size: int, function, ) -> None:
    """
    Args:
        backend: chosen backend ("nccl", "gloo")
        rank: identifier of each process
        world_size: total number of processes
        fn: distributed function that will be called?
    """
    os.environ["MASTER_ADDR"] = "localhost" # IP address of the machine running rank0
    os.environ["MASTER_PORT"] = "29500"

    # backend = "nccl" if torch.cuda.is_available() else "gloo"
    # initializes the default distributed process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    function(rank, world_size) # in the tutorial, we're calling the function 

# after finishes distributed training
def exit_process_group() -> None:
    dist.destroy_process_group()