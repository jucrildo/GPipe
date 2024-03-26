import torch
import torch.multiprocessing as mp
from setup import setup_group, destroy_group
from collective import scatter, reduce, gather, broadcast, all_reduce, all_gather

if __name__ == "__main__":
    #tensor = torch.tensor([10., 20., 30., 40.], dtype=torch.float32) # scatter_tensor
    tensor = torch.tensor(10, dtype=torch.float32)
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        # create an instance of the mp.Process
        p = mp.Process(target=setup_group, args=("gloo", rank, size, tensor*(rank+1), all_gather)) # eventually the call will be on training function
        # run the process
        p.start()
        # append each process
        processes.append(p)

    for p in processes:
        p.join() # wait for each process to finish

    destroy_group()