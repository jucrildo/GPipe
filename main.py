import torch
import torch.multiprocessing as mp
from setup import setup_group, destroy_group
from collective import scatter, reduce, gather, broadcast

if __name__ == "__main__":
    #tensor = torch.tensor([10., 20., 30., 40.], dtype=torch.float32) # scatter_tensor
    tensor = torch.tensor(10, dtype=torch.float32) # reduce tensor
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=setup_group, args=("gloo", rank, size, tensor*rank, gather))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

   # destroy_group()