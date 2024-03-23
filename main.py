import torch
import torch.multiprocessing as mp
from setup import setup_process_group, exit_process_group
from collective import scatter, reduce, gather, broadcast

if __name__ == "__main__":
    tensor = torch.tensor([10., 20., 30., 40.])
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=setup_process_group, args=("gloo", rank, size, scatter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()