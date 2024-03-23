import torch
import torch.multiprocessing as mp
from setup import setup_process_group, exit_process_group
from collective import do_scatter, reduce, gather, broadcast

if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=setup_process_group, args=("gloo", rank, size, do_scatter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()